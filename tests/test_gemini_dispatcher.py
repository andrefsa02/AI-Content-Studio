import base64
import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from google.api_core import exceptions as google_exceptions
from google.rpc.error_details_pb2 import ErrorInfo
from google.rpc.error_details_pb2 import RetryInfo
from google.protobuf.duration_pb2 import Duration

from api_clients import (
    GEMINI_ALL_TEXT_MODELS_QUOTA_EXHAUSTED,
    GEMINI_TEXT_MODEL,
    GEMINI_TEXT_MODEL_FALLBACK_CHAIN,
    GEMINI_TEXT_MODEL_OPERATION_ALLOWLIST,
    GEMINI_TTS_MODEL,
    NANO_BANANA_IMAGE_MODEL,
    MODEL_DAILY_QUOTA,
    MODEL_MINUTE_QUOTA,
    NON_QUOTA_ERROR,
    PROJECT_DAILY_QUOTA,
    GoogleClient,
    GeminiRequestDispatcher,
    classify_gemini_error,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def sleep(self, seconds):
        with self.lock:
            self.sleeps.append(seconds)
            self.value += seconds


class ImmediateDispatcher:
    def __init__(self):
        self.calls = 0

    def dispatch(self, request):
        self.calls += 1
        return request()


class GeminiDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.fake_time = FakeClock()
        self.dispatcher = GeminiRequestDispatcher(clock=self.fake_time, sleep=self.fake_time.sleep)
        self.immediate_dispatcher = ImmediateDispatcher()
        self.dispatcher_patch = patch("api_clients.gemini_request_dispatcher", self.immediate_dispatcher)
        self.dispatcher_patch.start()

    def tearDown(self):
        self.dispatcher_patch.stop()

    @staticmethod
    def quota_error(quota_id, dimensions=None, message="quota exhausted"):
        metadata = {
            "quota_id": quota_id,
            "quota_metric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
        }
        if dimensions is not None:
            metadata["quota_dimensions"] = json.dumps(dimensions)
        info = ErrorInfo(reason="RATE_LIMIT_EXCEEDED", domain="googleapis.com", metadata=metadata)
        return google_exceptions.ResourceExhausted(message, error_info=info)

    def make_text_client(self, model_behaviors):
        client = GoogleClient.__new__(GoogleClient)
        client.config = {"TEXT_ENGINE": "Gemini API"}
        client.safety_settings = {}
        models = {}
        for model_name in GEMINI_TEXT_MODEL_FALLBACK_CHAIN:
            model = Mock()
            behavior = model_behaviors.get(model_name, Mock(text=f"result from {model_name}"))
            if isinstance(behavior, list):
                model.generate_content.side_effect = behavior
            elif isinstance(behavior, BaseException):
                def raise_error(*args, error=behavior, **kwargs):
                    raise error
                model.generate_content.side_effect = raise_error
            else:
                model.generate_content.return_value = behavior
            models[model_name] = model
        client.text_model = models[GEMINI_TEXT_MODEL]
        client._text_model_for = Mock(side_effect=lambda model_name: models[model_name])
        return client, models

    def use_limiter(self, max_attempts=5):
        return GeminiRequestDispatcher(
            max_attempts=max_attempts,
            clock=self.fake_time,
            sleep=self.fake_time.sleep,
        )

    def test_sequential_requests_under_limit_are_accepted(self):
        results = [self.dispatcher.dispatch(lambda index=index: index) for index in range(4)]
        self.assertEqual(results, [0, 1, 2, 3])
        self.assertEqual(self.fake_time.sleeps, [])

    def test_more_than_four_requests_in_window_are_delayed(self):
        for _ in range(5):
            self.dispatcher.dispatch(lambda: "ok")
        self.assertEqual(self.fake_time.sleeps, [60.0])

    def test_concurrent_callers_are_serialized_and_rate_limited(self):
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        start_times = []
        failures = []

        def request():
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
                start_times.append(self.fake_time())
            time.sleep(0.005)
            with active_lock:
                active -= 1

        def caller():
            try:
                self.dispatcher.dispatch(request)
            except Exception as error:  # Captured for assertion on the main thread.
                failures.append(error)

        threads = [threading.Thread(target=caller) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(len(start_times), 6)
        self.assertEqual(max_active, 1)
        for index, start in enumerate(start_times):
            in_window = [other for other in start_times if start <= other < start + 60]
            self.assertLessEqual(len(in_window), 4)

    def test_http_429_is_retried_through_dispatcher(self):
        response = Mock(status_code=429, headers={}, text="rate limited")
        error = requests.exceptions.HTTPError(response=response)
        attempts = []

        def request():
            attempts.append(self.fake_time())
            if len(attempts) == 1:
                raise error
            return "success"

        self.assertEqual(self.dispatcher.dispatch(request), "success")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.fake_time.sleeps, [1.0])

    def test_provider_retry_delay_is_respected(self):
        response = Mock(status_code=429, headers={"Retry-After": "7"}, text="rate limited")
        error = requests.exceptions.HTTPError(response=response)
        attempts = []

        def request():
            attempts.append(True)
            if len(attempts) == 1:
                raise error
            return "success"

        self.assertEqual(self.dispatcher.dispatch(request), "success")
        self.assertEqual(self.fake_time.sleeps, [7.0])

    def test_rest_retry_info_delay_is_respected(self):
        response = Mock(status_code=429, headers={}, text="rate limited")
        response.json.return_value = {
            "error": {
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": {"seconds": 4, "nanos": 250_000_000},
                }]
            }
        }
        error = requests.exceptions.HTTPError(response=response)
        attempts = []

        def request():
            attempts.append(True)
            if len(attempts) == 1:
                raise error
            return "success"

        self.assertEqual(self.dispatcher.dispatch(request), "success")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.fake_time.sleeps, [4.25])

    def test_resource_exhausted_retry_delay_is_respected(self):
        error = google_exceptions.ResourceExhausted("Please retry in 2.5s")
        attempts = []

        def request():
            attempts.append(True)
            if len(attempts) == 1:
                raise error
            return "success"

        self.assertEqual(self.dispatcher.dispatch(request), "success")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.fake_time.sleeps, [2.5])

    def test_structured_retry_info_delay_is_respected(self):
        error = google_exceptions.ResourceExhausted(
            "quota exhausted",
            details=[RetryInfo(retry_delay=Duration(seconds=3, nanos=250_000_000))],
        )
        attempts = []

        def request():
            attempts.append(True)
            if len(attempts) == 1:
                raise error
            return "success"

        self.assertEqual(self.dispatcher.dispatch(request), "success")
        self.assertEqual(self.fake_time.sleeps, [3.25])

    def test_unrelated_exceptions_are_not_retried(self):
        attempts = []

        def request():
            attempts.append(True)
            raise ValueError("invalid request")

        with self.assertRaisesRegex(ValueError, "invalid request"):
            self.dispatcher.dispatch(request)
        self.assertEqual(len(attempts), 1)

    def test_generate_text_uses_dispatcher_and_disables_sdk_retries(self):
        client = GoogleClient.__new__(GoogleClient)
        client.config = {"TEXT_ENGINE": "Gemini API"}
        client.text_model = Mock()
        client.text_model.generate_content.return_value.text = "generated"

        with patch("api_clients.gemini_request_dispatcher", self.immediate_dispatcher):
            result = client._generate_text("prompt")

        self.assertEqual(result, "generated")
        self.assertEqual(self.immediate_dispatcher.calls, 1)
        client.text_model.generate_content.assert_called_once_with(
            "prompt", request_options={"retry": None}
        )

    def test_primary_text_model_succeeds_without_fallback(self):
        client, models = self.make_text_client({GEMINI_TEXT_MODEL: Mock(text="primary result")})
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter()):
            self.assertEqual(client._generate_text("prompt"), "primary result")
        models[GEMINI_TEXT_MODEL].generate_content.assert_called_once()
        self.assertEqual(client.last_text_request, {
            "requested_model": GEMINI_TEXT_MODEL,
            "effective_model": GEMINI_TEXT_MODEL,
            "fallback_used": False,
        })

    def test_daily_model_quota_falls_back_once_to_next_model(self):
        fallback = GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1]
        client, models = self.make_text_client({
            GEMINI_TEXT_MODEL: self.quota_error("GenerateRequestsPerDayPerModel-FreeTier"),
            fallback: Mock(text="fallback result"),
        })
        limiter = self.use_limiter()
        with patch("api_clients.gemini_request_dispatcher", limiter):
            self.assertEqual(client._generate_text("prompt"), "fallback result")
        models[GEMINI_TEXT_MODEL].generate_content.assert_called_once()
        models[fallback].generate_content.assert_called_once()
        self.assertEqual(client.last_text_request["effective_model"], fallback)
        self.assertTrue(client.last_text_request["fallback_used"])
        self.assertEqual(len(limiter._request_times), 2)
        self.assertEqual(self.fake_time.sleeps, [])

    def test_daily_quota_cascades_through_chain(self):
        first, second, third, final = GEMINI_TEXT_MODEL_FALLBACK_CHAIN
        client, models = self.make_text_client({
            first: self.quota_error("GenerateRequestsPerDayPerModel-FreeTier"),
            second: self.quota_error("GenerateRequestsPerDayPerModel-FreeTier"),
            third: Mock(text="third result"),
        })
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter()):
            self.assertEqual(client._generate_text("prompt"), "third result")
        for model_name in (first, second, third):
            models[model_name].generate_content.assert_called_once()
        models[final].generate_content.assert_not_called()

    def test_all_daily_quotas_raise_clear_error_after_one_attempt_each(self):
        client, models = self.make_text_client({
            model_name: self.quota_error("GenerateRequestsPerDayPerModel-FreeTier")
            for model_name in GEMINI_TEXT_MODEL_FALLBACK_CHAIN
        })
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter()):
            with self.assertRaisesRegex(RuntimeError, GEMINI_ALL_TEXT_MODELS_QUOTA_EXHAUSTED):
                client._generate_text("prompt")
        for model in models.values():
            model.generate_content.assert_called_once()
        self.assertEqual(self.fake_time.sleeps, [])

    def test_model_minute_quota_retries_same_model_before_success(self):
        error = self.quota_error("GenerateRequestsPerMinutePerModel-FreeTier", message="Please retry in 0.25s")
        client, models = self.make_text_client({GEMINI_TEXT_MODEL: [error, Mock(text="retried result")]})
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter()):
            self.assertEqual(client._generate_text("prompt"), "retried result")
        self.assertEqual(models[GEMINI_TEXT_MODEL].generate_content.call_count, 2)
        models[GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1]].generate_content.assert_not_called()
        self.assertEqual(self.fake_time.sleeps, [0.25])

    def test_project_daily_quota_does_not_fallback(self):
        client, models = self.make_text_client({
            GEMINI_TEXT_MODEL: self.quota_error(
                "GenerateRequestsPerDayPerProject-FreeTier", message="Please retry in 0.5s"
            ),
        })
        limiter = self.use_limiter(max_attempts=2)
        with patch("api_clients.gemini_request_dispatcher", limiter):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_PROJECT_QUOTA_EXHAUSTED"):
                client._generate_text("prompt")
        self.assertEqual(models[GEMINI_TEXT_MODEL].generate_content.call_count, 2)
        models[GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1]].generate_content.assert_not_called()
        self.assertEqual(self.fake_time.sleeps, [0.5])
        self.assertEqual(len(limiter._request_times), 2)

    def test_invalid_api_key_does_not_fallback(self):
        error = google_exceptions.Unauthenticated("invalid API key")
        client, models = self.make_text_client({GEMINI_TEXT_MODEL: error})
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter()):
            with self.assertRaises(google_exceptions.Unauthenticated):
                client._generate_text("prompt")
        models[GEMINI_TEXT_MODEL].generate_content.assert_called_once()
        models[GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1]].generate_content.assert_not_called()

    def test_unclassified_resource_exhausted_uses_retry_without_fallback(self):
        error = google_exceptions.ResourceExhausted("resource exhausted")
        client, models = self.make_text_client({GEMINI_TEXT_MODEL: error})
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter(max_attempts=2)):
            with self.assertRaises(google_exceptions.ResourceExhausted):
                client._generate_text("prompt")
        self.assertEqual(models[GEMINI_TEXT_MODEL].generate_content.call_count, 2)
        models[GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1]].generate_content.assert_not_called()

    def test_quota_classifier_uses_structured_scope_and_cadence(self):
        self.assertEqual(
            classify_gemini_error(self.quota_error("GenerateRequestsPerDayPerModel-FreeTier")),
            MODEL_DAILY_QUOTA,
        )
        self.assertEqual(
            classify_gemini_error(self.quota_error("GenerateRequestsPerMinutePerModel-FreeTier")),
            MODEL_MINUTE_QUOTA,
        )
        self.assertEqual(
            classify_gemini_error(self.quota_error("GenerateRequestsPerDayPerProject-FreeTier")),
            PROJECT_DAILY_QUOTA,
        )
        self.assertEqual(
            classify_gemini_error(self.quota_error("GenerateRequestsPerMinutePerProject-FreeTier")),
            "PROJECT_MINUTE_QUOTA",
        )
        self.assertEqual(
            classify_gemini_error(google_exceptions.ResourceExhausted("unclassified")),
            "OTHER_RESOURCE_EXHAUSTED",
        )
        self.assertEqual(classify_gemini_error(ValueError("bad input")), NON_QUOTA_ERROR)

    def test_operation_allowlist_enables_verified_chain_for_all_text_operations(self):
        self.assertEqual(GEMINI_TEXT_MODEL_OPERATION_ALLOWLIST["text"], GEMINI_TEXT_MODEL_FALLBACK_CHAIN)
        self.assertEqual(GEMINI_TEXT_MODEL_OPERATION_ALLOWLIST["json"], GEMINI_TEXT_MODEL_FALLBACK_CHAIN)
        self.assertEqual(GEMINI_TEXT_MODEL_OPERATION_ALLOWLIST["google_search"], GEMINI_TEXT_MODEL_FALLBACK_CHAIN)

    def test_json_mode_falls_back_to_verified_text_model(self):
        fallback = GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1]
        client, models = self.make_text_client({
            GEMINI_TEXT_MODEL: self.quota_error("GenerateRequestsPerDayPerModel-FreeTier"),
            fallback: Mock(text="json fallback"),
        })
        with patch("api_clients.gemini_request_dispatcher", self.use_limiter()):
            self.assertEqual(client._generate_text("prompt", as_json=True), "json fallback")
        models[GEMINI_TEXT_MODEL].generate_content.assert_called_once()
        models[fallback].generate_content.assert_called_once()

    def test_deep_research_uses_dispatcher(self):
        client = GoogleClient.__new__(GoogleClient)
        client.config = {"TEXT_ENGINE": "Gemini API"}
        client.api_key = "test-key"
        response = Mock()
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "research"}]}}]
        }
        news_client = Mock()
        news_client.get_news.return_value = ""

        with patch("api_clients.requests.post", return_value=response) as post:
            result = client.deep_research("topic", "English", news_client)

        self.assertEqual(result, "research")
        self.assertEqual(self.immediate_dispatcher.calls, 1)
        post.assert_called_once()
        self.assertEqual(client.last_text_request["effective_model"], GEMINI_TEXT_MODEL)

    def test_deep_research_falls_back_through_dispatcher(self):
        client = GoogleClient.__new__(GoogleClient)
        client.config = {"TEXT_ENGINE": "Gemini API"}
        client.api_key = "test-key"
        primary_error_body = {
            "error": {
                "code": 429,
                "message": "model quota exhausted",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "metadata": {
                        "quota_id": "GenerateRequestsPerDayPerModel-FreeTier",
                        "quota_metric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    },
                }],
            }
        }
        failed_response = Mock(status_code=429, headers={}, text=json.dumps(primary_error_body))
        failed_response.json.return_value = primary_error_body
        failed = requests.exceptions.HTTPError(response=failed_response)
        first_response = Mock(status_code=429, headers={}, text=json.dumps(primary_error_body))
        first_response.raise_for_status.side_effect = failed
        fallback_response = Mock()
        fallback_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "fallback research"}]}}]
        }
        fallback_response.raise_for_status.return_value = None
        news_client = Mock()
        news_client.get_news.return_value = ""
        limiter = self.use_limiter()

        with patch("api_clients.gemini_request_dispatcher", limiter), \
                patch("api_clients.requests.post", side_effect=[first_response, fallback_response]) as post:
            self.assertEqual(client.deep_research("topic", "English", news_client), "fallback research")

        self.assertEqual(post.call_count, 2)
        self.assertIn(GEMINI_TEXT_MODEL, post.call_args_list[0].args[0])
        self.assertIn(GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1], post.call_args_list[1].args[0])
        self.assertEqual(len(limiter._request_times), 2)
        self.assertEqual(client.last_text_request["effective_model"], GEMINI_TEXT_MODEL_FALLBACK_CHAIN[1])

    def test_gemini_image_uses_dispatcher(self):
        client = GoogleClient.__new__(GoogleClient)
        client.safety_settings = {}
        image_data = base64.b64encode(b"image-bytes").decode("ascii")
        part = SimpleNamespace(mime_type="image/png", inline_data=SimpleNamespace(data=image_data))
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        model = Mock()
        model.generate_content.return_value = response

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/image.png"
            with patch("api_clients.genai.GenerativeModel", return_value=model) as model_factory:
                client.gemini_nanobanana_image("image prompt", output_path)
            model_factory.assert_called_once_with(NANO_BANANA_IMAGE_MODEL, safety_settings={})
            with open(output_path, "rb") as image_file:
                self.assertEqual(image_file.read(), b"image-bytes")

        self.assertEqual(self.immediate_dispatcher.calls, 1)
        model.generate_content.assert_called_once_with(
            "image prompt", request_options={"retry": None}
        )

    def test_generate_tts_uses_dispatcher(self):
        client = GoogleClient.__new__(GoogleClient)
        client.api_key = "test-key"
        pcm_data = base64.b64encode(b"\x00\x00\x00\x00").decode("ascii")
        response = Mock()
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": pcm_data}}]}}]
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/audio.wav"
            with patch("api_clients.requests.post", return_value=response) as post:
                client.generate_tts("hello", output_path, {"SPEAKER1": "Kore"})

        self.assertEqual(self.immediate_dispatcher.calls, 1)
        post.assert_called_once()
        self.assertIn(GEMINI_TTS_MODEL, post.call_args.args[0])

    def test_clipper_analysis_makes_one_text_call(self):
        import pipeline_shorts

        client = Mock()
        client._generate_text.return_value = json.dumps([
            {"start": 1, "end": 31, "title": "Clip", "seo_title": "SEO"}
        ])
        callbacks = []
        words = [{"start": 0.0, "end": 0.5, "word": "hello"}]

        with patch("pipeline_shorts.load_config", return_value={}), \
                patch("pipeline_shorts.GoogleClient", return_value=client), \
                patch("pipeline_shorts.generate_captions", return_value=words), \
                patch("pipeline_shorts._persist_word_timestamps") as persist, \
                patch("pipeline_shorts.os.makedirs"):
            pipeline_shorts.analyze_video(
                "local-video.mp4", True, "test-job",
                lambda *args, **kwargs: callbacks.append((args, kwargs)), 1,
            )

        client._generate_text.assert_called_once()
        persist.assert_called_once_with(os.path.join("workspace", "clipper_test-job"), words)
        self.assertTrue(any(kwargs.get("result") for _, kwargs in callbacks))

    def test_clipper_rendering_makes_zero_text_calls(self):
        import pipeline_shorts

        client = Mock()
        callbacks = []
        selected_clips = [{"start": 0, "end": 5, "title": "Clip", "broll": []}]

        with patch("pipeline_shorts.load_config", return_value={"PIXABAY_API_KEY": ""}), \
                patch("pipeline_shorts.GoogleClient", return_value=client), \
                patch("pipeline_shorts._load_word_timestamps", return_value=[
                    {"word": "hello", "start": 0.0, "end": 0.5}
                ]), \
                patch("pipeline_shorts.os.makedirs"), \
                patch("pipeline_shorts.os.path.exists", return_value=False), \
                patch("pipeline_shorts.render_scene_aware_clip"), \
                patch("pipeline_shorts.generate_captions_from_timestamps"), \
                patch("pipeline_shorts.subprocess.run"):
            pipeline_shorts.render_youtube_clips(
                "test-job", "video.mp4", selected_clips,
                lambda *args, **kwargs: callbacks.append((args, kwargs)),
            )

        client._generate_text.assert_not_called()
        self.assertTrue(any(kwargs.get("result") for _, kwargs in callbacks))


if __name__ == "__main__":
    unittest.main()