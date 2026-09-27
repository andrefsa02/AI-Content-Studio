import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pysubs2

import pipeline
import pipeline_shorts
from pipeline import generate_captions_from_timestamps
from pipeline_shorts import (
    _clip_local_word_timestamps,
    _clipper_caption_style,
    _load_word_timestamps,
    _persist_word_timestamps,
)


def timestamp(word, start, end, **extra):
    return {"word": word, "start": start, "end": end, **extra}


class ClipperTimestampReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def test_analysis_persists_complete_timestamp_objects_after_one_transcription(self):
        job_id = "analysis-persist"
        words = [
            timestamp("Hello", 0.125, 0.5, probability=0.91),
            timestamp("world.", 0.625, 1.0, probability=0.88),
        ]
        client = Mock()
        client._generate_text.return_value = json.dumps([{"start": 0.1, "end": 0.9, "title": "Clip"}])

        with patch("pipeline_shorts.load_config", return_value={}), \
                patch("pipeline_shorts.GoogleClient", return_value=client), \
                patch("pipeline_shorts.generate_captions", return_value=words) as transcribe, \
                patch("pipeline_shorts.logging.info") as log_info:
            pipeline_shorts.analyze_video(
                "source.mp4", True, job_id,
                Mock(),
                num_clips=1,
            )

        transcribe.assert_called_once_with("source.mp4", None, "English")
        timestamp_path = Path("workspace") / f"clipper_{job_id}" / "word_timestamps.json"
        with timestamp_path.open("r", encoding="utf-8") as handle:
            persisted = json.load(handle)
        self.assertEqual(persisted, words)
        self.assertEqual(len(persisted), len(words))
        self.assertEqual([(word["start"], word["end"]) for word in persisted], [(0.125, 0.5), (0.625, 1.0)])
        log_info.assert_any_call(
            "Saved %d Whisper word timestamps to %s",
            2,
            os.path.join("workspace", f"clipper_{job_id}", "word_timestamps.json"),
        )

    def test_persistence_helper_writes_utf8_json_and_preserves_values(self):
        words = [timestamp("MONEY 💰", 10.125, 10.75, confidence=0.95)]
        output_dir = Path("workspace/test_job")
        output_dir.mkdir(parents=True)
        timestamp_path = _persist_word_timestamps(str(output_dir), words)
        self.assertTrue(os.path.isfile(timestamp_path))
        with open(timestamp_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), words)
        self.assertEqual(_load_word_timestamps(str(output_dir)), words)

    def test_render_loads_saved_timestamps_and_never_runs_whisper_for_three_clips(self):
        job_id = "three-clip-render"
        output_dir = Path("workspace") / f"clipper_{job_id}"
        output_dir.mkdir(parents=True)
        words = [
            timestamp("first", 0.0, 0.3), timestamp("clip", 0.35, 0.7),
            timestamp("second", 1.0, 1.3), timestamp("clip", 1.35, 1.7),
            timestamp("third", 2.0, 2.3), timestamp("clip", 2.35, 2.7),
        ]
        with (output_dir / "word_timestamps.json").open("w", encoding="utf-8") as handle:
            json.dump(words, handle, ensure_ascii=False)
        selected_clips = [
            {"start": 0.0, "end": 0.8, "title": "First"},
            {"start": 1.0, "end": 1.8, "title": "Second"},
            {"start": 2.0, "end": 2.8, "title": "Third"},
        ]
        callback_updates = []

        with patch("pipeline_shorts.load_config", return_value={"PIXABAY_API_KEY": ""}), \
                patch("pipeline_shorts.GoogleClient"), \
                patch("pipeline_shorts.render_scene_aware_clip") as crop, \
                patch("pipeline_shorts.generate_captions_from_timestamps") as write_ass, \
                patch("pipeline_shorts.subprocess.run"), \
                patch("pipeline.whisper.load_model", side_effect=AssertionError("Whisper must not run during render")) as load_model:
            pipeline_shorts.render_youtube_clips(
                job_id,
                "source.mp4",
                selected_clips,
                lambda *args, **kwargs: callback_updates.append((args, kwargs)),
            )

        self.assertEqual(crop.call_count, 3)
        self.assertEqual(write_ass.call_count, 3)
        self.assertEqual(load_model.call_count, 0)
        first_words = write_ass.call_args_list[0].args[0]
        self.assertEqual([word["word"] for word in first_words], ["first", "clip"])
        self.assertTrue(any(kwargs.get("result") for _, kwargs in callback_updates))

    def test_render_fails_clearly_before_crop_if_timestamp_file_missing(self):
        job_id = "missing-timestamps"
        errors = []
        with patch("pipeline_shorts.load_config", return_value={}), \
                patch("pipeline_shorts.GoogleClient"), \
                patch("pipeline_shorts.render_scene_aware_clip") as crop:
            pipeline_shorts.render_youtube_clips(
                job_id, "source.mp4", [{"start": 0, "end": 1}],
                lambda *args, **kwargs: errors.append(kwargs.get("error")),
            )
        crop.assert_not_called()
        self.assertTrue(any("word_timestamps.json" in str(error) for error in errors))

    def test_render_fails_clearly_before_crop_if_timestamp_file_is_invalid(self):
        job_id = "invalid-timestamps"
        output_dir = Path("workspace") / f"clipper_{job_id}"
        output_dir.mkdir(parents=True)
        (output_dir / "word_timestamps.json").write_text("{not valid json", encoding="utf-8")
        errors = []

        with patch("pipeline_shorts.load_config", return_value={}), \
                patch("pipeline_shorts.GoogleClient"), \
                patch("pipeline_shorts.render_scene_aware_clip") as crop:
            pipeline_shorts.render_youtube_clips(
                job_id, "source.mp4", [{"start": 0, "end": 1}],
                lambda *args, **kwargs: errors.append(kwargs.get("error")),
            )
        crop.assert_not_called()
        self.assertTrue(any("timestamps are unreadable" in str(error) for error in errors))

    def test_clip_filter_uses_overlap_and_converts_to_local_timestamps(self):
        words = [
            timestamp("A", 1.0, 1.5),
            timestamp("B", 4.0, 4.5),
            timestamp("C", 10.0, 10.5),
            timestamp("overlap", 2.8, 3.4),
        ]
        original = [dict(word) for word in words]

        selected = _clip_local_word_timestamps(words, 3.0, 8.0)

        self.assertEqual([word["word"] for word in selected], ["B", "overlap"])
        self.assertEqual((selected[0]["start"], selected[0]["end"]), (1.0, 1.5))
        self.assertAlmostEqual(selected[1]["start"], 0.0)
        self.assertAlmostEqual(selected[1]["end"], 0.4)
        self.assertEqual(words, original)

    def test_clip_filter_converts_full_video_timestamp_to_local_timestamp(self):
        words = [timestamp("word", 103.25, 104.0)]
        local = _clip_local_word_timestamps(words, 100.0, 105.0)
        self.assertAlmostEqual(local[0]["start"], 3.25)
        self.assertAlmostEqual(local[0]["end"], 4.0)
        self.assertEqual(words[0]["start"], 103.25)

    def test_timestamp_ass_helper_writes_style_without_loading_whisper(self):
        words = [timestamp("MONEY", 0.25, 0.5), timestamp("VIRAL", 0.5, 0.8)]
        ass_path = Path(self.temp_dir.name) / "timestamp-helper.ass"
        with patch("pipeline.whisper.load_model", side_effect=AssertionError("ASS helper must not load Whisper")) as load_model:
            generate_captions_from_timestamps(words, str(ass_path), _clipper_caption_style({}))

        subtitles = pysubs2.load(str(ass_path), format_="ass")
        self.assertTrue(ass_path.is_file())
        self.assertEqual(subtitles.info["PlayResX"], "720")
        self.assertEqual(subtitles.info["PlayResY"], "1280")
        self.assertEqual(subtitles.info["WrapStyle"], "2")
        style = subtitles.styles["Default"]
        self.assertEqual(style.fontsize, 20)
        self.assertTrue(style.bold)
        self.assertEqual(style.marginv, 110)
        self.assertEqual(style.borderstyle, 1)
        self.assertIn(r"\c&HFFFFFF&", subtitles.events[0].text)
        self.assertIn("💰", subtitles.events[0].text)
        self.assertIn("🔥", subtitles.events[1].text)
        self.assertEqual(load_model.call_count, 0)

    def test_timestamp_helper_source_offset_is_non_mutating(self):
        words = [timestamp("offset", 103.25, 104.0)]
        adjusted = generate_captions_from_timestamps(words, None, source_offset=100.0)
        self.assertAlmostEqual(adjusted[0]["start"], 3.25)
        self.assertAlmostEqual(adjusted[0]["end"], 4.0)
        self.assertEqual(words[0]["start"], 103.25)


if __name__ == "__main__":
    unittest.main()