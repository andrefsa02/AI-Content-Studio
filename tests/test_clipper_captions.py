import os
import tempfile
import unittest
from unittest.mock import patch

import pysubs2

from pipeline import _estimate_caption_width, _group_caption_words, generate_captions
from pipeline_shorts import _clipper_caption_style, _resolve_clipper_caption_font


def make_words(texts, starts=None, duration=0.25, gap=0.05):
    starts = starts or []
    words = []
    cursor = 0.0
    for index, text in enumerate(texts):
        start = starts[index] if index < len(starts) else cursor
        words.append({"word": text, "start": start, "end": start + duration})
        cursor = start + duration + gap
    return words


class FakeWhisperModel:
    def __init__(self, words):
        self.words = words

    def transcribe(self, audio_file, word_timestamps, language):
        return {"segments": [{"words": self.words}]}


class ClipperCaptionTests(unittest.TestCase):
    def group_text(self, texts, **kwargs):
        words = make_words(texts)
        groups = _group_caption_words(words, **kwargs)
        return [[word["word"] for word in line] for group in groups for line in group], groups

    def test_normal_short_phrase_stays_together(self):
        lines, groups = self.group_text(["We", "make", "short", "captions"])
        self.assertEqual(lines, [["We", "make", "short", "captions"]])
        self.assertEqual(len(groups), 1)

    def test_exactly_four_words_stay_together(self):
        _, groups = self.group_text(["One", "two", "three", "four"])
        self.assertEqual([len(line) for line in groups[0]], [4])

    def test_seven_words_fit_one_line(self):
        _, groups = self.group_text(["one", "two", "three", "four", "five", "six", "seven"])
        self.assertEqual([len(line) for line in groups[0]], [7])

    def test_more_than_seven_words_use_multiple_lines_and_groups(self):
        texts = [f"word{index}" for index in range(1, 22)]
        flattened, groups = self.group_text(texts)
        self.assertEqual([word for line in flattened for word in line], texts)
        self.assertGreater(len(groups), 1)
        self.assertTrue(all(len(group) <= 2 for group in groups))
        self.assertTrue(all(len(line) <= 7 for group in groups for line in group))
        for group in groups:
            for line in group:
                text = " ".join(word["word"].upper() for word in line)
                if len(line) > 1:
                    self.assertLessEqual(len(text), 34)
                    self.assertLessEqual(_estimate_caption_width(text), 30.0)

    def test_long_word_isolated_by_width_guard(self):
        long_word = "supercalifragilisticexpialidocious"
        _, groups = self.group_text(["A", long_word, "follows", "this"])
        long_word_lines = [
            line for group in groups for line in group
            if any(word["word"] == long_word for word in line)
        ]
        self.assertEqual(len(long_word_lines), 1)
        self.assertEqual([word["word"] for word in long_word_lines[0]], [long_word])

    def test_punctuation_ends_a_phrase(self):
        _, groups = self.group_text(["We", "can", "stop,", "then", "continue."])
        self.assertEqual(
            [[word["word"] for line in group for word in line] for group in groups],
            [["We", "can", "stop,"], ["then", "continue."]],
        )

    def test_large_inter_word_pause_ends_a_phrase(self):
        words = make_words(["first", "phrase", "second", "phrase"], starts=[0.0, 0.3, 2.0, 2.3])
        groups = _group_caption_words(words)
        self.assertEqual(
            [[word["word"] for line in group for word in line] for group in groups],
            [["first", "phrase"], ["second", "phrase"]],
        )

    def test_empty_and_invalid_entries_are_skipped(self):
        words = [
            None,
            {"word": "  ", "start": 0, "end": 0.2},
            {"word": "missing end", "start": 0.3},
            {"word": "bad time", "start": "bad", "end": 0.8},
            {"word": "zero duration", "start": 1, "end": 1},
            {"word": "works", "start": 1.2, "end": 1.5},
        ]
        groups = _group_caption_words(words)
        self.assertEqual([[word["word"] for line in group for word in line] for group in groups], [["works"]])

    def test_many_words_never_exceed_two_explicit_lines(self):
        groups = _group_caption_words(make_words([f"w{i}" for i in range(45)]))
        self.assertTrue(all(len(group) <= 2 for group in groups))
        self.assertTrue(all(len(line) <= 7 for group in groups for line in group))

    def test_word_timestamps_are_preserved_in_ass_events(self):
        words = make_words(["this", "phrase", "keeps", "all", "word", "times", "exact"])
        with tempfile.TemporaryDirectory() as directory:
            ass_path = os.path.join(directory, "caption.ass")
            with patch("pipeline.whisper.load_model", return_value=FakeWhisperModel(words)):
                returned = generate_captions(
                    "unused-local-audio.mp4",
                    ass_path,
                    "English",
                    style_opts=_clipper_caption_style({}),
                )
            subtitles = pysubs2.load(ass_path, format_="ass")

        self.assertEqual(returned, words)
        self.assertEqual(len(subtitles.events), len(words))
        for event, word in zip(subtitles.events, words):
            self.assertEqual(event.start, pysubs2.make_time(ms=int(word["start"] * 1000)))
            self.assertEqual(event.end, pysubs2.make_time(ms=int(word["end"] * 1000)))
        self.assertIn(r"\N", subtitles.events[0].text)

    def test_default_ass_fill_outline_and_shadow_are_opaque_as_intended(self):
        style = _clipper_caption_style({})
        self.assertEqual((style["primarycolor"].r, style["primarycolor"].g, style["primarycolor"].b), (255, 255, 255))
        self.assertEqual(style["primarycolor"].a, 0)
        self.assertEqual((style["outlinecolor"].r, style["outlinecolor"].g, style["outlinecolor"].b), (0, 0, 0))
        self.assertEqual(style["outlinecolor"].a, 0)
        self.assertEqual((style["backcolor"].r, style["backcolor"].g, style["backcolor"].b), (0, 0, 0))
        self.assertEqual(style["backcolor"].a, 150)
        self.assertEqual(style["outline"], 3.0)
        self.assertEqual(style["shadow"], 1)

    def test_clipper_ass_sets_playres_and_disables_automatic_wrapping(self):
        words = make_words(["This", "is", "a", "test"])
        with tempfile.TemporaryDirectory() as directory:
            ass_path = os.path.join(directory, "caption.ass")
            with patch("pipeline.whisper.load_model", return_value=FakeWhisperModel(words)):
                generate_captions(
                    "unused-local-audio.mp4",
                    ass_path,
                    "English",
                    style_opts=_clipper_caption_style({}),
                )
            subtitles = pysubs2.load(ass_path, format_="ass")

        self.assertEqual(subtitles.info["PlayResX"], "720")
        self.assertEqual(subtitles.info["PlayResY"], "1280")
        self.assertEqual(subtitles.info["WrapStyle"], "2")
        self.assertEqual(subtitles.styles["Default"].borderstyle, 1)

    def test_existing_themes_serialize_with_opaque_primary_fills(self):
        expected = {
            "default": ((255, 255, 255), (0, 0, 0)),
            "viral_yellow": ((255, 255, 0), (0, 0, 0)),
            "neon_cyber": ((0, 255, 255), (255, 0, 255)),
            "black_white": ((0, 0, 0), (255, 255, 255)),
        }
        word = make_words(["caption"])
        with tempfile.TemporaryDirectory() as directory:
            for theme, (primary_rgb, back_rgb) in expected.items():
                ass_path = os.path.join(directory, f"{theme}.ass")
                with patch("pipeline.whisper.load_model", return_value=FakeWhisperModel(word)):
                    generate_captions(
                        "unused-local-audio.mp4",
                        ass_path,
                        "English",
                        style_opts=_clipper_caption_style({"CAPTION_THEME": theme}),
                    )
                subtitles = pysubs2.load(ass_path, format_="ass")
                primary = subtitles.styles["Default"].primarycolor
                back = subtitles.styles["Default"].backcolor
                outline = subtitles.styles["Default"].outlinecolor
                self.assertEqual((primary.r, primary.g, primary.b), primary_rgb)
                self.assertEqual(primary.a, 0)
                self.assertEqual((back.r, back.g, back.b), back_rgb)
                self.assertEqual((outline.r, outline.g, outline.b, outline.a), (0, 0, 0, 0))
                self.assertEqual(subtitles.styles["Default"].borderstyle, 1)

    def test_configured_font_uses_arial_fallback_when_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            open(os.path.join(directory, "arial.ttf"), "wb").close()
            open(os.path.join(directory, "impact.ttf"), "wb").close()
            self.assertEqual(_resolve_clipper_caption_font("Impact", directory), "Impact")
            self.assertEqual(_resolve_clipper_caption_font("Roboto", directory), "Arial")

    def test_clipper_font_size_and_position_are_clipper_specific(self):
        style = _clipper_caption_style({"CAPTION_FONT": "Arial", "CAPTION_FONT_SIZE": 22})
        self.assertEqual(style["fontname"], "Arial")
        self.assertEqual(style["fontsize"], 20)
        self.assertTrue(style["bold"])
        self.assertEqual(style["alignment"], 2)
        self.assertEqual(style["marginv"], 110)

    def test_non_clipper_caption_call_keeps_legacy_layout(self):
        words = make_words(["one", "two", "three", "four", "five"])
        with tempfile.TemporaryDirectory() as directory:
            ass_path = os.path.join(directory, "legacy.ass")
            with patch("pipeline.whisper.load_model", return_value=FakeWhisperModel(words)):
                generate_captions("unused-local-audio.mp4", ass_path, "English", style_opts={})
            subtitles = pysubs2.load(ass_path, format_="ass")

        self.assertNotIn("PlayResX", subtitles.info)
        self.assertNotIn("PlayResY", subtitles.info)
        self.assertEqual(len(subtitles.events), len(words))
        self.assertNotIn(r"\N", subtitles.events[0].text)


if __name__ == "__main__":
    unittest.main()