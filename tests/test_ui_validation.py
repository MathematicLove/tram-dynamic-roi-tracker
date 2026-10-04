import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))

from validation import check_input_file, parse_camera_index, parse_speed  # noqa: E402


class TestParseSpeed:
    @pytest.mark.parametrize("raw,expected", [("40", 40), (" 25 ", 25), ("1", 1), ("200", 200)])
    def test_valid(self, raw, expected):
        assert parse_speed(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "abc", "40.5", "4 0", "-"])
    def test_not_a_whole_number(self, raw):
        with pytest.raises(ValueError):
            parse_speed(raw)

    @pytest.mark.parametrize("raw", ["0", "-5", "201", "9999"])
    def test_out_of_range(self, raw):
        with pytest.raises(ValueError):
            parse_speed(raw)


class TestParseCameraIndex:
    @pytest.mark.parametrize("raw,expected", [("", "0"), ("  ", "0"), ("0", "0"), (" 2 ", "2")])
    def test_valid(self, raw, expected):
        assert parse_camera_index(raw) == expected

    @pytest.mark.parametrize("raw", ["abc", "-1", "1.5"])
    def test_invalid(self, raw):
        with pytest.raises(ValueError):
            parse_camera_index(raw)


class TestCheckInputFile:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        assert check_input_file(str(f), "a video file") == str(f)

    @pytest.mark.parametrize("path", [None, ""])
    def test_nothing_selected(self, path):
        with pytest.raises(ValueError, match="Select a video file"):
            check_input_file(path, "a video file")

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="File not found"):
            check_input_file(str(tmp_path / "gone.mp4"), "a video file")

    def test_directory_is_not_a_file(self, tmp_path):
        with pytest.raises(ValueError):
            check_input_file(str(tmp_path), "a video file")
