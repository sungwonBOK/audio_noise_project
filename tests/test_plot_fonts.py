"""Korean font lookup for figures works on any OS and never crashes when no font exists."""
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

from plot_fonts import FALLBACK, korean_font_family


class KoreanFontTests(unittest.TestCase):
    def test_registers_font_file_and_returns_its_family(self):
        dejavu = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        self.assertEqual(korean_font_family(files=["/no/such/font.ttf", str(dejavu)], names=()), "DejaVu Sans")

    def test_no_font_falls_back_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            family = korean_font_family(files=["/no/such/font.ttf"], names=("No Such Korean Font",))
        self.assertEqual(family, FALLBACK)
        self.assertTrue(any("fonts-nanum" in str(w.message) for w in caught))

    def test_windows_uses_malgun_gothic(self):
        if not Path(r"C:\Windows\Fonts\malgun.ttf").is_file():
            self.skipTest("Malgun Gothic is not installed here")
        self.assertEqual(korean_font_family(), "Malgun Gothic")


if __name__ == "__main__":
    unittest.main()
