"""Korean font for matplotlib figures on Windows, macOS and Linux (Google Colab).

Colab/Linux has no Korean font by default; install one once per session:
    !apt-get -qq install -y fonts-nanum
Without it figures are still saved, but Korean text shows as boxes.
"""
import warnings
from pathlib import Path

from matplotlib import font_manager

FONT_FILES = (
    r"C:\Windows\Fonts\malgun.ttf",                            # Windows: Malgun Gothic
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",          # Linux/Colab: fonts-nanum
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",       # macOS
    "/Library/Fonts/AppleGothic.ttf",
)
FONT_NAMES = ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR")
FALLBACK = "DejaVu Sans"


def korean_font_family(files=FONT_FILES, names=FONT_NAMES):
    """Register the first Korean font found and return its family name (FALLBACK if none)."""
    for path in files:
        if Path(path).is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in names:
        if name in installed:
            return name
    warnings.warn("No Korean font found; Korean text in figures will show as boxes. "
                  "On Colab/Linux run: apt-get install -y fonts-nanum", stacklevel=2)
    return FALLBACK
