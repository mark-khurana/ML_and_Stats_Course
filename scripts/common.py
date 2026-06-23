"""
setup.py - Common functions and configuration for the course.
Import this module at the start of each chapter's Python code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Colour-blind friendly palette (matching R setup)
COURSE_COLOURS = [
    "#2c6fbb",  # blue
    "#e69f00",  # orange
    "#009e73",  # green
    "#cc79a7",  # pink
    "#56b4e9",  # light blue
    "#d55e00",  # red-orange
    "#0072b2",  # dark blue
    "#f0e442",  # yellow
]

# Set default plot style
sns.set_theme(style="whitegrid", palette=COURSE_COLOURS, font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def load_course_data(name: str) -> pd.DataFrame:
    """Load a course dataset by name (without extension)."""
    data_dir = Path(__file__).parent.parent / "data"
    path = data_dir / f"{name}.csv"
    if not path.exists():
        # Try relative path
        path = Path("data") / f"{name}.csv"
    return pd.read_csv(path)
