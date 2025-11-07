import pytest
from pathlib import Path
from slider import Beatmap

data_dir = Path(__file__).resolve().parent.parent / "example_data" / "beatmaps"


@pytest.mark.parametrize("beatmap_path", sorted(data_dir.glob("*.osu")))
def test_load_example_beatmap(beatmap_path: "str | Path") -> None:
    Beatmap.from_path(beatmap_path)
