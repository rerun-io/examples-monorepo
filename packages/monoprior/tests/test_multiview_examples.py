from pathlib import Path

from monopriors.gradio_ui._multiview_common import discover_multiview_examples


def test_discover_multiview_examples_returns_named_scenes_in_ui_order(tmp_path: Path) -> None:
    for scene_name in ("tree", "bench", "stairs", "car_landscape_12", "rp_capture_6"):
        (tmp_path / scene_name).mkdir()

    (tmp_path / "car_landscape_12" / "02.jpg").touch()
    (tmp_path / "car_landscape_12" / "01.JPG").touch()
    (tmp_path / "bench" / "002.png").touch()
    (tmp_path / "bench" / "001.png").touch()
    (tmp_path / "stairs" / "001.jpeg").touch()
    (tmp_path / "tree" / "001.png").touch()
    (tmp_path / "tree" / "notes.txt").touch()
    (tmp_path / "rp_capture_6" / "001.jpg").touch()

    examples = discover_multiview_examples(tmp_path)

    assert [label for label, _ in examples] == ["Car landscape · 2 views", "Bench · 2 views", "Stairs · 1 view", "Tree · 1 view"]
    assert examples[0][1] == [
        str(tmp_path / "car_landscape_12" / "01.JPG"),
        str(tmp_path / "car_landscape_12" / "02.jpg"),
    ]
    assert examples[1][1] == [
        str(tmp_path / "bench" / "001.png"),
        str(tmp_path / "bench" / "002.png"),
    ]
