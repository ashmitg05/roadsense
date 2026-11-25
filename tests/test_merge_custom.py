from pathlib import Path

import pytest

from src.data.merge_custom import merge


def test_merge_remaps_classes_and_keeps_videos_whole(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "export"
    (exp / "train/images").mkdir(parents=True)
    (exp / "train/labels").mkdir(parents=True)
    # Roboflow sorted classes alphabetically: pothole is id 2 there, 3 here
    (exp / "data.yaml").write_text(
        "names: [alligator_crack, longitudinal_crack, pothole, transverse_crack]\n"
    )
    for v in range(10):
        for f in range(3):
            stem = f"vid{v}__f{f}_jpg.rf.abc"
            (exp / "train/images" / f"{stem}.jpg").write_bytes(b"x")
            (exp / "train/labels" / f"{stem}.txt").write_text("2 0.5 0.5 0.1 0.1")
    splits = tmp_path / "splits"
    splits.mkdir()
    r = merge(exp, tmp_path / "out", splits, train_frac=0.3, seed=0)

    assert r["train_images"] == 9 and r["wild_test_images"] == 21
    lbl = (tmp_path / "out/labels/custom_vid0__f0_jpg.rf.abc.txt").read_text()
    assert lbl.startswith("3 ")
    train = (splits / "custom_train.txt").read_text()
    wild = (splits / "wild_test.txt").read_text()
    for v in r["train_videos"]:
        assert f"{v}__" not in wild and train.count(f"{v}__") == 3


def test_merge_rejects_unknown_class(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("names: [pothole, manhole]\n")
    with pytest.raises(ValueError, match="manhole"):
        merge(tmp_path, tmp_path / "o", tmp_path, 0.3, 0)
