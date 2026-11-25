from src.data.split import assign, group_frames


def test_group_frames_links_near_duplicates_only() -> None:
    a, b = 0, (1 << 64) - 1
    roots = group_frames([a, b, a ^ 1, b ^ 3, a ^ 0xFF], thresh=4)
    assert roots[0] == roots[2]  # non-adjacent duplicates still grouped
    assert roots[1] == roots[3]
    assert roots[0] != roots[1]
    assert roots[4] != roots[0]  # distance 8 > thresh


def test_assign_ratio_and_deterministic() -> None:
    groups = {g: [g] for g in range(100)}
    strata = dict.fromkeys(groups, 0)
    where = assign(groups, strata, seed=0)
    counts = {s: list(where.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 15, "test": 15}
    assert assign(groups, strata, seed=0) == where
