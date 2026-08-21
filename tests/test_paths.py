import numpy as np

from config import PATH_SPLITS
from core import paths


def test_all_declared_paths_obey_spawn_contract():
    for specs in PATH_SPLITS.values():
        for spec in specs:
            path = paths.build(spec)
            np.testing.assert_allclose(path["pts"][0], (0.0, 0.0), atol=1e-9)
            np.testing.assert_allclose(path["tangent"][0], (1.0, 0.0), atol=2e-3)
            assert np.all(np.diff(path["s"]) > 0)
            assert path["length"] > 0


def test_generated_paths_are_seeded_and_distinct():
    first = paths.build({"kind": "random_curvature", "params": {"seed": 10000}})
    replay = paths.build({"kind": "random_curvature", "params": {"seed": 10000}})
    other = paths.build({"kind": "random_curvature", "params": {"seed": 10001}})
    np.testing.assert_array_equal(first["pts"], replay["pts"])
    assert not np.array_equal(first["pts"], other["pts"])


def test_generated_seeds_do_not_cross_splits():
    seeds = {}
    for split, specs in PATH_SPLITS.items():
        current = {
            spec.get("params", {}).get("seed")
            for spec in specs
            if spec["kind"] == "random_curvature"
        }
        assert all(current.isdisjoint(previous) for previous in seeds.values())
        seeds[split] = current


def test_generated_validation_and_test_paths_are_projection_unambiguous():
    for split in ("validation", "held_out"):
        for spec in PATH_SPLITS[split]:
            if spec["kind"] == "random_curvature":
                paths.validate_geometry(paths.build(spec))


def test_unknown_path_fails_closed():
    try:
        paths.build({"kind": "does_not_exist"})
    except KeyError:
        return
    raise AssertionError("unknown path was accepted")
