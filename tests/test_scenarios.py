from scenarios import ScenarioManifest, build_manifest


def test_manifest_is_deterministic_unique_and_round_trips(tmp_path):
    first = build_manifest("validation", seed=123)
    second = build_manifest("validation", seed=123)
    assert first.digest() == second.digest()
    assert len({item.scenario_id for item in first.scenarios}) == len(first.scenarios)
    target = tmp_path / "manifest.json"
    first.save(target)
    loaded = ScenarioManifest.load(target)
    assert loaded.digest() == first.digest()


def test_manifest_balances_disturbance_conditions():
    manifest = build_manifest("validation")
    counts = {}
    for scenario in manifest.scenarios:
        key = (scenario.evaluation_mode, scenario.condition)
        counts[key] = counts.get(key, 0) + 1
    assert len(set(counts.values())) == 1
    assert set(counts) == {
        (mode, condition)
        for mode in ("stationary", "transient")
        for condition in ("nominal", "delay", "noise", "combined")
    }
