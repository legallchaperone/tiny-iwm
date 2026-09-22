from scripts.stage_a_data import select_scene_disjoint_samples


def test_stage_a_selection_is_deterministic_and_scene_disjoint() -> None:
    names = [
        "scene-c_0002.npz",
        "scene-a_0002.npz",
        "scene-b_0001.npz",
        "scene-a_0001.npz",
        "scene-d_0001.npz",
        "scene-c_0001.npz",
    ]

    first = select_scene_disjoint_samples(names, train_scenes=2, validation_scenes=1)
    second = select_scene_disjoint_samples(reversed(names), train_scenes=2, validation_scenes=1)

    assert first == second
    assert first == {
        "train": ["scene-a_0001.npz", "scene-b_0001.npz"],
        "validation": ["scene-d_0001.npz"],
    }
    train_scenes = {name.split("_", 1)[0] for name in first["train"]}
    validation_scenes = {name.split("_", 1)[0] for name in first["validation"]}
    assert train_scenes.isdisjoint(validation_scenes)
