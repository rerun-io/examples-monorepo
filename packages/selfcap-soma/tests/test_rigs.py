from selfcap_soma.catalog import rig_paths_from_kinds


def test_three_phone_layout_does_not_assume_four_phone_rig_numbers():
    layout = rig_paths_from_kinds(
        {"/world/rig_00": "exo", "/world/rig_01": "exo", "/world/rig_02": "exo", "/world/rig_03": "ego", "/world/rig_04": "quest"}
    )
    assert layout.oak == "/world/rig_03"
    assert layout.quest == "/world/rig_04"
    assert len(layout.exo) == 3
