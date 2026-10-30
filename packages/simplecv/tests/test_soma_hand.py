"""The shared correspondence preserves the anatomical COCO hand order."""

from simplecv.data.skeleton.soma_hand import SOMA_HAND25_TO_COCO21


def test_soma_hand_mapping_omits_only_non_thumb_metacarpals() -> None:
    soma = ["wrist"]
    for finger in ("thumb", "index", "middle", "ring", "little"):
        soma.extend(f"{finger}/{joint}" for joint in ("base", "proximal", "distal", "tip") if finger == "thumb")
        if finger != "thumb":
            soma.extend(f"{finger}/{joint}" for joint in ("metacarpal", "base", "proximal", "distal", "tip"))
    expected = ["wrist", *(f"{finger}/{joint}" for finger in ("thumb", "index", "middle", "ring", "little") for joint in ("base", "proximal", "distal", "tip"))]
    assert [soma[index] for index in SOMA_HAND25_TO_COCO21] == expected
