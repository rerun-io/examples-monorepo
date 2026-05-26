from serde.json import from_json, to_json

from wilor_nano.gradio_ui.simple_keypoint_ui import SimpleKeypointJson


def test_simple_keypoint_json_round_trips_xy() -> None:
    keypoint: SimpleKeypointJson = SimpleKeypointJson(
        application_id="wilor-test",
        recording_id="recording-1",
        xy=[12.5, 42.25],
    )

    payload: str = to_json(keypoint)
    decoded: SimpleKeypointJson = from_json(SimpleKeypointJson, payload)

    assert decoded.xy == [12.5, 42.25]
