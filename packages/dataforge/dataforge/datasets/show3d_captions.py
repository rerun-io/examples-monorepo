"""Independent SHOW3D captions layer."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import rerun as rr
from serde import serde

from dataforge import schema, writing
from dataforge.datasets.show3d_source import CAPTIONS_VERSION
from dataforge.identity import SequenceIdentity


@serde
@dataclass(frozen=True, slots=True)
class Caption:
    """All ten released caption fields."""

    object_alias: str
    """Object name used by the annotator."""
    action_hint: str
    """Short action label."""
    hand: str
    """Hand or hands used."""
    interaction_description: str
    """Detailed interaction."""
    start_state: str
    """Initial state."""
    end_state: str
    """Final state."""
    intent: str
    """Annotated intent."""
    scene_description: str
    """Capture setting."""
    additional_observations: str
    """Other observations, possibly empty."""
    overall_caption: str
    """Summary shown first in the instruction pane."""

    def markdown(self) -> str:
        """Summary followed by a Markdown definition list of structured fields."""
        pairs: tuple[tuple[str, str], ...] = (
            ("Object alias", self.object_alias),
            ("Action hint", self.action_hint),
            ("Hand", self.hand),
            ("Interaction description", self.interaction_description),
            ("Start state", self.start_state),
            ("End state", self.end_state),
            ("Intent", self.intent),
            ("Scene description", self.scene_description),
            ("Additional observations", self.additional_observations),
        )
        definitions: list[str] = [f"{label}\n:   " + value.replace("\n", "\n    ") for label, value in pairs]
        return self.overall_caption + "\n\n" + "\n\n".join(definitions)


def write_captions_layer(identity: SequenceIdentity, caption: Caption, target: Path) -> None:
    """Publish the static instruction and the caption fields a catalog user searches on."""
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        rr.log(schema.instruction_path(), rr.TextDocument(caption.markdown(), media_type="text/markdown"), static=True, recording=recording)
        recording.send_property(
            "captions",
            rr.AnyValues(
                version=pa.array([CAPTIONS_VERSION], type=pa.string()),
                hand=pa.array([caption.hand], type=pa.string()),
                overall_caption=pa.array([caption.overall_caption], type=pa.string()),
            ),
        )
