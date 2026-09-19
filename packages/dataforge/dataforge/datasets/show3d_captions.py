"""Independent SHOW3D captions and searchable properties layers."""

from dataclasses import dataclass, fields
from pathlib import Path

import pyarrow as pa
import rerun as rr
from serde import serde

from dataforge import schema, writing
from dataforge.datasets.show3d_source import CAPTIONS_VERSION, HAND_POSE_VERSION, OBJECT_POSE_VERSION, IndexRow
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


@dataclass(frozen=True, slots=True)
class EpisodeProperties:
    """Stable string schema for searchable scene metadata."""

    subject_id: str
    """Upstream subject."""
    split: str
    """Dataset split."""
    object_alias: str
    """Object token from the scene ID."""
    action: str
    """Action from the scene ID."""
    hand: str
    """Caption hand selection, or empty."""
    overall_caption: str
    """Caption summary, or empty."""
    hand_pose_version: str
    """Available hand annotation version, or empty."""
    object_pose_version: str
    """Available object annotation version, or empty."""
    captions_version: str
    """Available caption version, or empty."""


def write_properties_layer(identity: SequenceIdentity, source: IndexRow, caption: Caption | None, target: Path) -> None:
    """Publish one episode property chunk with a stable string schema."""
    values: EpisodeProperties = EpisodeProperties(
        subject_id=source.subject_id,
        split=source.split,
        object_alias=source.object_alias,
        action=source.action,
        hand=caption.hand if caption else "",
        overall_caption=caption.overall_caption if caption else "",
        hand_pose_version=HAND_POSE_VERSION if source.has_hand_pose else "",
        object_pose_version=OBJECT_POSE_VERSION if source.has_object_pose else "",
        captions_version=CAPTIONS_VERSION if source.has_caption else "",
    )
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        recording.send_property("episode", rr.AnyValues(**{f.name: pa.array([getattr(values, f.name)], pa.string()) for f in fields(values)}))


def write_captions_layer(identity: SequenceIdentity, caption: Caption, target: Path) -> None:
    """Publish the static instruction and caption provenance."""
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        rr.log(schema.instruction_path(), rr.TextDocument(caption.markdown(), media_type="text/markdown"), static=True, recording=recording)
        recording.send_property(
            "captions", rr.AnyValues(version=pa.array([CAPTIONS_VERSION], type=pa.string()), hand=pa.array([caption.hand], type=pa.string()))
        )
