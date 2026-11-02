"""SelfCap layouts: the segment layout at every stage, the table card, and the preview card table."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun.catalog import DatasetEntry
from rerun.encodings import EntityPathLike

from dataforge import blueprints, schema, writing
from dataforge.schema import SOMA_MESH, SOMA_QUALITY_PROPERTY
from dataforge.selfcap_schema import (
    DEFAULT_EXO_CAMERAS,
    EGO_STREAM_LAYOUTS,
    IMU_DEVICE,
    PREVIEW_BODY,
    PREVIEW_CENTER,
    PREVIEW_EXO_PINHOLE,
    PREVIEW_IMAGE_PLANE_DISTANCE,
    PREVIEW_LAYER,
    PREVIEW_SKELETON,
    PREVIEW_SOMA_XLO,
    PREVIEW_TIMELINE,
    PREVIEW_VIDEO_GAP_PX,
    PREVIEW_VIDEO_SIZE,
    QUEST_STREAM_LAYOUTS,
    rig_label_path,
)

SCENE_BACKGROUND: rrb.BackgroundKind = rrb.BackgroundKind.GradientDark
"""Background of every selfcap 3D view."""
SCENE_EYE: rrb.EyeControls3D = rrb.EyeControls3D(
    kind=rrb.Eye3DKind.Orbital, position=[2.0, 2.9, 2.0], look_target=[0.0, 1.6, 0.0], eye_up=[0.0, 1.0, 0.0], spin_speed=0.1
)
"""Start of every selfcap 3D view, in the default layout and the catalog cards alike: about 3.2 m from the world origin
(where the wearer stands in most captures), orbiting it slowly. The viewer's own fit takes in the whole room and starts
too far out. Tracking the headset does not help: on its transform the eye stays put, on its mesh it zooms onto the mesh."""
PREVIEW_EYE: rrb.EyeControls3D = rrb.EyeControls3D(position=[0.0, 0.55, -1.9], look_target=[0.0, 0.2, 0.0], eye_up=[0.0, 1.0, 0.0], spin_speed=0.2)
"""Card eye: world is RUB (+Y up, -Z forward), so an elevated front view of the body at ``PREVIEW_CENTER``."""
TABLE_FIELDS: tuple[tuple[str, str], ...] = (
    ("property:calib:session", "Capture / session"),
    ("property:calib:exo_cameras", "Phones"),
    (f"property:{SOMA_QUALITY_PROPERTY}:status", "SOMA quality"),
)
"""Segment-table columns every layout shows, as (column, header)."""
ALWAYS_HIDDEN_COLUMNS: tuple[str, ...] = ("property:capture:collector",)
"""Hidden even when the segment table does not list them at publish time."""
TABLE_SORT_COLUMN: str = "property:calib:session"
"""Column that orders the segment table."""
HIDDEN_PINHOLE_OVERLAYS: tuple[str, ...] = ("sapiens2", "pred", "coco133_uv_projected", "quest_coco133_uv_projected")
"""Per-camera overlays kept in the view tree for comparison but hidden: the display skeleton stands for them."""
LOUDNESS_WINDOW_S: float = 2.0
"""Half-width of the loudness plot around the time cursor: a tap and its echo on the other phones fit in it."""


@dataclass(frozen=True, slots=True)
class CameraPane:
    """One camera's identity in the blueprint: where it lives and what to call it."""

    name: str
    """Human label shown on the 2D view."""
    rig: int
    """Rig index owning the camera."""
    cam: int
    """Camera index within the rig."""
    kind: str
    """Device group the pane belongs to: ``exo``, ``ego``, or ``quest``.

    Blueprints cannot select entities by AnyValues, so the exo/ego grouping the
    layout communicates has to travel with the pane list itself."""


def segment_rig_kinds(entry: DatasetEntry, rigs: range, *extra: str) -> list[tuple[dict[str, object], dict[int, str]]]:
    """Each registered segment's catalog row (its rig nodes plus ``extra`` entities) and the ``kind`` of each rig, in one query."""
    table: pa.Table = entry.filter_contents([*(schema.rig_path(rig) for rig in rigs), *extra]).reader(index=None).to_arrow_table()
    return [(row, {rig: kind[0] for rig in rigs if (kind := row.get(f"{schema.rig_path(rig)}:kind"))}) for row in table.to_pylist()]


def build_blueprint(panes: tuple[CameraPane, ...], ego_rig: int, *, posed_phones: bool = False) -> rrb.Blueprint:
    """The SelfCap layout at every stage: 3D scene and head cameras over the phones, their sound, and the OAK IMU.

    A view lists every overlay a later layer may add (Quest hands, skeleton, SOMA), so the one layout serves a
    freshly ingested capture and a processed one. Only the phone poses change what can be drawn: an unposed phone
    would sit at the world origin in 3D and project the hands through the wrong pose, so until calibration the scene
    leaves the phones out and their panes show only the video. The SOMA mesh override is always there: a segment
    without the ``soma`` layer has nothing for it to act on.

    Args:
        panes: Camera labels and indices, in rig-then-camera order.
        ego_rig: Rig index of the OAK, whose ``imu_00`` feeds the time-series views.
        posed_phones: The phones carry a calibrated pose.

    Returns:
        The blueprint embedded in every base rrd (phones unposed) and, with index labels, the catalog default.
    """
    hide: rrb.EntityBehavior = rrb.EntityBehavior(visible=False)
    phones: list[CameraPane] = [pane for pane in panes if pane.kind == "exo"]
    pinholes: list[str] = [schema.pinhole_path(pane.rig, pane.cam) for pane in panes]
    frames: list[tuple[int, str]] = [(pane.rig, schema.cam_path(pane.rig, pane.cam)) for pane in panes]
    frames += [(pane.rig, rig_label_path(pane.rig)) for pane in panes if pane.kind == "exo"]
    hidden: dict[EntityPathLike, rrb.EntityBehavior] = dict.fromkeys(
        ["/world/gt/sapiens2", "/world/pred", *(f"{pinhole}/{overlay}" for pinhole in pinholes for overlay in HIDDEN_PINHOLE_OVERLAYS)], hide
    )

    def camera(pane: CameraPane) -> rrb.Spatial2DView:
        # A pane rooted at its pinhole projects the 3D overlays it lists through the transform tree. Exact for the
        # Quest and the phones; the OAK lenses are wide and Rerun has no distortion model, so there the joints land
        # 25-80 px off, worst at the image edge.
        pinhole: str = schema.pinhole_path(pane.rig, pane.cam)
        if pane.kind == "exo" and not posed_phones:
            # A model draws only its points at or above schema.SHOWN_MIN_CONFIDENCE (schema.shown_path); the full set stays, hidden.
            full_sets: dict[EntityPathLike, rrb.EntityBehavior] = dict.fromkeys(
                (*(schema.coco133_uv_path(pane.rig, pane.cam, pred=pred) for pred in ("sapiens2", "rtmw")), f"{pinhole}/sapiens2/coco133_uv"), hide
            )
            return blueprints.camera_view(pane.name, pane.rig, pane.cam, contents=[f"{pinhole}/**"], overrides=full_sets)
        contents: list[str] = [f"{pinhole}/**", "/world/gt/**", "/world/pred/**"]
        overrides: dict[EntityPathLike, rrb.EntityBehavior] = hidden.copy()
        if pane.rig != ego_rig:
            contents.append("/world/skeleton/**")
            overrides[SOMA_MESH] = rrb.EntityBehavior(visible=True)
        if pane.kind == "exo":
            # The other rigs' camera frames and names (cam nodes only, without their frustum or video) show in the phone.
            contents.extend(path for rig, path in frames if rig != pane.rig)
        return blueprints.camera_view(pane.name, pane.rig, pane.cam, contents=contents, overrides=overrides)

    # The head cameras show as bare frustums: their image planes would cover the hands. A posed phone shows its video
    # in its frustum; an unposed one would sit at the world origin. Camera-plane overlays would repeat the body
    # around the world skeleton (hidden below).
    heads: list[str] = [schema.pinhole_path(pane.rig, pane.cam) for pane in panes if pane.kind != "exo"]
    excluded: list[str] = [
        *(rule for pinhole in heads for rule in (f"+ {pinhole}", f"- {pinhole}/**")),
        *([] if posed_phones else [f"- {schema.rig_path(pane.rig)}/**" for pane in phones]),
    ]
    scene_overrides: dict[EntityPathLike, rrb.EntityBehavior] = hidden | dict.fromkeys(
        (f"{pinhole}/{overlay}" for pinhole in pinholes for overlay in ("skeleton", "camera_centers")), hide
    )
    scene_overrides[SOMA_MESH] = rrb.EntityBehavior(visible=True)
    scene = rrb.Spatial3DView(name="Scene", origin="/world", contents=["/world/**", *excluded], overrides=scene_overrides, background=SCENE_BACKGROUND,
                              eye_controls=SCENE_EYE)

    oak: list[rrb.Spatial2DView] = [camera(pane) for pane in panes if pane.kind == "ego"]
    quest: list[rrb.Spatial2DView] = [camera(pane) for pane in panes if pane.kind == "quest"]
    # Sound starts muted: before alignment the phones play the same tap up to seconds apart.
    strip = rrb.Horizontal(
        *(
            rrb.Vertical(
                camera(pane),
                rrb.AudioView(origin=schema.audio_path(pane.rig, 0), name=f"{pane.name} · sound", playback=rrb.AudioPlayback(volume=0.0)),
                row_shares=[4.0, 1.0],
            )
            for pane in phones
        ),
        name="Phones",
    )
    window = rrb.VisibleTimeRange(
        schema.TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-LOUDNESS_WINDOW_S),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=LOUDNESS_WINDOW_S),
    )
    loudness = rrb.TimeSeriesView(
        name="Loudness per phone (dB over its own floor)",
        contents=[schema.loudness_path(pane.rig, 0) for pane in phones],
        axis_y=rrb.ScalarAxis(range=(0.0, 45.0)),
        time_ranges=[window],
        plot_legend=rrb.PlotLegend(visible=True),
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                scene,
                rrb.Vertical(oak[0], rrb.Horizontal(*oak[1:]), rrb.Horizontal(*quest), name="Head cameras", row_shares=[1.3, 1.0, 1.0]),
                column_shares=[2.2, 1.0],
            ),
            strip,
            loudness,
            rrb.Horizontal(
                blueprints.sensor_plot("Gyroscope", schema.imu_path(ego_rig, IMU_DEVICE), schema.gyro_path(ego_rig, IMU_DEVICE)),
                blueprints.sensor_plot("Accelerometer", schema.imu_path(ego_rig, IMU_DEVICE), schema.accel_path(ego_rig, IMU_DEVICE)),
            ),
            row_shares=[3.0, 2.0, 0.8, 0.8],
        ),
        # An explicit TimePanel is not collapsed by collapse_panels, so it states its own state.
        rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
        collapse_panels=True,
    )


def default_blueprint(*, posed_phones: bool = True) -> rrb.Blueprint:
    """``build_blueprint`` for the modal four-phone rig, with index labels: device names vary per session.

    Register publishes it with ``posed_phones`` read from the catalog (``catalog_blueprints``); the default suits
    the callers that run after calibration.
    """
    phones: list[CameraPane] = [CameraPane(f"Phone {rig + 1}", rig, 0, "exo") for rig in range(DEFAULT_EXO_CAMERAS)]
    ego_rig, quest_rig = DEFAULT_EXO_CAMERAS, DEFAULT_EXO_CAMERAS + 1
    head: list[CameraPane] = [CameraPane(f"OAK {stream}", ego_rig, cam, "ego") for cam, (stream, _) in enumerate(EGO_STREAM_LAYOUTS)]
    head += [CameraPane(f"Quest {stream}", quest_rig, cam, "quest") for cam, (stream, _) in enumerate(QUEST_STREAM_LAYOUTS)]
    return build_blueprint((*phones, *head), ego_rig, posed_phones=posed_phones)


def table_blueprint(*, posed_phones: bool = True) -> rrb.Blueprint:
    """The scene beside one camera: the OAK with the Quest hands, or once the phones are posed, phone 1 with the body.

    Rig 00 exists in both three- and four-phone sessions. The scene excludes all possible video slots, so each
    card decodes only its pane's stream.
    """
    slots: list[tuple[int, int]] = [(rig, cam) for rig in range(DEFAULT_EXO_CAMERAS + 2) for cam in range(3)]
    unposed: list[str] = [] if posed_phones else [f"- {schema.rig_path(rig)}/**" for rig in range(DEFAULT_EXO_CAMERAS)]
    scene: rrb.Spatial3DView = rrb.Spatial3DView(
        name="Scene",
        origin="/world",
        contents=["/world/**", "- /world/pred/**", f"{SOMA_MESH}/**", *blueprints.video_exclusions(slots), *unposed],
        background=SCENE_BACKGROUND,
        eye_controls=SCENE_EYE,
    )
    if posed_phones:
        body: list[str] = ["/world/gt/**", "/world/skeleton/**", f"{SOMA_MESH}/**"]
        pane: rrb.Spatial2DView = blueprints.camera_view("Phone 1", 0, 0, contents=[schema.video_path(0, 0) + "/**", *body])
    else:
        ego_rig: int = DEFAULT_EXO_CAMERAS
        pane = blueprints.camera_view("OAK rgb + Quest hands", ego_rig, 0, contents=[schema.video_path(ego_rig, 0) + "/**", "/world/gt/**"])
    return blueprints.exoego_table_blueprint(scene, pane)


def preview_table_blueprint(columns: list[str], fields: writing.TableFields) -> rrb.table.TableBlueprint:
    """The card table of a dataset whose segments hold the ``preview`` layer: previews loop on ``preview_time``.

    A card shows a 3D "Body" view and a 2D "Exo + ego" view. The fields are ``table_fields``; see ``writing.published_columns``.

    Args:
        columns: Every column of the segment table at publish time.
    fields: The dataset's ``table_fields``.
    """
    # Origin = /preview/center (the window's mean body position), so a fixed camera fits every session. The rigs come
    # from /world: pinhole frustums (and device frames), without the pinholes' 2D children (videos, overlays). The
    # moving Quest pose is on preview_time in the preview layer; the OAK mount hangs off it.
    slots: list[tuple[int, int]] = [(rig, cam) for rig in range(DEFAULT_EXO_CAMERAS + 2) for cam in range(3)]
    contents: list[str] = [f"+ {PREVIEW_BODY}/**"]
    for rig in range(DEFAULT_EXO_CAMERAS + 2):
        contents.append(f"+ {schema.rig_path(rig)}/**")
        for cam in range(3):
            pinhole: str = schema.pinhole_path(rig, cam)
            contents += [f"- {pinhole}/**", f"+ {pinhole}"]
    planes: dict[EntityPathLike, rr.Pinhole] = {schema.pinhole_path(rig, cam): rr.Pinhole.from_fields(image_plane_distance=PREVIEW_IMAGE_PLANE_DISTANCE) for rig, cam in slots}
    body: rrb.Spatial3DView = rrb.Spatial3DView(
        name="Body", origin=PREVIEW_CENTER, contents=contents, line_grid=False, overrides=planes, background=SCENE_BACKGROUND, eye_controls=PREVIEW_EYE
    )
    # The exo video sits under a scaled copy of the exo phone's pinhole, so the 3D skeleton and xlo mesh project onto it
    # like in the main view; the OAK video and its 2D skeleton are stacked below it in the same image space.
    width, height = PREVIEW_VIDEO_SIZE
    videos: rrb.Spatial2DView = rrb.Spatial2DView(
        name="Exo + ego",
        origin=PREVIEW_EXO_PINHOLE,
        contents=[f"+ {PREVIEW_EXO_PINHOLE}/**", f"+ {PREVIEW_SKELETON}/**", f"+ {PREVIEW_SOMA_XLO}/**"],
        visual_bounds=rrb.VisualBounds2D(x_range=[0, width], y_range=[0, 2 * height + PREVIEW_VIDEO_GAP_PX]),
    )
    preview: rrb.table.Column = rrb.table.Column(writing.SEGMENT_LINK_COLUMN, column_type=rrb.table.PreviewCell(body, videos))
    published: writing.PublishedColumns = writing.published_columns(fields, columns)

    def layout_columns(shown: tuple[writing.TableField, ...], hidden: tuple[str, ...]) -> list[rrb.table.Column]:
        return [
            preview,
            *(rrb.table.Column(field.column, name=field.name) for field in shown),
            *(rrb.table.Column(column, visible=False) for column in hidden),
        ]

    return rrb.table.TableBlueprint(
        table_layout=rrb.table.TableLayout(columns=layout_columns(fields.table, published.table_hidden)),
        card_layout=rrb.table.CardLayout(
            title="property:RecordingInfo:name", link=writing.SEGMENT_LINK_COLUMN, fields=layout_columns(fields.cards, published.cards_hidden)
        ),
        default_layout="cards",
        previews_config=rrb.table.PreviewsConfig(timeline=PREVIEW_TIMELINE, start_point=0.0, play_state="Playing"),
        sort_by=None if published.sort_by is None else rrb.table.SortBy(published.sort_by, direction="Ascending"),
    )


def catalog_blueprints(entry: DatasetEntry, fields: writing.TableFields) -> tuple[rrb.Blueprint, rrb.Blueprint | rrb.table.TableBlueprint]:
    """Both layouts, as the layers on the catalog allow.

    The catalog holds one default layout for all its segments, and an unposed phone drawn in 3D would sit at the
    world origin, so the phones are posed only when every phone of every segment has a pose (calibrate's ``cam_00``):
    one uncalibrated phone anywhere keeps the phones out of 3D. One query reads every segment. The table is the
    preview card when any segment holds ``preview``, so one failed segment does not turn the cards off for all.
    """
    rigs: range = range(DEFAULT_EXO_CAMERAS)
    posed: list[bool] = [
        row.get(f"{schema.cam_path(rig, 0)}:Transform3D:translation") is not None
        for row, kinds in segment_rig_kinds(entry, rigs, *(schema.cam_path(rig, 0) for rig in rigs))
        for rig, kind in kinds.items()
        if kind == "exo"
    ]
    posed_phones: bool = bool(posed) and all(posed)
    segment_layout: rrb.Blueprint = default_blueprint(posed_phones=posed_phones)
    segments: pa.Table = entry.segment_table().to_arrow_table()
    if any(PREVIEW_LAYER in layer_names for layer_names in segments["rerun_layer_names"].to_pylist()):
        return segment_layout, preview_table_blueprint(segments.schema.names, fields)
    return segment_layout, table_blueprint(posed_phones=posed_phones)


def table_fields() -> writing.TableFields:
    """Session, phone count and SOMA quality at a glance; every other column stays selectable in the viewer's menu.

    The calibration session combines capture date and session prefix, and orders the rows. The collector's name
    stays hidden even in a catalog that does not hold the column yet.
    """
    fields: tuple[writing.TableField, ...] = tuple(writing.TableField(column, name) for column, name in TABLE_FIELDS)
    return writing.TableFields(cards=fields, table=fields, hidden=ALWAYS_HIDDEN_COLUMNS, sort_by=TABLE_SORT_COLUMN)
