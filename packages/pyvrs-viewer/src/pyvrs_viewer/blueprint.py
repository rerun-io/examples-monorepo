"""Dynamic Rerun blueprint generation based on discovered VRS streams.

Creates a layout with camera views, per-frame metadata, configuration text,
and IMU time series plots.
"""

import rerun.blueprint as rrb


def create_vrs_blueprint(
    camera_entities: list[str],
    imu_entities: list[str],
) -> rrb.BlueprintLike:
    """Create a Rerun blueprint based on discovered VRS streams.

    Layout:
      - Cameras: each camera gets a Spatial2DView + TextDocumentView for its
        per-frame metadata and static configuration
      - IMU: TimeSeriesViews for accelerometer and gyroscope
      - Overall: cameras on top, metadata in a tab, IMU below

    Args:
        camera_entities: Entity path names for camera streams.
        imu_entities: Entity path names for IMU streams.

    Returns:
        A Rerun blueprint ready to send via rr.send_blueprint().
    """
    # Camera image views
    camera_views: list[rrb.SpaceView] = [
        rrb.Spatial2DView(origin=entity, name=entity)
        for entity in camera_entities
    ]

    # Per-stream text metadata views (configuration + per-frame data)
    text_views: list[rrb.SpaceView] = []
    for entity in camera_entities:
        text_views.append(rrb.TextDocumentView(origin=f"{entity}/configuration", name=f"{entity}/config"))
        text_views.append(rrb.TextDocumentView(origin=f"{entity}/data", name=f"{entity}/metadata"))
    for entity in imu_entities:
        text_views.append(rrb.TextDocumentView(origin=f"{entity}/configuration", name=f"{entity}/config"))

    # IMU time series views
    imu_views: list[rrb.SpaceView] = []
    for entity in imu_entities:
        imu_views.append(rrb.TimeSeriesView(origin=f"{entity}/accelerometer", name=f"{entity}/accel"))
        imu_views.append(rrb.TimeSeriesView(origin=f"{entity}/gyroscope", name=f"{entity}/gyro"))

    # Build layout sections
    sections: list[rrb.Container | rrb.SpaceView] = []
    section_shares: list[int] = []

    if camera_views:
        grid_columns: int = min(len(camera_views), 2)
        sections.append(rrb.Grid(*camera_views, grid_columns=grid_columns))
        section_shares.append(3)

    if imu_views:
        sections.append(rrb.Grid(*imu_views, grid_columns=min(len(imu_views), 2)))
        section_shares.append(1)

    if text_views:
        sections.append(rrb.Tabs(*text_views, name="Metadata"))
        section_shares.append(1)

    if not sections:
        return rrb.Blueprint(auto_space_views=True)

    if len(sections) == 1:
        return rrb.Blueprint(sections[0], collapse_panels=True)

    return rrb.Blueprint(
        rrb.Vertical(*sections, row_shares=section_shares),
        collapse_panels=True,
    )
