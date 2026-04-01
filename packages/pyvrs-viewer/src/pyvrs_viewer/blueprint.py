"""Dynamic Rerun blueprint generation based on discovered VRS streams.

Creates a layout with camera views on top and IMU time series plots below.
"""

import rerun.blueprint as rrb


def create_vrs_blueprint(
    camera_entities: list[str],
    imu_entities: list[str],
) -> rrb.BlueprintLike:
    """Create a Rerun blueprint based on discovered VRS streams.

    Layout:
      - Cameras only: Grid with auto columns
      - Cameras + IMU: Vertical split — cameras on top, IMU below
      - IMU only: Horizontal layout of time series

    Args:
        camera_entities: Entity path names for camera streams.
        imu_entities: Entity path names for IMU streams.

    Returns:
        A Rerun blueprint ready to send via rr.send_blueprint().
    """
    camera_views: list[rrb.SpaceView] = [
        rrb.Spatial2DView(
            origin=entity,
            name=entity,
        )
        for entity in camera_entities
    ]

    imu_views: list[rrb.SpaceView] = []
    for entity in imu_entities:
        imu_views.append(
            rrb.TimeSeriesView(
                origin=f"{entity}/accelerometer",
                name=f"{entity}/accel",
            )
        )
        imu_views.append(
            rrb.TimeSeriesView(
                origin=f"{entity}/gyroscope",
                name=f"{entity}/gyro",
            )
        )

    # Build layout based on what streams are available
    contents: list[rrb.Container | rrb.SpaceView] = []

    if camera_views:
        grid_columns: int = min(len(camera_views), 2)
        contents.append(
            rrb.Grid(
                *camera_views,
                grid_columns=grid_columns,
            )
        )

    if imu_views:
        contents.append(
            rrb.Grid(
                *imu_views,
                grid_columns=min(len(imu_views), 2),
            )
        )

    if not contents:
        return rrb.Blueprint(auto_space_views=True)

    if len(contents) == 1:
        return rrb.Blueprint(contents[0], collapse_panels=True)

    # Cameras take more space than IMU plots
    return rrb.Blueprint(
        rrb.Vertical(
            *contents,
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
