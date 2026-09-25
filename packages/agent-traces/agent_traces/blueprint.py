"""Default viewer layout for one converted session."""

import rerun.blueprint as rrb
from rerun import encodings


def session_blueprint() -> rrb.Blueprint:
    """Build the layout baked into every session recording.

    Returns:
        Conversation and tool logs on the left, images and token plots on the right,
        with the wall timeline expanded underneath.
    """
    conversation: rrb.Tabs = rrb.Tabs(
        rrb.TextLogView(
            name="Conversation",
            origin="/",
            contents=["+ /conversation/**", "+ /agents/**/conversation/**", "- /conversation/thinking", "- /agents/**/conversation/thinking"],
        ),
        rrb.TextLogView(name="Thinking", origin="/", contents=["+ /conversation/thinking", "+ /agents/**/conversation/thinking"]),
    )
    tools: rrb.Tabs = rrb.Tabs(
        rrb.TextLogView(
            name="Tools",
            origin="/",
            contents=["+ /tools/**", "+ /agents/**/tools/**", "- /tools/elapsed_ms/**", "- /agents/**/tools/elapsed_ms/**"],
        ),
        rrb.TextLogView(name="Lifecycle", origin="/", contents=["+ /lifecycle/**", "+ /agents/**/lifecycle/**"]),
    )
    whole_session: rrb.archetypes.TimeAxis = rrb.archetypes.TimeAxis(
        view_range=encodings.TimeRange(encodings.TimeRangeBoundary.infinite(), encodings.TimeRangeBoundary.infinite())
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(conversation, tools, row_shares=[3, 2]),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Images", origin="/media/images"),
                    rrb.TimeSeriesView(
                        name="Tokens per request",
                        origin="/usage",
                        contents=["+ /usage/input_tokens", "+ /usage/output_tokens", "+ /usage/thinking_tokens"],
                        axis_x=whole_session,
                    ),
                    rrb.TimeSeriesView(
                        name="Cache tokens",
                        origin="/usage",
                        contents=["+ /usage/cache_read_tokens", "+ /usage/cache_creation_tokens"],
                        axis_x=whole_session,
                    ),
                    rrb.TimeSeriesView(name="Tool elapsed (ms)", origin="/tools/elapsed_ms", axis_x=whole_session),
                    row_shares=[3, 2, 2, 2],
                ),
                column_shares=[3, 2],
            ),
            rrb.DataframeView(
                name="Turns", origin="/turns", contents=["+ /turns"], query=rrb.archetypes.DataframeQuery(timeline="wall", apply_latest_at=False)
            ),
            row_shares=[4, 1],
        ),
        rrb.TimePanel(state="expanded", timeline="wall"),
    )
