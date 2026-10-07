"""Default viewer layout for one converted session or the full catalog."""

from collections.abc import Collection

import rerun.blueprint as rrb
from rerun import encodings


def session_blueprint(entities: Collection[str]) -> rrb.Blueprint:
    """Show only entity families present in one recording."""
    def has_data(*prefixes: str) -> bool:
        """Match shared text paths and scalar series, including their /children siblings."""
        return any(entity == prefix or entity.startswith(f"{prefix}/") for entity in entities for prefix in prefixes)

    conversation: list[rrb.View] = []
    if has_data("conversation/user", "conversation/assistant", "conversation/injected", "conversation/compaction", "conversation/inter_agent"):
        conversation.append(rrb.TextLogView(
            name="Conversation", origin="/",
            contents=["+ /conversation/**", "- /conversation/thinking/**", "- /conversation/current"],
        ))
    if has_data("conversation/thinking"):
        conversation.append(rrb.TextLogView(name="Thinking", origin="/", contents=["+ /conversation/thinking/**"]))
    messages: list[rrb.Container | rrb.View] = [rrb.Tabs(*conversation, active_tab=0)] if conversation else []
    if has_data("conversation/current"):
        messages.append(rrb.TextDocumentView(name="Current message", origin="/conversation/current"))

    tools: list[rrb.View] = []
    if has_data("tools"):
        tools.append(rrb.TextLogView(name="Tools", origin="/", contents=["+ /tools"]))
    for name, prefix in (("Executions", "executions"), ("Context", "context")):
        if has_data(prefix):
            tools.append(rrb.TextLogView(name=name, origin="/", contents=[f"+ /{prefix}/**"]))

    if has_data("lifecycle"):
        tools.append(rrb.TextLogView(name="Lifecycle", origin="/", contents=["+ /lifecycle/**"]))

    whole_session: rrb.archetypes.TimeAxis = rrb.archetypes.TimeAxis(
        view_range=encodings.TimeRange(encodings.TimeRangeBoundary.infinite(), encodings.TimeRangeBoundary.infinite())
    )
    plots: list[rrb.View] = []
    if has_data("media/images"):
        plots.append(rrb.Spatial2DView(name="Images", origin="/media/images"))
    if has_data("usage/input_tokens", "usage/output_tokens", "usage/thinking_tokens"):
        plots.append(rrb.TimeSeriesView(
            name="Tokens per request", origin="/usage",
            contents=["+ /usage/input_tokens/**", "+ /usage/output_tokens/**", "+ /usage/thinking_tokens/**"], axis_x=whole_session,
        ))
    if has_data("usage/cache_read_tokens", "usage/cache_creation_tokens"):
        plots.append(rrb.TimeSeriesView(
            name="Cache tokens", origin="/usage",
            contents=["+ /usage/cache_read_tokens/**", "+ /usage/cache_creation_tokens/**"], axis_x=whole_session,
        ))
    if has_data("elapsed/tools"):
        plots.append(rrb.TimeSeriesView(name="Tool elapsed (ms)", origin="/elapsed/tools", axis_x=whole_session))

    left: list[rrb.Container | rrb.View] = []
    if messages:
        left.append(rrb.Horizontal(*messages))
    if tools:
        left.append(rrb.Tabs(*tools, active_tab=0))
    columns: list[rrb.Container] = []
    if left:
        columns.append(rrb.Vertical(*left, row_shares=[3, 2][:len(left)]))
    if plots:
        columns.append(rrb.Vertical(*plots))
    layout: list[rrb.Container | rrb.View] = []
    if columns:
        layout.append(rrb.Horizontal(*columns, column_shares=[3, 2][:len(columns)]))
    if has_data("turns"):
        layout.append(rrb.DataframeView(
            name="Turns", origin="/turns", contents=["+ /turns"],
            query=rrb.archetypes.DataframeQuery(timeline="wall", apply_latest_at=False),
        ))
    return rrb.Blueprint(
        *([rrb.Vertical(*layout, row_shares=[4, 1][:len(layout)])] if layout else []),
        rrb.TimePanel(state="expanded", timeline="wall"), auto_views=False, auto_layout=False,
    )


def catalog_blueprint() -> rrb.Blueprint:
    """Show every supported family for browsing a catalog of recordings."""
    return session_blueprint({
        "conversation/user", "conversation/thinking", "conversation/current", "tools", "lifecycle", "media/images",
        "usage/input_tokens", "usage/cache_read_tokens", "elapsed/tools", "turns", "executions", "context",
    })
