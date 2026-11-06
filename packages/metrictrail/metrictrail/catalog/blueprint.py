"""MetricTrail's two catalog layouts, built on dataforge's catalog-blueprint hook (``DataforgeDataset.catalog_blueprints``).

dataforge's VSLAM-LAB layout is generic: ground truth plus every prediction layer under ``/world/runs`` in a palette of
its own. MetricTrail owns the layout of the datasets it stacks onto, so it subclasses the dataset, overrides the hook,
and publishes with ``dataforge.apis.register.republish_blueprints``:

* ``default`` (the dataset default): ground truth (green) and ``metrictrail``, ours, the shipped TensorRT path (magenta).
  The 3D views hide every other run; the panes are rgb, the sensor depth, our keyframe depth and, where the layer has
  them, our keyframe normals.
* ``comparison`` (on demand): ground truth, ``metrictrail``, ``metrictrail_paper`` (the torch reproduction, violet) and
  ``metrictrail_upstream`` (unmodified upstream ViPE, amber), with a depth pane per run.

A run the layout names but the dataset does not hold gets a pane whose title says it was not run there (7-Scenes, KITTI
and LaMAria have no shipped-path run). dataforge builds the views; this module picks the runs, the legend and the pane
titles, and hides the runs a layout leaves out (``restyle``).

``dataforge-lamaria`` is dataforge's own dataset (10 segments; MetricTrail has a layer on ``lamaria__R_01_easy`` only),
so its default layout stays dataforge's: ``default`` is refused there and ``comparison`` is published only on explicit
request. Restore dataforge's layout afterwards with ``dataforge-publish-blueprints ... lamaria``.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import rerun.blueprint as rrb
from dataforge import schema
from dataforge.apis.register import republish_blueprints
from dataforge.datasets.base import DataforgeDataset
from dataforge.datasets.lamaria import LamariaConfig, LamariaDataset
from dataforge.datasets.vslam_lab import Prediction, VslamLabConfig, VslamLabDataset, find_predictions
from rerun.catalog import CatalogClient, DatasetEntry

from metrictrail.catalog.layer import LAYER_COLORS, LayerSource, normals_entities, run_root
from metrictrail.palette import GT_RGB

LayoutName: TypeAlias = Literal["default", "comparison"]
"""``default``: ground truth and ours; ``comparison``: ground truth and every MetricTrail run (development view)."""
LAYOUT_SOURCES: dict[LayoutName, tuple[LayerSource, ...]] = {
    "default": ("metrictrail",),
    "comparison": ("metrictrail", "metrictrail_paper", "metrictrail_upstream"),
}
"""The runs each layout shows, in pane order."""
GT_SOURCE: str = "gt"
"""dataforge's ground-truth run source (``/world/runs/gt``)."""
PATH_COLORS: dict[str, tuple[int, int, int]] = {GT_SOURCE: GT_RGB, **LAYER_COLORS}
"""Path colour per run source: the explainer legend."""
SOURCE_TITLES: dict[LayerSource, str] = {
    "metrictrail": "ours",
    "metrictrail_paper": "torch port",
    "metrictrail_upstream": "upstream ViPE",
}
"""Pane-title prefix per run (short: a pane is a quarter of the camera grid wide)."""
NOT_RUN: str = "not run on this dataset"
"""Title suffix of the pane of a run the dataset does not hold."""
SHARED_DATASETS: frozenset[str] = frozenset({"dataforge-lamaria"})
"""Datasets whose default layout dataforge owns: MetricTrail publishes only ``comparison`` there, on request."""


def depth_entity(source: LayerSource) -> str:
    """The keyframe-depth entity of a MetricTrail layer."""
    return f"{run_root(source)}/cam/pinhole/depth"


def pane_title(source: LayerSource, entity: str, present: bool) -> str:
    """``<run>: keyframe <depth|normals>``, or ``<run>: not run on this dataset``."""
    return f"{SOURCE_TITLES[source]}: keyframe {entity.rsplit('/', 1)[1]}" if present else f"{SOURCE_TITLES[source]}: {NOT_RUN}"


def shown_predictions(found: Sequence[Prediction], layout: LayoutName) -> tuple[Prediction, ...]:
    """The layout's runs as dataforge predictions, in layout order; a run the dataset lacks gets its depth pane only."""
    by_source: dict[str, Prediction] = {prediction.source: prediction for prediction in found}
    return tuple(by_source.get(source, Prediction(source, depth=(depth_entity(source),))) for source in LAYOUT_SOURCES[layout])


def iter_views(item: object) -> Iterable[rrb.View]:
    """Every view under a blueprint, container or view."""
    if isinstance(item, rrb.Blueprint):
        yield from iter_views(item.root_container)
    elif isinstance(item, rrb.View):
        yield item
    elif isinstance(item, rrb.Container):
        for child in item.contents:
            yield from iter_views(child)


def content_rules(contents: object) -> list[str]:
    """A view's ``contents`` as a list of rules (dataforge writes a newline-joined string or a list)."""
    if contents is None:
        return ["$origin/**"]
    if isinstance(contents, str):
        return [line for line in contents.splitlines() if line.strip()]
    return [str(rule) for rule in contents]  # pyrefly: ignore[not-iterable]


def restyle(blueprint: rrb.Blueprint, layout: LayoutName, present: set[str], held: set[str], with_normals: set[str]) -> rrb.Blueprint:
    """Hide, in the 3D views, the runs ``held`` by the dataset that ``layout`` leaves out, and title the MetricTrail panes.

    A run with surface normals shows its normals-coloured cloud in the overview (origin ``/``) and its RGB cloud in the
    other 3D views. ``present``: the shown runs the dataset holds (the others' panes say they were not run).
    """
    shown: tuple[LayerSource, ...] = LAYOUT_SOURCES[layout]
    hidden: list[str] = sorted(source for source in held if source != GT_SOURCE and source not in shown)
    for view in iter_views(blueprint):
        if isinstance(view, rrb.Spatial3DView):
            overview: bool = str(view.origin) == "/"
            normals_rules: list[str] = [
                f"- {run_root(source)}/points/**" if overview else f"- {normals_entities(source)['points_normals']}/**"
                for source in shown
                if source in with_normals
            ]
            view.contents = [*content_rules(view.contents), *(f"- {schema.run_path(source)}/**" for source in hidden), *normals_rules]
        elif isinstance(view, rrb.Spatial2DView):
            for source in shown:
                rules: list[str] = content_rules(view.contents)
                if str(view.origin).startswith(f"{run_root(source)}/") and rules:
                    view.name = pane_title(source, rules[0], source in present)
    return blueprint


def held_runs(entry: DatasetEntry) -> tuple[tuple[Prediction, ...], set[str]]:
    """The dataset's prediction layers (dataforge's ``find_predictions`` over its schema) and the runs with normals."""
    found: tuple[Prediction, ...] = find_predictions((column.entity_path, column.archetype) for column in entry.schema().component_columns())
    with_normals: set[str] = {prediction.source for prediction in found if any(entity.endswith("/normals") for entity in prediction.images)}
    return found, with_normals


class MetricTrailVslamLab(VslamLabDataset):
    """A dataforge VSLAM-LAB dataset whose catalog layouts are MetricTrail's (``layout``)."""

    def __init__(self, config: VslamLabConfig, layout: LayoutName) -> None:
        super().__init__(config)
        self.layout: LayoutName = layout

    def catalog_blueprints(self, entry: DatasetEntry) -> tuple[rrb.Blueprint, rrb.Blueprint]:
        """dataforge's segment and card layouts with the layout's runs, the legend, and the other runs hidden."""
        found, with_normals = held_runs(entry)
        return self.layouts(found, with_normals)

    def layouts(self, found: Sequence[Prediction], with_normals: set[str]) -> tuple[rrb.Blueprint, rrb.Blueprint]:
        """Both layouts from the prediction layers the dataset holds (no catalog access)."""
        shown: tuple[Prediction, ...] = shown_predictions(found, self.layout)
        present: set[str] = {prediction.source for prediction in found}
        colors: Mapping[str, tuple[int, int, int]] = {source: PATH_COLORS[source] for source in (GT_SOURCE, *LAYOUT_SOURCES[self.layout])}
        return (
            restyle(self.segment_blueprint(shown, colors), self.layout, present, present, with_normals),
            restyle(self.card_blueprint(shown, colors), self.layout, present, present, with_normals),
        )


class MetricTrailLamaria(LamariaDataset):
    """dataforge's LaMAria dataset with MetricTrail's layouts: its own layout plus a depth pane per shown run.

    LaMAria's layout takes no prediction layers, so the runs keep their layers' own (legend) colours and the panes are
    added to its camera grid. The dataset is dataforge's: :func:`metrictrail_dataset` gives only ``comparison`` here.
    """

    def __init__(self, config: LamariaConfig, layout: LayoutName) -> None:
        super().__init__(config)
        self.layout: LayoutName = layout

    def catalog_blueprints(self, entry: DatasetEntry) -> tuple[rrb.Blueprint, rrb.Blueprint]:
        """See :class:`MetricTrailLamaria`."""
        found, with_normals = held_runs(entry)
        return self.layouts(found, with_normals)

    def layouts(self, found: Sequence[Prediction], with_normals: set[str]) -> tuple[rrb.Blueprint, rrb.Blueprint]:
        """Both layouts from the prediction layers the dataset holds (no catalog access)."""
        present: set[str] = {prediction.source for prediction in found}
        segment: rrb.Blueprint = self.default_blueprint()
        grids: list[rrb.Grid] = [item for item in iter_containers(segment) if isinstance(item, rrb.Grid)]
        if len(grids) != 1:
            raise ValueError(f"LaMAria's layout has {len(grids)} camera grids; expected one")
        for prediction in shown_predictions(found, self.layout):
            for entity in (*prediction.depth, *prediction.images):
                parent: str = entity.rsplit("/", 1)[0]
                grids[0].contents = (*grids[0].contents, rrb.Spatial2DView(name=entity, origin=parent, contents=[entity]))
        card: rrb.Blueprint = self.table_blueprint()
        return restyle(segment, self.layout, present, present, with_normals), restyle(card, self.layout, present, present, with_normals)


def iter_containers(item: object) -> Iterable[rrb.Container]:
    """Every container under a blueprint or container."""
    if isinstance(item, rrb.Blueprint):
        yield from iter_containers(item.root_container)
    elif isinstance(item, rrb.Container):
        yield item
        for child in item.contents:
            yield from iter_containers(child)


def metrictrail_dataset(catalog_name: str, layout: LayoutName) -> DataforgeDataset:
    """The dataforge dataset behind ``catalog_name`` with MetricTrail's ``layout``.

    Raises:
        ValueError: If MetricTrail has no layout for that dataset, or ``layout`` is ``default`` on a shared dataset.
    """
    if catalog_name in SHARED_DATASETS and layout == "default":
        raise ValueError(
            f"{catalog_name} keeps dataforge's default layout; MetricTrail publishes only --publish-blueprint comparison there"
        )
    if catalog_name == "dataforge-lamaria":
        return MetricTrailLamaria(LamariaConfig(), layout)
    prefix: str = "dataforge-vslam-lab-"
    if not catalog_name.startswith(prefix):
        raise ValueError(f"{catalog_name}: MetricTrail layouts exist for dataforge-vslam-lab-<benchmark> and dataforge-lamaria only")
    return MetricTrailVslamLab(VslamLabConfig(benchmark=catalog_name.removeprefix(prefix)), layout)  # pyrefly: ignore[bad-argument-type]


def publish_layout(catalog_url: str, catalog_name: str, layout: LayoutName, output_root: Path) -> None:
    """Publish ``layout`` as the dataset's default segment and card layouts (dataforge's ``republish_blueprints``: new
    stamped ``<output_root>/blueprints/*.rbl`` files, the dataset's earlier blueprint entries retired).

    Raises:
        ValueError: Before any catalog access, if MetricTrail may not publish ``layout`` on ``catalog_name``
            (``default`` on ``dataforge-lamaria``, whose default layout is dataforge's).
    """
    dataset: DataforgeDataset = metrictrail_dataset(catalog_name, layout)
    entry: DatasetEntry = CatalogClient(catalog_url).get_dataset(catalog_name)
    republish_blueprints(entry, dataset, catalog_name, output_root)
    for path in (output_root / "blueprints").glob(f"{catalog_name}*.rbl"):
        path.chmod(0o664)  # the ssd-8tb NAS share gives new files mode 000
