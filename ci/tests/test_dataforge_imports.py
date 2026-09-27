"""dataforge layering: a dataset imports only its own modules and the shared ones; shared modules import no dataset.

The review stack lands shared modules first and then one dataset per PR, so every
prefix of it must import cleanly; the datasets are therefore read off the files
present, never listed here. A dataset owns ``datasets/<dataset>.py`` and every
``datasets/<dataset>_*.py``; ``datasets/base.py`` is shared, and ``datasets/__init__.py``
is the registry that imports them all.
"""

import ast
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATAFORGE_DIR: Path = REPO_ROOT / "packages" / "dataforge" / "dataforge"
DATASETS_DIR: Path = DATAFORGE_DIR / "datasets"
DATASETS_PACKAGE: str = "dataforge.datasets"
SHARED_DATASET_MODULES: frozenset[str] = frozenset({"base"})


def _imported_dataset_modules(path: Path) -> set[str]:
    """Stems of every dataforge.datasets.<stem> the file imports, at any depth (lazy imports included)."""
    stems: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            modules = [node.module] if node.module != DATASETS_PACKAGE else [f"{DATASETS_PACKAGE}.{alias.name}" for alias in node.names]
        else:
            continue
        stems.update(module.split(".")[2] for module in modules if module.startswith(f"{DATASETS_PACKAGE}."))
    return stems


def _dataset_owner(stem: str, datasets: set[str]) -> str:
    return next(dataset for dataset in datasets if stem == dataset or stem.startswith(f"{dataset}_"))


def test_datasets_import_no_other_dataset() -> None:
    stems: set[str] = {path.stem for path in DATASETS_DIR.glob("*.py")} - {"__init__"} - SHARED_DATASET_MODULES
    datasets: set[str] = {stem for stem in stems if not any(stem.startswith(f"{other}_") for other in stems if other != stem)}
    assert datasets, f"no dataset modules under {DATASETS_DIR}"
    problems: list[str] = []
    for stem in sorted(stems):
        owner: str = _dataset_owner(stem, datasets)
        for imported in sorted(_imported_dataset_modules(DATASETS_DIR / f"{stem}.py") - SHARED_DATASET_MODULES):
            if imported not in stems or _dataset_owner(imported, datasets) != owner:
                problems.append(f"datasets/{stem}.py ({owner}) imports dataforge.datasets.{imported}")
    assert not problems, "a dataset imports another dataset's module; move the shared code into dataforge/:\n" + "\n".join(problems)


def test_shared_modules_import_no_dataset() -> None:
    problems: list[str] = [
        f"{path.name} imports dataforge.datasets.{imported}"
        for path in sorted(DATAFORGE_DIR.glob("*.py"))
        for imported in sorted(_imported_dataset_modules(path) - SHARED_DATASET_MODULES)
    ]
    assert not problems, "a shared dataforge module imports a dataset:\n" + "\n".join(problems)
