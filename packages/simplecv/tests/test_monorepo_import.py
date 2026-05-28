from pathlib import Path

import simplecv


def test_simplecv_imports_from_monorepo_package() -> None:
    package_file: Path = Path(simplecv.__file__).resolve()
    package_root: Path = Path(__file__).resolve().parents[1]

    assert package_file.is_relative_to(package_root / "simplecv")
