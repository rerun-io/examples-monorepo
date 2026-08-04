import os
import subprocess
import sys
from pathlib import Path

from simplecv.apis.convert_to_rrd import ConvertEgoConfig
from simplecv.apis.download_aria_gen2_pilot import DownloadConfig as AriaDownloadConfig
from simplecv.apis.download_hot3d import DownloadConfig as Hot3dDownloadConfig
from simplecv.apis.download_mamma import DownloadConfig as MammaDownloadConfig
from simplecv.apis.preprocess_aria_gen2_pilot import PreprocessConfig as AriaPreprocessConfig
from simplecv.apis.preprocess_hot3d import PreprocessConfig as Hot3dPreprocessConfig
from simplecv.apis.preprocess_mamma import PreprocessConfig as MammaPreprocessConfig
from simplecv.configs.dataset_paths import (
    ARIA_GEN2_PILOT_ROOT,
    EGO_DEX_ROOT,
    HOT3D_ROOT,
    MAMMA_AV1_1080_ROOT,
    MAMMA_SOURCE_ROOT,
    ROBOCAP_ROOT,
    UMETRACK_SPLIT_ROOT,
)
from simplecv.configs.exoego_dataset_configs import dataset_defaults


def test_dataset_root_environment_override_reaches_dataset_config(tmp_path: Path) -> None:
    package_root: Path = Path(__file__).resolve().parents[1]
    dataset_root: Path = tmp_path / "datasets"
    env: dict[str, str] = os.environ.copy()
    env["SIMPLECV_DATA_ROOT"] = str(dataset_root)
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            "from simplecv.configs.exoego_dataset_configs import dataset_defaults; "
            "print(dataset_defaults['umetrack'].root_directory)",
        ],
        cwd=package_root,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(dataset_root / "umetrack-split")


def test_available_datasets_default_to_nas_paths() -> None:
    expected_paths: dict[str, tuple[str, Path]] = {
        "aria-gen2": ("base_directory", ARIA_GEN2_PILOT_ROOT),
        "ego-dex": ("root_directory", EGO_DEX_ROOT),
        "hot3d": ("base_directory", HOT3D_ROOT),
        "mamma": ("root_directory", MAMMA_AV1_1080_ROOT),
        "robocap": ("root_directory", ROBOCAP_ROOT),
        "umetrack": ("root_directory", UMETRACK_SPLIT_ROOT),
    }

    for dataset_name, (attribute_name, expected_path) in expected_paths.items():
        assert getattr(dataset_defaults[dataset_name], attribute_name) == expected_path


def test_related_workflows_default_to_nas_paths() -> None:
    aria_download_config = AriaDownloadConfig()
    assert aria_download_config.urls_json == ARIA_GEN2_PILOT_ROOT / "AriaGen2PilotDataset_download_urls.json"
    assert aria_download_config.output_dir == ARIA_GEN2_PILOT_ROOT
    assert AriaPreprocessConfig().root == ARIA_GEN2_PILOT_ROOT

    hot3d_download_config = Hot3dDownloadConfig(urls_json=Path("/tmp/Hot3DAria_download_urls.json"))
    assert hot3d_download_config.output_dir == HOT3D_ROOT / "aria"
    assert Hot3dPreprocessConfig().root == HOT3D_ROOT / "aria"

    assert MammaDownloadConfig().output_dir == MAMMA_SOURCE_ROOT
    assert MammaPreprocessConfig().root_directory == MAMMA_SOURCE_ROOT

    ego_dex_config = ConvertEgoConfig()
    assert ego_dex_config.root_directory == EGO_DEX_ROOT / "test"
    assert ego_dex_config.save_directory == EGO_DEX_ROOT / "test-rrd"
