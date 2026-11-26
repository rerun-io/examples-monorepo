"""Runtime SLAM configuration, independent of benchmark definitions."""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from tomllib import TOMLDecodeError

from serde import SerdeError, coerce, field, serde
from serde.toml import from_toml

from slam_rs import _core

SLAM_CONFIG_PATH: Path = Path(__file__).resolve().parents[1] / "slam.toml"
"""Checked-in runtime settings, beside the package."""


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class DatasetProperties:
    """The VIO config shared by every segment of one dataset.

    The catalog supplies sensor geometry, noise and timestamp metadata.
    """

    name: str
    """Catalog dataset name."""
    vio_config: Path
    """Dataset configuration path, relative to slam.toml."""
    video_time_is_absolute: bool = field(default=False, skip_if_default=True)
    """Whether the recordings' ``video_time`` already is the device clock (VSLAM-LAB keeps each benchmark's own stamps).

    False is MSD, whose ``video_time`` is relative to ``property:capture:start_time_ns``;
    :func:`slam_rs.catalog_feed.dataset_rig` turns it into the feed's rig profile.
    """
    frameset_tolerance_ns: int = field(default=0, skip_if_default=True)
    """How far a camera's frame may sit from cam0's and still join its frameset; 0 demands the identical frame list in every camera.

    EuRoC's raw cameras share stamps, but the VSLAM-LAB conversion keeps cam1 by frame count, not by stamp:
    MH_04_difficult loses one stereo pair and V2_03_difficult its last 295, so it matches framesets instead.
    """


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class RobocapConfig:
    """RoboCap device and estimator files; Rust owns the rig facts."""

    device_id: str
    """The device whose sessions the catalog holds: segment ids are ``robocap__<device_id>__<session_id>``."""
    vio_config: str
    """VIO configuration selected for replay, relative to the package root."""
    calibration: str
    """Rig calibration selected for replay, at the core rig profile's downscale, relative to the package root."""


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class SlamConfig:
    """Runtime settings shared by processing and evaluation commands."""

    schema_version: int
    """Runtime configuration schema version."""
    catalog_url: str
    """Default catalog server, overridable by the command."""
    datasets: tuple[DatasetProperties, ...] = field(rename="dataset")
    """Estimator configurations by dataset."""
    robocap: RobocapConfig
    """RoboCap camera selection and estimator files."""
    package_root: Path = field(skip=True, default=SLAM_CONFIG_PATH.parent, compare=False)
    """Directory containing slam.toml; estimator file paths are relative to it."""

    def dataset(self, name: str) -> DatasetProperties:
        """The dataset with this name.

        Raises:
            ValueError: If the configuration has no such dataset.
        """
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise ValueError(f"{name!r} has no runtime settings; have {[d.name for d in self.datasets]}")

    def vio_config_text(self, dataset_name: str, profile: str = "reference") -> str:
        """The VIO config one dataset's segments run with, as its file's own text.

        Args:
            dataset_name: Catalog dataset name.
            profile: Named config overlay; reference preserves the original text.

        Returns:
            The overlaid JSON (original file text for reference), ready for :meth:`slam_rs._core.VioConfig.from_json`.

        Raises:
            ValueError: If the configuration has no such dataset.
        """
        return self._config_text(self.dataset(dataset_name).vio_config, profile)

    def robocap_config_text(self, profile: str = "reference") -> str:
        """The VIO config RoboCap sessions run with, overlaid like :meth:`vio_config_text`."""
        return self._config_text(self.robocap.vio_config, profile)

    def _config_text(self, relative: str | Path, profile: str) -> str:
        path: Path = self.package_root / relative
        return profiled_config_text(path, profile, path.parent / "profiles")


CATALOG_URL_ENV: str = "SLAM_RS_CATALOG_URL"
"""Environment override for ``catalog_url``: the catalog server tools and tests read from.

``slam.toml`` names the default (a server on this host); a machine that reads another server, e.g.
``rerun+http://dgx-spark:9988``, sets this instead of editing the checked-in file.
"""


def load_slam_config(path: Path = SLAM_CONFIG_PATH) -> SlamConfig:
    """Read runtime settings without opening benchmark definitions."""
    try:
        parsed: SlamConfig = from_toml(SlamConfig, path.read_text())
    except (SerdeError, TOMLDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    if parsed.schema_version != 2:
        raise ValueError(f"{path}: expected schema_version 2")
    return replace(parsed, package_root=path.parent, catalog_url=os.environ.get(CATALOG_URL_ENV, parsed.catalog_url))


def config_text_sha256(text: str) -> str:
    """Identify the resolved configuration without changing its serialization.

    Args:
        text: The exact text returned by :func:`profiled_config_text`.

    Returns:
        The hexadecimal SHA-256 of the text encoded as UTF-8.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def overlay_config_text(config_text: str, overlay: str) -> str:
    """Merge a JSON object into a VIO config's ``value0``: a nested object updates the section of that name, any other
    value replaces the key (an experiment's ``--config-overlay``, e.g. ``{"backend.loop_closing": {"frame_time_margin_s": 120.0}}``).

    Text in, text out, as :func:`profiled_config_text`: the Rust config parses the result, which is also what the run's
    config hash covers.

    Args:
        config_text: The config JSON.
        overlay: A JSON object.

    Returns:
        The merged config JSON.

    Raises:
        ValueError: If ``overlay`` is not a JSON object, or Rust rejects the merged config.
    """
    changes: object = json.loads(overlay)
    if not isinstance(changes, dict):
        raise ValueError(f"--config-overlay must be a JSON object, got {overlay!r}")
    document: dict = json.loads(config_text)
    values: dict = document["value0"]
    for key, value in changes.items():
        values[key] = {**values[key], **value} if isinstance(value, dict) and isinstance(values.get(key), dict) else value
    merged: str = json.dumps(document)
    _core.VioConfig.from_json(merged)
    return merged


def profiled_config_text(path: Path, profile: str = "reference", profiles: Path = SLAM_CONFIG_PATH.parent / "configs/profiles") -> str:
    """Read a config and apply a named overlay; empty overlays preserve its text.

    Args:
        path: Base config JSON.
        profile: Overlay file stem.
        profiles: Directory holding the flat config-key overlays.

    Returns:
        Config JSON with the overlay applied.

    Raises:
        ValueError: If Rust rejects the config, an overlay key, or its value.
    """
    text: str = path.read_text()
    overlay: dict = json.loads((profiles / f"{profile}.json").read_text())
    if not overlay:
        return text
    document: dict = json.loads(text)
    values: dict = document["value0"]
    values.update(overlay)
    resolved: str = json.dumps(document)
    _core.VioConfig.from_json(resolved)
    return resolved
