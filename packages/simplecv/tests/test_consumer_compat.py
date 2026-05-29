from __future__ import annotations


def test_legacy_ego_dataset_configs_import_path_reexports_exoego_union() -> None:
    from simplecv.configs.ego_dataset_configs import AnnotatedEgoDatasetUnion, dataset_defaults
    from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion

    assert AnnotatedEgoDatasetUnion is AnnotatedExoEgoDatasetUnion
    assert "hocap" in dataset_defaults


def test_legacy_rerun_log_utils_exports_confidence_scores_to_rgb() -> None:
    from simplecv.rerun_custom_types import confidence_scores_to_rgb as canonical_confidence_scores_to_rgb
    from simplecv.rerun_log_utils import confidence_scores_to_rgb

    assert confidence_scores_to_rgb is canonical_confidence_scores_to_rgb
