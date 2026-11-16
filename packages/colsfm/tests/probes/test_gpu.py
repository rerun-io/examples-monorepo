"""Probe (h): what CUDA this pycolmap build exposes."""

from __future__ import annotations

import pycolmap
import pycolmap.pyceres as colmap_ceres


def test_build_reports_cuda() -> None:
    """The conda-forge `cuda_130*` build reports CUDA and sees the device.

    The `cpu_*` build of the same version reports `has_cuda == False` and is what
    the solver picks when the environment does not pin the build string, so this
    probe doubles as a check that the pixi environment kept the CUDA build.
    """
    print(f"[probe] {pycolmap.COLMAP_version} / {pycolmap.COLMAP_build}")
    print(f"[probe] has_cuda={pycolmap.has_cuda} devices={pycolmap.get_num_cuda_devices()}")
    assert pycolmap.has_cuda is True
    assert "with CUDA" in pycolmap.COLMAP_build
    assert pycolmap.get_num_cuda_devices() >= 1


def test_gpu_switch_moved_to_the_outer_options() -> None:
    """`use_gpu` lives on `FeatureExtractionOptions`, not on `SiftExtractionOptions`.

    COLMAP 4.2 split the per-extractor tuning (`sift`, `aliked`, `loma`) from the
    shared runtime settings (`type`, `use_gpu`, `gpu_index`, `num_threads`,
    `max_image_size`).  Code written against COLMAP 3.x's
    `SiftExtractionOptions.use_gpu` raises `AttributeError` here.

    The colsfm pipeline uses TensorRT ALIKED/LightGlue rather than COLMAP's own
    extractors, so this is recorded as available, not as used.
    """
    assert not hasattr(pycolmap.SiftExtractionOptions(), "use_gpu")
    assert not hasattr(pycolmap.SiftMatchingOptions(), "use_gpu")

    extraction: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    assert extraction.use_gpu is True
    assert extraction.gpu_index == "-1"
    assert extraction.type == pycolmap.FeatureExtractorType.SIFT
    assert hasattr(extraction, "sift") and hasattr(extraction, "aliked")

    matching: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    assert matching.use_gpu is True
    assert matching.gpu_index == "-1"
    assert hasattr(matching, "sift") and hasattr(matching, "aliked")
    # New in 4.2: matching is rig-aware.
    assert hasattr(matching, "rig_verification")
    assert hasattr(matching, "skip_image_pairs_in_same_frame")

    device_names: set[str] = {
        name for name in dir(pycolmap.Device) if not name.startswith("_")
    } - {"name", "value"}
    assert {"auto", "cpu", "cuda"} <= device_names


def test_bundle_adjustment_gpu_paths() -> None:
    """Two GPU paths exist for BA: Ceres CUDA and the CASPAR backend.

    conda-forge's `ceres-solver` is a `gpu*` build here, so
    `CeresBundleAdjustmentOptions.use_gpu` has a CUDA linear algebra library to
    reach; `CasparBundleAdjustmentOptions` is COLMAP 4.2's own GPU solver and
    carries its own `gpu_index`.
    """
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    assert options.backend == pycolmap.BundleAdjustmentBackend.CERES
    assert options.ceres.use_gpu is False
    assert options.ceres.min_num_images_gpu_solver == 50

    caspar: pycolmap.CasparBundleAdjustmentOptions = options.caspar
    assert caspar.gpu_index == "-1"
    assert caspar.solver_iter_max == 200

    # Ceres' CUDA-backed linear solvers are named in the enum whether or not the
    # runtime picks them.
    solver_names: set[str] = {
        name for name in dir(colmap_ceres.DenseLinearAlgebraLibraryType) if name.isupper()
    }
    print(f"[probe] ceres dense linear algebra libraries: {sorted(solver_names)}")
    assert "CUDA" in solver_names


def test_feature_matcher_types_include_onnx_paths() -> None:
    """COLMAP 4.2 already has ALIKED and LightGlue ONNX matchers built in.

    That is worth knowing before reimplementing them: `FeatureExtractorType`
    carries ALIKED variants and `FeatureMatcherType` a LightGlue ONNX matcher.
    They are listed here as discovered capability, not as a decision.
    """
    extractor_names: set[str] = {
        name for name in dir(pycolmap.FeatureExtractorType) if name.isupper()
    }
    matcher_names: set[str] = {name for name in dir(pycolmap.FeatureMatcherType) if name.isupper()}
    print(f"[probe] extractors={sorted(extractor_names)} matchers={sorted(matcher_names)}")
    assert {"SIFT", "ALIKED_N32", "ALIKED_N16ROT"} <= extractor_names
    assert {"ALIKED_LIGHTGLUE", "ALIKED_BRUTEFORCE", "SIFT_LIGHTGLUE"} <= matcher_names
    aliked: pycolmap.AlikedExtractionOptions = pycolmap.AlikedExtractionOptions()
    assert aliked.max_num_features == 2048
    assert aliked.min_score == 0.2
    # The default model path is a `url;filename;sha256` download spec pointing at
    # a COLMAP release asset — nothing ships in the conda package, so running
    # COLMAP's own ALIKED needs network access on first use.
    url, filename, digest = aliked.n32_model_path.split(";")
    assert url.startswith("https://github.com/colmap/colmap/releases/")
    assert filename == "aliked-n32.onnx"
    assert len(digest) == 64
    print(f"[probe] built-in ALIKED model spec: {aliked.n32_model_path}")
    lightglue: pycolmap.LightGlueONNXMatchingOptions = pycolmap.LightGlueONNXMatchingOptions()
    # LightGlue has no default at all; the caller must supply the ONNX file.
    assert lightglue.model_path == ""
    assert lightglue.min_score == 0.1
