import tyro

from gsplat_rust_renderer.apis.visualize_brush_training import VisualizeBrushTrainingConfig, main

if __name__ == "__main__":
    main(tyro.cli(VisualizeBrushTrainingConfig))
