#!/bin/sh
# Keep one stable Cargo source path across the prod and dev environments.
gsplat_brush_link="$PIXI_PROJECT_ROOT/packages/gsplat-rust-renderer/target/brush-src"
if [ ! -e "$gsplat_brush_link" ]; then
    mkdir -p "$PIXI_PROJECT_ROOT/packages/gsplat-rust-renderer/target"
    ln -sfn "$CONDA_PREFIX/share/brush-src" "$gsplat_brush_link"
fi
unset gsplat_brush_link
