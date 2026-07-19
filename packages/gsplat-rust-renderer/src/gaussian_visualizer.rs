//! Custom Gaussian splat visualizer — the Rerun-facing middle layer.
//!
//! # Role in the Pipeline
//!
//! This module sits between Rerun's data store and the GPU renderer
//! ([`crate::gaussian_renderer`]).  Each frame it:
//!
//! 1. **Queries** the data store for any entity that matches the upstream
//!    `Gaussians3D` archetype (centers, and optionally scales, quaternions,
//!    colors, spherical-harmonic coefficients, and a show-SH toggle).
//!
//! 2. **Builds or reuses** a [`RenderGaussianCloud`] — a packed, renderer-ready
//!    representation of the Gaussian data.  Clouds are cached per entity path
//!    in the store's `Memoizers` ([`CloudCache`] — visualizer instances are
//!    recreated every frame and cannot hold state) and only rebuilt when the
//!    data or transform signature changes.
//!
//! 3. **Submits** the full cloud + camera to [`GaussianDrawData`] which
//!    drives the GPU render pass.  Culling and depth sorting happen entirely
//!    on the GPU (Brush model) — there is no CPU pre-pass.
//!
//! # Wire contract (upstream `Gaussians3D`)
//!
//! The component descriptors below match the in-development upstream
//! `rerun.archetypes.Gaussians3D` archetype byte-for-byte, so the same Python
//! logger (`gsplat_rust_renderer.gaussians3d.Gaussians3D`) feeds both this
//! renderer and a future native viewer.  Only `centers` is required:
//!
//! | Component | Rerun type | Arrow layout | Absent default |
//! |-----------|-----------|--------------|----------------|
//! | `Gaussians3D:centers` | `Position3D` | `FixedSizeList<f32, 3>` | (required) |
//! | `Gaussians3D:scales` | `Scale3D` | `FixedSizeList<f32, 3>` | `0.01` per axis |
//! | `Gaussians3D:quaternions` | `RotationQuat` | `FixedSizeList<f32, 4>` xyzw | identity |
//! | `Gaussians3D:colors` | `Color` | `u32` `0xRRGGBBAA` | white, opacity 1 |
//! | `Gaussians3D:sh_coefficients` | `SphericalHarmonics3` | `FixedSizeList<f16, 45>` | none (DC only) |
//! | `Gaussians3D:show_spherical_harmonics` | `ShowSphericalHarmonics` | `bool` (mono) | true |
//!
//! Opacity lives in the color alpha (`alpha / 255`).  The base color RGB is the
//! SH degree-0 (DC) term already folded to unorm via `SH_C0 * f_dc + 0.5`;
//! `sh_coefficients` carries **only** degrees 1–3 as 45 `f16` values in
//! coefficient-major layout `[c1.rgb, c2.rgb, …, c15.rgb]` (value index
//! `3 * coeff + channel`).  See [`build_render_cloud`] for how this maps onto
//! the GPU pipeline's DC-at-coeff-0 layout.
//!
//! # Rerun Extension Points
//!
//! The two traits that make this a Rerun visualizer are:
//!
//! - [`IdentifiedViewSystem`] — provides the string identifier
//!   `"Gaussians3D"` that the blueprint uses to bind an entity to this
//!   visualizer (e.g. `overrides={entity: rrb.Visualizer("Gaussians3D")}`).
//!
//! - [`VisualizerSystem`] — the `execute()` method is called once per frame by
//!   the Rerun viewer with the current view context and query.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash as _, Hasher as _};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use crate::gaussian_renderer::GaussianDrawData;
use glam::{Affine3A, Quat, Vec2, Vec3};
use half::f16;
use re_view::{DataResultQuery as _, VisualizerInstructionQueryResults};
use re_view_spatial::{SpatialViewState, TransformTreeContext};
use re_viewer_context::external::re_chunk_store::{
    ChunkDeletionReason, ChunkDirectLineageReport, ChunkStoreDiff, ChunkStoreEvent,
};
use re_viewer_context::external::re_entity_db::EntityDb;
use re_viewer_context::{
    AppOptions, Cache, IdentifiedViewSystem, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, VisualizerExecutionOutput, VisualizerQueryInfo,
    VisualizerSystem,
};
use rerun::{Archetype as _, Component as _, ComponentType};

// ── Imports from gsplat_core (the Rerun-free algorithm module) ───────────
use crate::gsplat_core::constants::SH_C0;
use crate::gsplat_core::{
    CameraApproximation, RenderGaussianCloud, RenderShCoefficients, approximate_bounds_from_points,
    normalize_quat_or_identity,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Archetype definition
// ═══════════════════════════════════════════════════════════════════════════════
//
// The `Gaussians3D` archetype defines the **component contract** between the
// Python logger and this Rust visualizer.  Both sides must agree on the
// archetype name and component descriptors (archetype + component identifier +
// component type).  The Python side
// (`gsplat_rust_renderer.gaussians3d.Gaussians3D`) implements `rr.AsComponents`
// and produces the exact same descriptors.

/// Fully-qualified upstream archetype name shared with the Python logger.
const ARCHETYPE: &str = "rerun.archetypes.Gaussians3D";

/// Number of `f16` values in the wire `sh_coefficients` block: 15 rest
/// coefficients (SH degrees 1–3) × 3 channels.  The DC term is NOT here.
const SH_REST_VALUES: usize = 45;

/// Coefficients-per-channel the GPU pipeline is fed: 1 DC + 15 rest = degree 3.
/// The wire always carries all 45 degree-1..3 values (zero-padded for models
/// trained at a lower degree), so we always reconstruct a full degree-3 block.
const PIPELINE_COEFFS_PER_CHANNEL: usize = 16;

/// Marker type implementing the Rerun `Archetype` trait.  This tells the
/// viewer which components an entity needs in order to be rendered by our
/// custom visualizer.
struct Gaussians3D;

impl rerun::Archetype for Gaussians3D {
    fn name() -> rerun::ArchetypeName {
        ARCHETYPE.into()
    }

    fn display_name() -> &'static str {
        "Gaussians 3D"
    }

    /// Only `centers` is required; everything else has a sensible default.
    fn required_components() -> std::borrow::Cow<'static, [rerun::ComponentDescriptor]> {
        vec![Self::descriptor_centers()].into()
    }

    /// Scales, rotations, colors, SH coefficients, and the show-SH toggle are
    /// all optional — see the module-level table for absent-value semantics.
    fn optional_components() -> std::borrow::Cow<'static, [rerun::ComponentDescriptor]> {
        vec![
            Self::descriptor_scales(),
            Self::descriptor_quaternions(),
            Self::descriptor_colors(),
            Self::descriptor_sh_coefficients(),
            Self::descriptor_show_spherical_harmonics(),
        ]
        .into()
    }
}

/// Component descriptor builders.  Each descriptor specifies the archetype
/// name, a unique component identifier within that archetype (what we query
/// by), and the underlying Rerun component type that carries the actual data.
impl Gaussians3D {
    /// World-space Gaussian center positions — `FixedSizeList<f32, 3>`.
    fn descriptor_centers() -> rerun::ComponentDescriptor {
        static DESCRIPTOR: LazyLock<rerun::ComponentDescriptor> =
            LazyLock::new(|| rerun::ComponentDescriptor {
                archetype: Some(ARCHETYPE.into()),
                component: "Gaussians3D:centers".into(),
                component_type: Some(rerun::components::Position3D::name()),
            });
        (*DESCRIPTOR).clone()
    }

    /// Per-axis scale factors (already exponentiated) — `FixedSizeList<f32, 3>`.
    fn descriptor_scales() -> rerun::ComponentDescriptor {
        static DESCRIPTOR: LazyLock<rerun::ComponentDescriptor> =
            LazyLock::new(|| rerun::ComponentDescriptor {
                archetype: Some(ARCHETYPE.into()),
                component: "Gaussians3D:scales".into(),
                component_type: Some(rerun::components::Scale3D::name()),
            });
        (*DESCRIPTOR).clone()
    }

    /// Per-splat rotation quaternions in `[x, y, z, w]` order — `FixedSizeList<f32, 4>`.
    fn descriptor_quaternions() -> rerun::ComponentDescriptor {
        static DESCRIPTOR: LazyLock<rerun::ComponentDescriptor> =
            LazyLock::new(|| rerun::ComponentDescriptor {
                archetype: Some(ARCHETYPE.into()),
                component: "Gaussians3D:quaternions".into(),
                component_type: Some(rerun::components::RotationQuat::name()),
            });
        (*DESCRIPTOR).clone()
    }

    /// Per-splat packed `0xRRGGBBAA` color; RGB is the unorm DC color, alpha the
    /// opacity — `u32`.
    fn descriptor_colors() -> rerun::ComponentDescriptor {
        static DESCRIPTOR: LazyLock<rerun::ComponentDescriptor> =
            LazyLock::new(|| rerun::ComponentDescriptor {
                archetype: Some(ARCHETYPE.into()),
                component: "Gaussians3D:colors".into(),
                component_type: Some(rerun::components::Color::name()),
            });
        (*DESCRIPTOR).clone()
    }

    /// Optional degree-1..3 SH coefficients — `FixedSizeList<f16, 45>`,
    /// coefficient-major, DC term excluded.  The `SphericalHarmonics3` /
    /// `ShowSphericalHarmonics` component types are not yet in the released
    /// `rerun` crate, so we name them by string to match the wire.
    fn descriptor_sh_coefficients() -> rerun::ComponentDescriptor {
        static DESCRIPTOR: LazyLock<rerun::ComponentDescriptor> =
            LazyLock::new(|| rerun::ComponentDescriptor {
                archetype: Some(ARCHETYPE.into()),
                component: "Gaussians3D:sh_coefficients".into(),
                component_type: Some(ComponentType::from("rerun.components.SphericalHarmonics3")),
            });
        (*DESCRIPTOR).clone()
    }

    /// Optional single `bool` toggle — when `false`, `sh_coefficients` is
    /// ignored and only the base color is rendered.
    fn descriptor_show_spherical_harmonics() -> rerun::ComponentDescriptor {
        static DESCRIPTOR: LazyLock<rerun::ComponentDescriptor> =
            LazyLock::new(|| rerun::ComponentDescriptor {
                archetype: Some(ARCHETYPE.into()),
                component: "Gaussians3D:show_spherical_harmonics".into(),
                component_type: Some(ComponentType::from(
                    "rerun.components.ShowSphericalHarmonics",
                )),
            });
        (*DESCRIPTOR).clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Visualizer state and cache
// ═══════════════════════════════════════════════════════════════════════════════

/// The visualizer that Rerun instantiates **fresh every frame** and calls
/// `execute()` on.  It is stateless: the cloud cache lives in the store's
/// [`re_viewer_context::Memoizers`] (see [`CloudCache`]) — state on this
/// struct would silently be thrown away after each frame.
#[derive(Default)]
pub struct GaussianSplatVisualizer;

/// Persistent per-store cache of packed render clouds, keyed by
/// `"gaussian_splats::{entity_path}"`.
///
/// Lives in the store's `Memoizers` so it survives across frames (visualizer
/// system instances do not).  Entries are dropped when the store reports
/// chunk changes for their entity (re-log, GC) and on memory pressure; the
/// [`CloudSignature`] check on access catches everything else.
#[derive(Default)]
pub struct CloudCache(HashMap<String, CachedCloud>);

impl Cache for CloudCache {
    fn name(&self) -> &'static str {
        "GaussianCloudCache"
    }

    fn purge_memory(&mut self) {
        // Standard rerun cache contract: drop everything not in use.  After a
        // purge, the next frame rebuilds every visible entity's cloud in one
        // go — a deliberate one-frame hitch in exchange for reclaimed memory.
        self.0.clear();
    }

    fn on_store_events(&mut self, events: &[&ChunkStoreEvent], _entity_db: &EntityDb) {
        for event in events {
            // Only evict when the entity's DATA changed.  Storage reshuffles
            // (compaction, virtual→physical swaps, split cleanup) preserve
            // content — and if they move batch boundaries, the signature's
            // row-id content hash forces a rebuild on next access anyway.
            let entity_path = match &event.diff {
                ChunkStoreDiff::Addition(add) => {
                    if matches!(
                        add.direct_lineage,
                        ChunkDirectLineageReport::CompactedFrom(_)
                    ) {
                        continue;
                    }
                    add.chunk_before_processing.entity_path()
                }
                ChunkStoreDiff::Deletion(del) => {
                    if !matches!(del.reason, ChunkDeletionReason::GarbageCollection) {
                        continue;
                    }
                    del.chunk.entity_path()
                }
                _ => continue,
            };
            self.0.remove(&format!("gaussian_splats::{entity_path}"));
        }
    }
}

impl re_byte_size::MemUsageTreeCapture for CloudCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        let bytes: u64 = self
            .0
            .values()
            .map(|entry| {
                let cloud = &entry.cloud;
                let per_splat = (3 + 4 + 3 + 1 + 3) * std::mem::size_of::<f32>();
                let sh = cloud
                    .sh_coeffs
                    .as_ref()
                    .map_or(0, |sh| sh.coefficients.len() * std::mem::size_of::<f32>());
                (cloud.len() * per_splat + sh) as u64
            })
            .sum();
        re_byte_size::MemUsageTree::Bytes(bytes)
    }
}

/// A cached cloud together with the signature that was used to build it.
/// When the signature changes (different splat data, count, SH shape, or
/// transform), the cloud is rebuilt from the current query results.
struct CachedCloud {
    signature: CloudSignature,
    cloud: Arc<RenderGaussianCloud>,
    /// Monotonically increasing build id, bumped on every rebuild.  The
    /// renderer compares it against the generation its per-entity GPU buffers
    /// were uploaded from and re-uploads the splat data on mismatch.
    generation: u64,
}

/// Source of [`CachedCloud::generation`] values.  Global so a generation is
/// never reused, even across visualizer instances or cache evictions.
static CLOUD_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Debug-logs the "no camera yet, skipping this frame" case exactly once.
/// The condition is normal (it only happens on the first frame or two before
/// the view has computed its eye), so logging it every frame would be noise.
static CAMERA_MISSING_LOG_ONCE: std::sync::Once = std::sync::Once::new();

/// Lightweight fingerprint of a cloud's configuration.  Two signatures are
/// equal if and only if the cloud data can be reused without rebuilding.
#[derive(Clone, PartialEq, Eq)]
struct CloudSignature {
    /// Total number of Gaussian splats.
    expected_splats: usize,
    /// Bit-exact representation of the 3×4 entity transform.  Using raw bits
    /// avoids floating-point comparison issues.
    transform_bits: [u32; 12],
    /// Hash of the store row-ids backing every component batch.  Re-logging
    /// an entity writes new rows, so this changes exactly when the underlying
    /// data does — including same-count, same-transform content changes.
    content_hash: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rerun trait implementations
// ═══════════════════════════════════════════════════════════════════════════════

impl IdentifiedViewSystem for GaussianSplatVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "Gaussians3D".into()
    }
}

impl VisualizerSystem for GaussianSplatVisualizer {
    /// Tell Rerun which archetype this visualizer handles.
    fn visualizer_query_info(&self, _app_options: &AppOptions) -> VisualizerQueryInfo {
        let queried_components = [
            Gaussians3D::descriptor_centers(),
            Gaussians3D::descriptor_scales(),
            Gaussians3D::descriptor_quaternions(),
            Gaussians3D::descriptor_colors(),
            Gaussians3D::descriptor_sh_coefficients(),
            Gaussians3D::descriptor_show_spherical_harmonics(),
        ];
        VisualizerQueryInfo::single_required_component::<rerun::components::Position3D>(
            &Gaussians3D::descriptor_centers(),
            &queried_components,
        )
    }

    /// Called once per frame.  This is the main entry point for the visualizer.
    ///
    /// # Per-frame flow
    ///
    /// For each entity that matches the `Gaussians3D` archetype:
    /// 1. Query `centers` (required) + the optional scales/quats/colors/SH/toggle
    /// 2. Compute a cache signature (splat count + SH presence + transform)
    /// 3. Build or reuse the `RenderGaussianCloud`
    /// 4. Extract the current camera from the 3D view state
    /// 5. Submit to `GaussianDrawData` for GPU rendering (the GPU culls and
    ///    depth-sorts — no CPU pre-pass)
    fn execute(
        &self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        context_systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        let mut output = VisualizerExecutionOutput::default();

        // ── Camera (view-global, resolved once) ───────────────────────
        // Use the eye committed to the spatial view state — the same eye the
        // view's `ViewBuilder` renders every other primitive with.  We never
        // invent a camera: on the very first frame(s), before the view's UI
        // pass has run `eye_state.update`, there is no eye yet.  Skip the
        // whole view for this frame and request a *delayed* repaint — the
        // next frame has a real eye and renders from the correct camera.
        // (This is what removed the old "tiny/misplaced splats that snap into
        // place" artifact, which came from a synthetic bounding-box fallback
        // camera fully decoupled from the real view.)  The delay bounds a
        // *persistently* camera-less view (unpublished viewport, unexpected
        // view state) to ~10 repaints/s instead of a max-FPS busy-loop.
        let Some(camera) = camera_from_view(ctx, query) else {
            CAMERA_MISSING_LOG_ONCE.call_once(|| {
                re_log::debug!(
                    "Gaussian splat view has no 3D camera yet (view eye not initialized); \
                     skipping splat rendering this frame and requesting a delayed repaint."
                );
            });
            ctx.egui_ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
            return Ok(output);
        };

        // The transform tree tells us how each entity's coordinate frame
        // relates to the view's coordinate frame.
        let transforms = context_systems.get::<TransformTreeContext>(&output)?;
        let mut draw_data = GaussianDrawData::new(ctx.render_ctx());

        // Iterate over every entity in the current view that has been assigned
        // to this visualizer (via blueprint override or automatic matching).
        for (data_result, instruction) in query.iter_visualizer_instruction_for(Self::identifier())
        {
            // ── Step 1: Query components from the data store ──────────
            let results =
                data_result.query_archetype_with_history::<Gaussians3D>(ctx, query, instruction);
            let results = VisualizerInstructionQueryResults::new(instruction, &results, &output);

            let centers = results.iter_required(Gaussians3D::descriptor_centers().component);
            if centers.is_empty() {
                continue;
            }

            // Everything except `centers` is optional; absent components fall
            // back to per-splat defaults inside `build_render_cloud`.
            let scales = results.iter_optional(Gaussians3D::descriptor_scales().component);
            let quaternions =
                results.iter_optional(Gaussians3D::descriptor_quaternions().component);
            let colors = results.iter_optional(Gaussians3D::descriptor_colors().component);
            let sh = results.iter_optional(Gaussians3D::descriptor_sh_coefficients().component);
            let show_sh =
                results.iter_optional(Gaussians3D::descriptor_show_spherical_harmonics().component);
            let expected_splats = count_splats_in_results(centers.slice::<[f32; 3]>());

            // `show_spherical_harmonics` is a mono `bool`: absent → true, so SH
            // is used whenever `sh_coefficients` is present unless explicitly
            // toggled off.  Read the latest row's first value.
            let show_sh_value: Option<bool> = show_sh
                .slice::<bool>()
                .last()
                .and_then(|(_index, buffer)| (!buffer.is_empty()).then(|| buffer.value(0)));
            let build_sh = !sh.is_empty() && show_sh_value != Some(false);

            // ── Step 2: Resolve entity transform ──────────────────────
            let transform = transforms
                .target_from_entity_path(data_result.entity_path.hash())
                .and_then(|result| result.as_ref().ok())
                .map(|transform_info| {
                    transform_info
                        .single_transform_required_for_entity(
                            &data_result.entity_path,
                            Gaussians3D::name(),
                        )
                        .as_affine3a()
                })
                .unwrap_or(Affine3A::IDENTITY);

            // ── Step 3: Build or reuse the render cloud ───────────────
            // Content identity: Rerun's own query-result hash covers the
            // resolved row-ids of EVERY component — store results, blueprint
            // overrides, and view defaults — so it changes whenever any
            // component (required or optional) resolves to different data,
            // including timeline scrubs of individually-logged components.
            // Only genuinely external state is folded in on top: the resolved
            // SH toggle and the entity transform.
            let mut content_hasher = DefaultHasher::new();
            results.query_result_hash().hash(&mut content_hasher);
            build_sh.hash(&mut content_hasher);

            let label = format!("gaussian_splats::{}", data_result.entity_path);
            let signature = CloudSignature {
                expected_splats,
                transform_bits: transform.to_cols_array().map(f32::to_bits),
                content_hash: content_hasher.finish(),
            };
            // Build the cloud only when this entity path is new or its
            // signature changed (e.g. different splat count after re-logging);
            // steady-state frames reuse the copy cached in the store's
            // memoizers (state on `self` would not survive the frame).
            let store_ctx = ctx.viewer_ctx.store_context;
            let cached: Option<(Arc<RenderGaussianCloud>, u64)> = store_ctx
                .memoizer::<CloudCache, _>(|cache| {
                    cache
                        .0
                        .get(&label)
                        .filter(|entry| entry.signature == signature)
                        .map(|entry| (entry.cloud.clone(), entry.generation))
                });
            let (cloud, cloud_generation) = if let Some(hit) = cached {
                hit
            } else {
                if crate::gaussian_renderer::fps_probe_enabled() {
                    eprintln!("[fps-probe] REBUILDING {label} ({expected_splats} splats)");
                }
                // Build OUTSIDE the cache lock — packing a million-splat cloud
                // takes ~100 ms and must not block other cache users (memory
                // panel, begin_frame).  If another view raced us to it, keep
                // the winner's entry and drop our build.
                let cloud = Arc::new(build_render_cloud(
                    expected_splats,
                    centers.slice::<[f32; 3]>(),
                    scales.slice::<[f32; 3]>(),
                    quaternions.slice::<[f32; 4]>(),
                    colors.slice::<u32>(),
                    sh.slice::<[f16; SH_REST_VALUES]>(),
                    transform,
                    build_sh,
                ));
                // One fresh entry, built once; if a racing view already
                // cached the same signature, ours is dropped (the unused
                // generation bump is harmless — only equality matters).
                let fresh = CachedCloud {
                    signature,
                    cloud,
                    generation: CLOUD_GENERATION.fetch_add(1, Ordering::Relaxed),
                };
                store_ctx.memoizer::<CloudCache, _>(|cache| {
                    use std::collections::hash_map::Entry;
                    let entry = match cache.0.entry(label.clone()) {
                        Entry::Occupied(mut occupied) => {
                            if occupied.get().signature != fresh.signature {
                                occupied.insert(fresh);
                            }
                            occupied.into_mut()
                        }
                        Entry::Vacant(vacant) => vacant.insert(fresh),
                    };
                    (entry.cloud.clone(), entry.generation)
                })
            };

            // ── Step 4: Submit to the GPU renderer ────────────────────
            // Culling and depth sorting happen on the GPU (Brush model);
            // the full cloud + camera is all the renderer needs.
            draw_data.add_batch(ctx.render_ctx(), &label, &cloud, cloud_generation, &camera);
        }

        output.draw_data = vec![draw_data.into()];
        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cloud construction (Rerun-specific)
// ═══════════════════════════════════════════════════════════════════════════════

/// Default per-axis scale when the wire omits `scales` (upstream default).
const DEFAULT_SCALE: f32 = 0.01;

/// Default packed color when the wire omits `colors`: opaque white
/// (`0xRRGGBBAA` = `0xFFFFFFFF`).
const DEFAULT_COLOR: u32 = 0xFFFF_FFFF;

/// Convert the upstream `Gaussians3D` wire components into a packed
/// [`RenderGaussianCloud`].
///
/// This is the only place that knows about Rerun's query result format.  The
/// renderer only sees the flat arrays in `RenderGaussianCloud`.  The entity
/// `transform` is baked into the positions so the GPU shaders don't need a
/// per-entity transform matrix.
///
/// # Wire → GPU SH mapping
///
/// The GPU pipeline (`gaussian_project.wgsl` + `RenderShCoefficients`) expects
/// coefficient-major SH with the **DC term at coefficient 0** and the shader
/// reconstructing color as `SH_C0 * dc + 0.5 + Σ basis·rest`.  The wire instead
/// folds DC into the unorm `colors` RGB and ships only the 15 rest coefficients
/// (degrees 1–3) as 45 `f16` in coefficient-major order (value index
/// `3*coeff + channel`).  So when `build_sh` is set we emit, per splat, a
/// [`PIPELINE_COEFFS_PER_CHANNEL`]-coefficient block:
///
/// * coeff 0 (indices `base+0..3`) = reconstructed DC `f_dc = (rgb_unorm − 0.5) / SH_C0`
/// * coeffs 1..=15 (indices `base+3..48`) = the 45 wire values copied verbatim
///   (`base + 3 + (3*coeff + channel)`), since both layouts are coefficient-major.
///
/// The wire is always 45 values (degree 3), zero-padded for lower-degree
/// models, so we always feed the pipeline a full degree-3 block — the extra
/// SH ALU for an all-zero band is negligible.
#[allow(clippy::too_many_arguments)]
fn build_render_cloud<'a, Idx, ICenters, IScales, IQuaternions, IColors, ISh>(
    expected_splats: usize,
    centers: ICenters,
    scales: IScales,
    quaternions: IQuaternions,
    colors: IColors,
    sh_rest: ISh,
    transform: Affine3A,
    build_sh: bool,
) -> RenderGaussianCloud
where
    Idx: Ord,
    ICenters: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
    IScales: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
    IQuaternions: IntoIterator<Item = (Idx, &'a [[f32; 4]])>,
    IColors: IntoIterator<Item = (Idx, &'a [u32])>,
    ISh: IntoIterator<Item = (Idx, &'a [[f16; SH_REST_VALUES]])>,
{
    // Pre-size everything: at 325k+ splats, growing `sh_flat` (48 floats per
    // splat) through doubling means ~20 multi-MB realloc+memcpy rounds.
    let mut means_world = Vec::with_capacity(expected_splats);
    let mut quats_world = Vec::with_capacity(expected_splats);
    let mut scales_world = Vec::with_capacity(expected_splats);
    let mut opacities_world = Vec::with_capacity(expected_splats);
    let mut colors_world = Vec::with_capacity(expected_splats);
    // Coefficient-major flat SH, `PIPELINE_COEFFS_PER_CHANNEL * 3` per splat.
    let mut sh_flat: Vec<f32> = Vec::with_capacity(if build_sh {
        expected_splats * PIPELINE_COEFFS_PER_CHANNEL * 3
    } else {
        0
    });

    // `range_zip_1x4` iterates `centers` (required) alongside the four optional
    // arrays in lockstep, yielding one row at a time; absent components arrive
    // as `None`.
    for (_index, centers, scales, quats, colors, sh_rest) in
        re_query::range_zip_1x4(centers, scales, quaternions, colors, sh_rest)
    {
        for (row_index, center) in centers.iter().enumerate() {
            // Only positions are transformed here — quaternions and scales stay
            // in entity-local space.  This is intentional and matches Brush:
            // the GPU projection shader builds the 3D covariance from the
            // untransformed quat + scale, then applies the view matrix during
            // the Jacobian-based 2D projection.  Applying the entity transform
            // to the covariance would require decomposing rotation and non-uniform
            // scale from the affine, which is fragile for arbitrary transforms.
            means_world.push(transform.transform_point3(Vec3::from_array(*center)));

            // Quaternion: identity when absent (upstream default).
            let quat = quats
                .and_then(|quats| quats.get(row_index).or_else(|| quats.last()))
                .map(|quat| Quat::from_xyzw(quat[0], quat[1], quat[2], quat[3]))
                .unwrap_or(Quat::IDENTITY);
            quats_world.push(normalize_quat_or_identity(quat));

            // Scale: 0.01 per axis when absent.  Clamp to a small positive
            // minimum to avoid degenerate (zero-volume) Gaussians — 1e-6
            // matches the Python side, the GPU shader, and Brush.
            let scale = scales
                .and_then(|scales| scales.get(row_index).or_else(|| scales.last()))
                .map(|scale| Vec3::from_array(*scale))
                .unwrap_or(Vec3::splat(DEFAULT_SCALE));
            scales_world.push(scale.max(Vec3::splat(1e-6)));

            // Color: opaque white when absent.  Alpha carries opacity; RGB is
            // the unorm base (DC-folded) color the shader uses when SH is off.
            let color = colors
                .and_then(|colors| colors.get(row_index).or_else(|| colors.last()))
                .copied()
                .unwrap_or(DEFAULT_COLOR);
            let [r, g, b, a] = re_sdk_types::datatypes::Rgba32::from_u32(color).to_array();
            let rgb = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
            opacities_world.push(a as f32 / 255.0);
            colors_world.push(rgb);

            if build_sh {
                // Coeff 0 = DC reconstructed from the unorm base color.
                sh_flat.push((rgb[0] - 0.5) / SH_C0);
                sh_flat.push((rgb[1] - 0.5) / SH_C0);
                sh_flat.push((rgb[2] - 0.5) / SH_C0);
                // Coeffs 1..=15 = the 45 rest values (zeros if this splat has
                // none, keeping every splat's block a fixed 48 floats so the
                // GPU `splat_index * coeffs_per_channel` indexing stays valid).
                match sh_rest.and_then(|sh_rest| sh_rest.get(row_index).or_else(|| sh_rest.last()))
                {
                    Some(rest) => sh_flat.extend(rest.iter().map(|value| value.to_f32())),
                    None => sh_flat.extend(std::iter::repeat_n(0.0_f32, SH_REST_VALUES)),
                }
            }
        }
    }

    let bounds_world = approximate_bounds_from_points(&means_world);
    let sh_coeffs = build_sh.then(|| RenderShCoefficients {
        coeffs_per_channel: PIPELINE_COEFFS_PER_CHANNEL,
        coefficients: Arc::from(sh_flat),
    });

    RenderGaussianCloud {
        means_world: Arc::from(means_world),
        quats: Arc::from(quats_world),
        scales: Arc::from(scales_world),
        opacities: Arc::from(opacities_world),
        colors_dc: Arc::from(colors_world),
        sh_coeffs,
        bounds_world,
    }
}

/// Count the total number of splats across all component batches.
fn count_splats_in_results<'a, I, Idx>(centers: I) -> usize
where
    I: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
{
    centers
        .into_iter()
        .map(|(_index, positions)| positions.len())
        .sum()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Camera extraction (Rerun-specific)
// ═══════════════════════════════════════════════════════════════════════════════

/// Try to extract camera parameters from the Rerun 3D view's interactive
/// orbit camera.  Returns `None` if the view hasn't been set up yet.
fn camera_from_view(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<CameraApproximation> {
    camera_from_spatial_view_state(ctx, query)
}

/// Read the eye state from the Spatial3DView and convert it into our
/// simplified camera representation.
fn camera_from_spatial_view_state(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
) -> Option<CameraApproximation> {
    // Downcast the generic view state to the 3D-specific one.
    let spatial_view_state = ctx.view_state.as_any().downcast_ref::<SpatialViewState>()?;
    let eye = spatial_view_state.state_3d.eye_state.last_eye?;
    let vertical_fov = eye.fov_y?;
    let viewport_size_px = published_viewport_size_px(ctx, query)?;
    let aspect_ratio = (viewport_size_px.x / viewport_size_px.y.max(1.0)).max(1e-4);
    let near_plane = eye.near();

    Some(CameraApproximation {
        // Rerun stores `world_from_rub_view` (world ← view); we need the
        // inverse (world → view).
        view_from_world: Affine3A::from_mat4(eye.world_from_rub_view.inverse().to_mat4()),
        projection_from_view: glam::Mat4::perspective_infinite_rh(
            vertical_fov,
            aspect_ratio,
            near_plane,
        ),
        world_position: eye.pos_in_world(),
        viewport_size_px,
        near_plane,
    })
}

/// Read the viewport rectangle from the egui cache and convert to physical pixels.
fn published_viewport_size_px(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<Vec2> {
    let view_info = ctx.egui_ctx().memory_mut(|memory| {
        memory
            .caches
            .cache::<re_viewer_context::ViewRectPublisher>()
            .get(&query.view_id)
            .cloned()
    })?;
    // Shrink slightly to avoid edge artifacts.
    let rect = view_info.rect.shrink(2.5);
    if !rect.is_positive() {
        return None;
    }
    // Convert from logical UI points to physical pixels.
    let viewport_size_px = rect.size() * ctx.egui_ctx().pixels_per_point();
    Some(Vec2::new(
        viewport_size_px.x.max(1.0),
        viewport_size_px.y.max(1.0),
    ))
}
