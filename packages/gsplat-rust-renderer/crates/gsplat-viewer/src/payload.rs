//! The compressed payload a splat entity draws instead of its `GaussianSplats3D` rows (which then serve as the
//! preview: bounds and the native fallback). Resolved once per entity from the same query as the rows; when an
//! entity carries several payloads, SOG wins, then the Streamed SOG, then the ASOG.
use crate::visualizer::{attribute_generation, hash64, last_scalar, last_slice};
use re_view::HybridResultsChunkIter;
use re_viewer::external::{re_sdk_types, re_view, re_viewer_context};
use re_viewer_context::{ViewContext, ViewQuery, VisualizerInstruction};

/// Components per payload archetype, in [`components`] order.
const SOG: usize = 8;
const ASOG: usize = 3;
const LOD: usize = 10;
pub(crate) const COUNT: usize = SOG + ASOG + LOD;

/// The payload archetypes' components: `SogSplats3D` (8), `AnimatedSogSplats3D` (3), then `SogLodSplats3D` (10).
pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; COUNT] {
    let mut out = [crate::sog::components()[0]; COUNT];
    out[..SOG].copy_from_slice(&crate::sog::components());
    out[SOG..SOG + ASOG].copy_from_slice(&crate::asog::components());
    out[SOG + ASOG..].copy_from_slice(&crate::sog_lod::components());
    out
}

pub(crate) enum Payload<'a> {
    /// meta.json is parsed here, once per entity; the planes decode once per payload (row ids), never per time.
    Sog {
        meta: Box<gsplat_core::sog::Meta>,
        webp: gsplat_core::sog::SogWebp<'a>,
        generation: u64,
    },
    /// A Streamed SOG: lod-meta.json parsed once per payload generation, planes split per file every frame (slices).
    SogLod {
        info: std::sync::Arc<crate::sog_lod::Info>,
        metas: Vec<re_sdk_types::ArrowString>,
        webp: Vec<gsplat_core::sog::SogWebp<'a>>,
        generation: u64,
    },
    /// Decoded once per payload; a time change is one motion dispatch in the view (time is not in the key).
    Asog {
        bytes: &'a [u8],
        active_threshold: f32,
        info: crate::asog::Info,
        /// Clip time at the view's time cursor, in seconds.
        time: f32,
        generation: u64,
    },
}

/// The entity's payload from `iters` (one per [`components`] entry), or `Err` with a warning when a payload is
/// present but unusable.
pub(crate) fn resolve<'a>(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
    instruction: &VisualizerInstruction,
    query_results: &re_view::BlueprintResolvedResults<'_>,
    iters: &'a [HybridResultsChunkIter<'a>; COUNT],
) -> Option<Result<Payload<'a>, String>> {
    let [
        m,
        ml,
        mu,
        q,
        s,
        sh0,
        c,
        l,
        a_bytes,
        a_origin,
        a_threshold,
        lod @ ..,
    ] = iters;
    if let Some((meta, webp)) = sog_webp([m, ml, mu, q, s, sh0, c, l]) {
        let generation = attribute_generation(
            instruction,
            query_results,
            crate::sog::components().into_iter(),
        );
        return Some(
            gsplat_core::sog::Meta::parse(&meta)
                .map(|meta| Payload::Sog {
                    meta: Box::new(meta),
                    webp,
                    generation,
                })
                .map_err(|error| error.to_string()),
        );
    }
    if let Some(resolved) = sog_lod(ctx, instruction, query_results, lod) {
        return Some(resolved);
    }
    let bytes = last_slice::<u8>(a_bytes)?;
    let active_threshold =
        last_scalar::<f32>(a_threshold, gsplat_core::trigs::ASOG_ACTIVE_THRESHOLD);
    let generation = hash64((
        attribute_generation(
            instruction,
            query_results,
            crate::asog::components().into_iter(),
        ),
        active_threshold.to_bits(),
    ))
    .hash64();
    let info = ctx
        .viewer_ctx
        .store_context
        .memoizer::<crate::asog::AsogCache, _>(|cache| cache.info(generation, bytes));
    Some(
        info.map(|info| Payload::Asog {
            bytes,
            active_threshold,
            info,
            time: crate::asog::clip_seconds(
                ctx.recording().timeline_type(&query.timeline),
                query.latest_at,
                info.fps,
                last_scalar::<f64>(a_origin, 0.0),
            ),
            generation,
        })
        .map_err(|error| format!("AnimatedSogSplats3D: {error}")),
    )
}

/// meta.json and the stored WebP planes of `SogSplats3D`, when meta and the five required planes are present.
fn sog_webp<'a>(
    [
        meta,
        means_l,
        means_u,
        quats,
        scales,
        sh0,
        centroids,
        labels,
    ]: [&'a HybridResultsChunkIter<'a>; 8],
) -> Option<(String, gsplat_core::sog::SogWebp<'a>)> {
    let meta = meta.slice::<String>().last()?.1.first()?.to_string();
    let webp = gsplat_core::sog::SogWebp {
        means_l: last_slice::<u8>(means_l)?,
        means_u: last_slice::<u8>(means_u)?,
        quats: last_slice::<u8>(quats)?,
        scales: last_slice::<u8>(scales)?,
        sh0: last_slice::<u8>(sh0)?,
        sh_centroids: last_slice::<u8>(centroids),
        sh_labels: last_slice::<u8>(labels),
    };
    Some((meta, webp))
}

/// The `SogLodSplats3D` payload when its lod-meta.json is present.
fn sog_lod<'a>(
    ctx: &ViewContext<'_>,
    instruction: &VisualizerInstruction,
    query_results: &re_view::BlueprintResolvedResults<'_>,
    [lod_meta, metas, sizes, planes @ ..]: &'a [HybridResultsChunkIter<'a>; LOD],
) -> Option<Result<Payload<'a>, String>> {
    let lod_meta = lod_meta.slice::<String>().last()?.1.into_iter().next()?;
    let metas = metas
        .slice::<String>()
        .last()
        .map(|(_, values)| values)
        .unwrap_or_default();
    let generation = attribute_generation(
        instruction,
        query_results,
        crate::sog_lod::components().into_iter(),
    );
    let texts: Vec<&str> = metas.iter().map(|m| m.as_str()).collect();
    let info = ctx
        .viewer_ctx
        .store_context
        .memoizer::<crate::sog_lod::LodInfoCache, _>(|cache| {
            cache.info(generation, lod_meta.as_str(), &texts)
        });
    let planes: [Option<&'a [u8]>; 7] = std::array::from_fn(|p| last_slice::<u8>(&planes[p]));
    Some(info.and_then(|info| {
        let webp = crate::sog_lod::split_planes(
            info.meta.filenames.len(),
            last_slice::<u32>(sizes).unwrap_or_default(),
            planes,
        )?;
        Ok(Payload::SogLod {
            info,
            metas,
            webp,
            generation,
        })
    }))
}
