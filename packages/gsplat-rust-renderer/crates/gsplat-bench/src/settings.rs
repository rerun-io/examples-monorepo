//! Reference-backend support for the shared render controls.
pub use gsplat_render::settings::RenderSettings;

pub fn validate_backend(
    settings: &RenderSettings,
    kind: crate::renderers::Implementation,
    metadata: gsplat_core::RenderMode,
) -> crate::Result<()> {
    settings.validate()?;
    if matches!(kind, crate::renderers::Implementation::Native)
        && (settings.mode(metadata) != gsplat_core::RenderMode::Default
            || settings.splat_scale != 1.0
            || settings.min_scale.is_some())
    {
        return Err(crate::Error::Unsupported(
            "render-mode, scale, or floor controls for this reference renderer".into(),
        ));
    }
    Ok(())
}
