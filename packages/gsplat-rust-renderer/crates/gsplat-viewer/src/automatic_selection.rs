//! Process-local blueprint defaults. Explicit viewer choices always win.
use re_sdk_types::blueprint::archetypes::ActiveVisualizers;

pub const COMPUTE: &str = "ComputeGaussianSplats3D";
pub const NATIVE: &str = "GaussianSplats3D";
use re_viewer_context::{
    BlueprintContext as _, IdentifiedViewSystem, MissingChunkReporter, ViewContext,
    ViewContextSystem, ViewContextSystemOncePerFrameResult, ViewQuery, ViewSystemIdentifier,
};

/// Explicit native, compute, or empty instructions all override automatic selection.
pub fn has_explicit_visualizers(
    viewer: &re_viewer_context::ViewerContext<'_>,
    data: &re_viewer_context::DataResult,
) -> bool {
    let component = ActiveVisualizers::descriptor_instruction_ids().component;
    viewer
        .store_context
        .blueprint
        .latest_at(
            viewer.blueprint_query,
            &data.override_base_path,
            [component],
        )
        .get(component)
        .is_some()
}

#[derive(Default)]
pub struct AutomaticSplatSelection;
impl IdentifiedViewSystem for AutomaticSplatSelection {
    fn identifier() -> ViewSystemIdentifier {
        "AutomaticSplatSelection".into()
    }
}

fn select_default<'a>(explicit: bool, types: &[&'a str]) -> Option<Vec<&'a str>> {
    if explicit || !types.contains(&NATIVE) {
        return None;
    }
    let mut selected: Vec<_> = types
        .iter()
        .copied()
        .filter(|name| *name != NATIVE && *name != COMPUTE)
        .collect();
    selected.push(COMPUTE);
    Some(selected)
}
impl ViewContextSystem for AutomaticSplatSelection {
    fn execute(
        &mut self,
        ctx: &ViewContext<'_>,
        _: &MissingChunkReporter,
        query: &ViewQuery<'_>,
        _: &ViewContextSystemOncePerFrameResult,
    ) {
        let viewer = ctx.viewer_ctx;
        for (data, _) in query.iter_visualizer_instruction_for(NATIVE.into()) {
            let explicit = has_explicit_visualizers(viewer, data);
            let types: Vec<_> = data
                .visualizer_instructions
                .iter()
                .map(|i| i.visualizer_type.as_str())
                .collect();
            let Some(selected) = select_default(explicit, &types) else {
                continue;
            };
            let instructions: Vec<_> = selected
                .into_iter()
                .map(|name| {
                    let source = if name == COMPUTE { NATIVE } else { name };
                    let mut instruction = data
                        .visualizer_instructions
                        .iter()
                        .find(|i| i.visualizer_type.as_str() == source)
                        .expect("selected existing instruction")
                        .clone();
                    instruction.visualizer_type = name.into();
                    instruction
                })
                .collect();
            for instruction in &instructions {
                instruction.write_instruction_to_blueprint(viewer);
            }
            viewer.save_blueprint_archetype(
                data.override_base_path.clone(),
                &ActiveVisualizers::new(instructions.iter().map(|i| i.id)),
            );
            ctx.egui_ctx().request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_replace_only_native_and_deduplicate_compute() {
        let types = ["Points3D", NATIVE, COMPUTE];
        assert_eq!(
            select_default(false, &types),
            Some(vec!["Points3D", COMPUTE])
        );
    }
    #[test]
    fn explicit_native_compute_or_empty_selection_is_preserved() {
        for types in [vec![NATIVE], vec![COMPUTE], vec![]] {
            assert_eq!(select_default(true, &types), None);
        }
        assert_eq!(select_default(false, &["Points3D"]), None);
    }
}
