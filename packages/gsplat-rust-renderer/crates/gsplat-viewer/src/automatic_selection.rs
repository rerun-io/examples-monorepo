//! Process-local blueprint defaults. Explicit viewer choices always win.
use re_sdk_types::blueprint::archetypes::ActiveVisualizers;
use re_viewer_context::{
    BlueprintContext as _, IdentifiedViewSystem, MissingChunkReporter, ViewContext,
    ViewContextSystem, ViewContextSystemOncePerFrameResult, ViewQuery, ViewSystemIdentifier,
};

#[derive(Default)]
pub struct AutomaticSplatSelection;
impl IdentifiedViewSystem for AutomaticSplatSelection {
    fn identifier() -> ViewSystemIdentifier {
        "AutomaticSplatSelection".into()
    }
}

fn select_default<'a>(explicit: bool, types: &[&'a str]) -> Option<Vec<&'a str>> {
    if explicit || !types.contains(&"GaussianSplats3D") {
        return None;
    }
    let mut selected: Vec<_> = types
        .iter()
        .copied()
        .filter(|name| *name != "GaussianSplats3D" && *name != "ComputeGaussianSplats3D")
        .collect();
    selected.push("ComputeGaussianSplats3D");
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
        let component = ActiveVisualizers::descriptor_instruction_ids().component;
        for (data, _) in query.iter_visualizer_instruction_for("GaussianSplats3D".into()) {
            let explicit = viewer
                .store_context
                .blueprint
                .latest_at(
                    viewer.blueprint_query,
                    &data.override_base_path,
                    [component],
                )
                .get(component)
                .is_some();
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
                    let source = if name == "ComputeGaussianSplats3D" {
                        "GaussianSplats3D"
                    } else {
                        name
                    };
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
        let types = ["Points3D", "GaussianSplats3D", "ComputeGaussianSplats3D"];
        assert_eq!(
            select_default(false, &types),
            Some(vec!["Points3D", "ComputeGaussianSplats3D"])
        );
    }
    #[test]
    fn explicit_native_compute_or_empty_selection_is_preserved() {
        for types in [
            vec!["GaussianSplats3D"],
            vec!["ComputeGaussianSplats3D"],
            vec![],
        ] {
            assert_eq!(select_default(true, &types), None);
        }
        assert_eq!(select_default(false, &["Points3D"]), None);
    }
}
