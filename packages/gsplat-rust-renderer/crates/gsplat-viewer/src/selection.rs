//! Process-local blueprint defaults. Explicit viewer choices always win.
use re_sdk_types::blueprint::archetypes::ActiveVisualizers;
use re_viewer::external::{re_sdk_types, re_viewer_context};

pub(crate) const COMPUTE: &str = "ComputeGaussianSplats3D";

pub(crate) const NATIVE: &str = "GaussianSplats3D";
use re_viewer_context::{
    BlueprintContext as _, IdentifiedViewSystem, MissingChunkReporter, ViewContext,
    ViewContextSystem, ViewContextSystemOncePerFrameResult, ViewQuery, ViewSystemIdentifier,
};

/// Explicit native, compute, or empty instructions all override automatic selection.
pub(crate) fn has_explicit_visualizers(
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
pub(crate) struct AutomaticSplatSelection;

impl IdentifiedViewSystem for AutomaticSplatSelection {
    fn identifier() -> ViewSystemIdentifier {
        "AutomaticSplatSelection".into()
    }
}

fn select_default(
    explicit: bool,
    instructions: &[re_viewer_context::VisualizerInstruction],
) -> Option<Vec<re_viewer_context::VisualizerInstruction>> {
    if explicit {
        return None;
    }
    let mut compute = instructions
        .iter()
        .find(|i| i.visualizer_type.as_str() == NATIVE)?
        .clone();
    compute.visualizer_type = COMPUTE.into();
    let mut selected: Vec<_> = instructions
        .iter()
        .filter(|i| ![NATIVE, COMPUTE].contains(&i.visualizer_type.as_str()))
        .cloned()
        .collect();
    selected.push(compute);
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
            let Some(instructions) = select_default(explicit, &data.visualizer_instructions) else {
                continue;
            };
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

    fn instruction(name: &'static str) -> re_viewer_context::VisualizerInstruction {
        re_viewer_context::VisualizerInstruction::new(
            re_sdk_types::blueprint::components::VisualizerInstructionId::new_random(),
            name.into(),
            &re_viewer::external::re_log_types::EntityPath::from("entity"),
            std::iter::once((
                re_sdk_types::archetypes::GaussianSplats3D::descriptor_centers().component,
                re_viewer_context::VisualizerComponentSource::SourceComponent {
                    source_component:
                        re_sdk_types::archetypes::GaussianSplats3D::descriptor_centers().component,
                    selector: name.to_owned(),
                },
            ))
            .collect(),
        )
    }

    #[test]
    fn defaults_replace_only_native_and_deduplicate_compute() {
        let native = instruction(NATIVE);
        let other = instruction("Points3D");
        let instructions = [other.clone(), native.clone(), instruction(COMPUTE)];
        let selected = select_default(false, &instructions).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].id, other.id);
        assert_eq!(selected[0].component_mappings, other.component_mappings);
        assert_eq!(selected[1].override_path, native.override_path);
        assert_eq!(selected[1].id, native.id);
        assert_eq!(selected[1].component_mappings, native.component_mappings);
        assert_eq!(selected[1].visualizer_type.as_str(), COMPUTE);
    }

    #[test]
    fn explicit_native_compute_or_empty_selection_is_preserved() {
        for instructions in [
            vec![instruction(NATIVE)],
            vec![instruction(COMPUTE)],
            vec![],
        ] {
            assert!(select_default(true, &instructions).is_none());
        }
        assert!(select_default(false, &[instruction("Points3D")]).is_none());
    }
}
