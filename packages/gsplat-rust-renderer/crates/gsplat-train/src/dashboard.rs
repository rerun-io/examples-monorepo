//! Declarative training dashboard shared by live and saved recordings.
use rerun::blueprint::{
    Blueprint, BlueprintActivation, ContainerLike, Grid, Horizontal, Spatial2DView, Spatial3DView,
    Tabs, TimeSeriesView, Vertical,
};

pub fn send(
    rec: &rerun::RecordingStream,
    eval_count: usize,
    compute: bool,
    video: bool,
    up: Option<[f32; 3]>,
) -> anyhow::Result<()> {
    let mut scene = Spatial3DView::new("Scene")
        .with_origin("world")
        .with_contents(["world/**"]);
    if compute {
        scene = scene.with_override(
            "world/splats",
            rerun::blueprint::Visualizer::new("ComputeGaussianSplats3D"),
        );
    }
    let ContainerLike::View(scene) = ContainerLike::from(scene) else {
        unreachable!()
    };
    let eye_path = format!("{}/EyeControls3D", scene.blueprint_path());
    let pairs = (0..eval_count.min(4)).map(|i| {
        Horizontal::new([
            Spatial2DView::new(format!("GT {i}"))
                .with_origin(format!("eval/view_{i}/ground_truth"))
                .into(),
            Spatial2DView::new(format!("Render {i}"))
                .with_origin(format!("eval/view_{i}/render"))
                .into(),
        ])
        .into()
    });
    let top = if eval_count == 0 {
        Horizontal::new([scene.into()])
    } else {
        Horizontal::new([scene.into(), Grid::new(pairs).with_grid_columns(2).into()])
            .with_column_shares([1.0, 2.0])
    };
    let quality: Vec<ContainerLike> = [
        ("Loss", "loss/**"),
        ("PSNR", "psnr/**"),
        ("SSIM", "ssim/**"),
        ("Splats", "splats/**"),
    ]
    .into_iter()
    .map(|(name, path)| TimeSeriesView::new(name).with_contents([path]).into())
    .collect();
    let graphs: ContainerLike = if video {
        Horizontal::new(quality).into()
    } else {
        Horizontal::new([
            Tabs::new(quality).with_name("Quality").into(),
            Tabs::new([
                TimeSeriesView::new("Learning rates")
                    .with_contents(["lr/**"])
                    .into(),
                TimeSeriesView::new("Step time")
                    .with_contents(["train/**"])
                    .into(),
            ])
            .into(),
            TimeSeriesView::new("Refine")
                .with_contents(["refine/**"])
                .into(),
            TimeSeriesView::new("Memory")
                .with_contents(["memory/**"])
                .into(),
        ])
        .into()
    };
    let blueprint = Blueprint::new(Vertical::new([top.into(), graphs]).with_row_shares([3.0, 1.0]))
        .with_auto_layout(false)
        .with_auto_views(false)
        .with_blueprint_panel(
            rerun::blueprint::BlueprintPanel::new()
                .with_state(rerun::blueprint::components::PanelState::Collapsed),
        )
        .with_selection_panel(
            rerun::blueprint::SelectionPanel::new()
                .with_state(rerun::blueprint::components::PanelState::Collapsed),
        )
        .with_time_panel(
            rerun::blueprint::TimePanel::new()
                .with_timeline("iterations")
                .with_state(rerun::blueprint::components::PanelState::Collapsed),
        );
    if !video && up.is_none() {
        blueprint.send(rec, BlueprintActivation::default())?;
        return Ok(());
    }
    // SDK 0.38.1 has no public Spatial3DView property setter. Append the eye
    // property to the same blueprint store through its public memory transport.
    let (buffer, storage) = rerun::RecordingStreamBuilder::new("gsplat-train").memory()?;
    blueprint.send(&buffer, BlueprintActivation::default())?;
    let mut messages = storage.take();
    let activation = messages
        .iter()
        .find_map(|message| match message {
            rerun::log::LogMsg::BlueprintActivationCommand(command) => Some(command.clone()),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("blueprint activation missing"))?;
    // The temporary memory recording also has its own data-store header.
    // Forward only the blueprint, or the viewer selects that empty recording.
    messages.retain(|message| {
        message.store_id() == &activation.blueprint_id
            && !matches!(message, rerun::log::LogMsg::BlueprintActivationCommand(_))
    });
    let (properties, property_storage) = rerun::RecordingStreamBuilder::new("gsplat-train")
        .recording_id(activation.blueprint_id.recording_id().clone())
        .blueprint()
        .memory()?;
    properties.set_time_sequence("blueprint", 0);
    let mut eye = rerun::external::re_sdk_types::blueprint::archetypes::EyeControls3D::new();
    if video {
        eye = eye.with_spin_speed(0.15);
    }
    if let Some(up) = up {
        eye = eye.with_eye_up(up);
    }
    properties.log(eye_path.as_str(), &eye)?;
    messages.extend(property_storage.take());
    rec.send_blueprint(messages, activation);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn visualizer_override_is_opt_in_and_eye_uses_scene_up() {
        for compute in [false, true] {
            let (rec, storage) = rerun::RecordingStreamBuilder::new("dashboard-test")
                .memory()
                .unwrap();
            super::send(&rec, 4, compute, true, Some([0.0, 0.0, 1.0])).unwrap();
            let mut override_seen = false;
            let mut eye_seen = false;
            for message in storage.take() {
                if let rerun::log::LogMsg::ArrowMsg(_, arrow) = message {
                    let chunk = re_chunk::Chunk::from_arrow_msg(&arrow).unwrap();
                    let expected_override = rerun::external::re_sdk_types::blueprint::archetypes::VisualizerInstruction::new("ComputeGaussianSplats3D").visualizer_type.unwrap();
                    if let Some(actual) =
                        chunk.component_batch_raw(expected_override.descriptor.component, 0)
                    {
                        override_seen |= *actual.unwrap() == *expected_override.array;
                    }
                    if chunk.entity_path().to_string().ends_with("/EyeControls3D") {
                        let expected = rerun::external::re_sdk_types::blueprint::archetypes::EyeControls3D::new().with_eye_up([0.0, 0.0, 1.0]).eye_up.unwrap();
                        assert_eq!(
                            *chunk
                                .component_batch_raw(expected.descriptor.component, 0)
                                .unwrap()
                                .unwrap(),
                            *expected.array
                        );
                        eye_seen = true;
                    }
                }
            }
            assert_eq!(override_seen, compute);
            assert!(eye_seen);
        }
    }

    #[test]
    fn dashboard_does_not_add_an_empty_data_recording() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("gsplat-train")
            .memory()
            .unwrap();
        super::send(&rec, 4, true, true, Some([0.0, 0.0, 1.0])).unwrap();
        let messages = storage.take();
        let recordings: std::collections::HashSet<_> = messages
            .iter()
            .filter(|message| message.store_id().kind() == rerun::StoreKind::Recording)
            .map(|message| message.store_id().clone())
            .collect();
        assert_eq!(
            recordings.len(),
            1,
            "only the caller's data recording may be forwarded"
        );
        assert!(
            messages.iter().any(|message| matches!(
                message,
                rerun::log::LogMsg::BlueprintActivationCommand(_)
            ))
        );
    }
}
