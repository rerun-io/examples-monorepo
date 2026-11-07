//! `DepthVideoStream`: depth carried in the luma plane of a video stream.
//!
//! A custom archetype made of built-in component types. Its descriptors are those of the in-tree
//! prototype (`rerun.archetypes.DepthVideoStream`), so a recording works in both viewers.
//! The Python helper in `depth_video_viewer/log.py` writes them with the stock SDK.
//!
//! All components except `sample` are typically logged statically once per entity;
//! `sample` is logged once per frame, as for `VideoStream`.

use re_viewer::external::re_sdk_types::{
    ArchetypeName, Component as _, ComponentDescriptor, ComponentIdentifier, ComponentType,
    components,
};

/// The archetype name the descriptors carry.
pub const ARCHETYPE: &str = "rerun.archetypes.DepthVideoStream";

fn descriptor(field: &str, component_type: ComponentType) -> ComponentDescriptor {
    let archetype = ArchetypeName::from(ARCHETYPE);
    ComponentDescriptor {
        archetype: Some(archetype),
        component: ComponentIdentifier::from_archetype_field(archetype, field),
        component_type: Some(component_type),
    }
}

/// Required: the codec of the samples. Only AV1 decodes at full bit depth here.
pub fn descriptor_codec() -> ComponentDescriptor {
    descriptor("codec", components::VideoCodec::name())
}

/// One encoded depth frame (an AV1 temporal unit) per row. B-frames are not supported.
pub fn descriptor_sample() -> ComponentDescriptor {
    descriptor("sample", components::VideoSample::name())
}

/// Decoded luma units per meter. For codes in units of `u` millimetres this is `1000 / u`.
pub fn descriptor_meter() -> ComponentDescriptor {
    descriptor("meter", components::DepthMeter::name())
}

/// Colormap for the decoded depth.
pub fn descriptor_colormap() -> ComponentDescriptor {
    descriptor("colormap", components::Colormap::name())
}

/// Visible depth range in decoded units; `[0, 2^bit_depth - 1]` if omitted.
pub fn descriptor_depth_range() -> ComponentDescriptor {
    descriptor("depth_range", components::ValueRange::name())
}

/// Point radius of the 3D depth cloud as a fraction of the projected pixel size.
pub fn descriptor_point_fill_ratio() -> ComponentDescriptor {
    descriptor("point_fill_ratio", components::FillRatio::name())
}

pub fn all_components() -> Vec<ComponentDescriptor> {
    vec![
        descriptor_codec(),
        descriptor_sample(),
        descriptor_meter(),
        descriptor_colormap(),
        descriptor_depth_range(),
        descriptor_point_fill_ratio(),
    ]
}

/// Viewer default for `meter`: one unit is one millimetre.
pub const DEFAULT_METER: f32 = 1000.0;

/// Viewer default for `colormap`, as for depth images.
pub const DEFAULT_COLORMAP: components::Colormap = components::Colormap::Turbo;

#[cfg(test)]
mod tests {
    use super::*;

    /// The Python helper and the prototype write `DepthVideoStream:<field>` under the full archetype name.
    #[test]
    fn descriptors_match_the_python_helper() {
        let codec = descriptor_codec();
        assert_eq!(codec.component.as_str(), "DepthVideoStream:codec");
        assert_eq!(codec.archetype.unwrap().as_str(), ARCHETYPE);
        assert_eq!(
            codec.component_type.unwrap().as_str(),
            "rerun.components.VideoCodec"
        );
        assert_eq!(
            descriptor_point_fill_ratio().component.as_str(),
            "DepthVideoStream:point_fill_ratio"
        );
    }
}
