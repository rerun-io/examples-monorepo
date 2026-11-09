//! `DepthVideoStream`: depth carried in the luma plane of a video stream.
//!
//! Out-of-tree scaffolding: a custom archetype made of built-in component types, deleted once `DepthVideoStream`
//! ships in Rerun. Its descriptors are those of the in-tree prototype (`rerun.archetypes.DepthVideoStream`), so a
//! recording works in both viewers. The Python helper in `depth_video_viewer/log.py` writes them with the stock SDK.
//!
//! All fields except `sample` are typically logged statically once per entity; `sample` is logged once per frame,
//! as for `VideoStream`:
//! - `codec` (required): only AV1 decodes at full bit depth here;
//! - `sample`: one AV1 temporal unit per frame, no B-frames;
//! - `meter`: decoded units per meter, `1000 / u` for codes in units of `u` millimetres (default 1000);
//! - `colormap` (default: Rerun's depth colormap);
//! - `depth_range`: visible range in decoded units (default `[0, 2^bit_depth - 1]`);
//! - `point_fill_ratio`: 3D point radius as a fraction of the projected pixel size.

use re_viewer::external::re_sdk_types::{
    ArchetypeName, Component as _, ComponentDescriptor, ComponentIdentifier, ComponentType,
    components,
};

/// The archetype name the descriptors carry.
pub const ARCHETYPE: &str = "rerun.archetypes.DepthVideoStream";

#[derive(Clone, Copy)]
pub enum Field {
    Codec,
    Sample,
    Meter,
    Colormap,
    DepthRange,
    PointFillRatio,
}

impl Field {
    const ALL: [Self; 6] = [
        Self::Codec,
        Self::Sample,
        Self::Meter,
        Self::Colormap,
        Self::DepthRange,
        Self::PointFillRatio,
    ];

    /// Field name and component type.
    fn name_and_type(self) -> (&'static str, ComponentType) {
        match self {
            Self::Codec => ("codec", components::VideoCodec::name()),
            Self::Sample => ("sample", components::VideoSample::name()),
            Self::Meter => ("meter", components::DepthMeter::name()),
            Self::Colormap => ("colormap", components::Colormap::name()),
            Self::DepthRange => ("depth_range", components::ValueRange::name()),
            Self::PointFillRatio => ("point_fill_ratio", components::FillRatio::name()),
        }
    }

    pub fn descriptor(self) -> ComponentDescriptor {
        let (field, component_type) = self.name_and_type();
        let archetype = ArchetypeName::from(ARCHETYPE);
        ComponentDescriptor {
            archetype: Some(archetype),
            component: ComponentIdentifier::from_archetype_field(archetype, field),
            component_type: Some(component_type),
        }
    }

    pub fn component(self) -> ComponentIdentifier {
        self.descriptor().component
    }
}

pub fn all_components() -> Vec<ComponentDescriptor> {
    Field::ALL.map(Field::descriptor).to_vec()
}

/// Viewer default for `meter`: one unit is one millimetre.
pub const DEFAULT_METER: f32 = 1000.0;

/// Viewer default for `point_fill_ratio`: points as wide as a pixel.
pub const DEFAULT_POINT_FILL_RATIO: f32 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// The Python helper and the prototype write `DepthVideoStream:<field>` under the full archetype name.
    #[test]
    fn descriptors_match_the_python_helper() {
        let codec = Field::Codec.descriptor();
        assert_eq!(codec.component.as_str(), "DepthVideoStream:codec");
        assert_eq!(codec.archetype.unwrap().as_str(), ARCHETYPE);
        assert_eq!(codec.component_type, Some(components::VideoCodec::name()));
        assert_eq!(
            Field::PointFillRatio.descriptor().component_type,
            Some(components::FillRatio::name())
        );
        assert_eq!(
            Field::PointFillRatio.component().as_str(),
            "DepthVideoStream:point_fill_ratio"
        );
    }
}
