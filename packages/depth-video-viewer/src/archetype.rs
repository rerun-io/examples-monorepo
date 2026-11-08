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

use re_viewer::external::re_sdk_types::{ArchetypeName, ComponentDescriptor, ComponentIdentifier};

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

/// Field name and component type, indexed by [`Field`].
const FIELDS: [(Field, &str, &str); 6] = [
    (Field::Codec, "codec", "rerun.components.VideoCodec"),
    (Field::Sample, "sample", "rerun.components.VideoSample"),
    (Field::Meter, "meter", "rerun.components.DepthMeter"),
    (Field::Colormap, "colormap", "rerun.components.Colormap"),
    (
        Field::DepthRange,
        "depth_range",
        "rerun.components.ValueRange",
    ),
    (
        Field::PointFillRatio,
        "point_fill_ratio",
        "rerun.components.FillRatio",
    ),
];

impl Field {
    pub fn descriptor(self) -> ComponentDescriptor {
        let (_, field, component_type) = FIELDS[self as usize];
        let archetype = ArchetypeName::from(ARCHETYPE);
        ComponentDescriptor {
            archetype: Some(archetype),
            component: ComponentIdentifier::from_archetype_field(archetype, field),
            component_type: Some(component_type.into()),
        }
    }

    pub fn component(self) -> ComponentIdentifier {
        self.descriptor().component
    }
}

pub fn all_components() -> Vec<ComponentDescriptor> {
    FIELDS
        .iter()
        .map(|(field, ..)| field.descriptor())
        .collect()
}

/// Viewer default for `meter`: one unit is one millimetre.
pub const DEFAULT_METER: f32 = 1000.0;

#[cfg(test)]
mod tests {
    use super::*;
    use re_viewer::external::re_sdk_types::{Component as _, components};

    /// The Python helper and the prototype write `DepthVideoStream:<field>` under the full archetype name.
    #[test]
    fn descriptors_match_the_python_helper() {
        for (i, (field, ..)) in FIELDS.iter().enumerate() {
            assert_eq!(*field as usize, i, "FIELDS is indexed by Field");
        }
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
