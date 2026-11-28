"""Mask layer contract (track 3, v2): per camera ``<camera>/pinhole/people_mask``, a ``SegmentationImage``
(nonzero = masked) with a static ``Transform3D(scale=[sx, sy, 1])`` from mask cells to image pixels.

Soft mattes sit beside it: per camera ``<camera>/pinhole/people_matte``, an 8-bit PNG ``EncodedImage`` (255 = person,
0 = background, values between = partial coverage) at the stored video size and grid."""

import rerun as rr
from simplecv.catalog_layers import LENS_MASK_LAYER as LENS_MASK_LAYER

MASK_NAME: str = "people_mask"
"""Leaf entity of a mask under ``<camera>/pinhole``."""
MATTE_NAME: str = "people_matte"
"""Leaf entity of a soft people matte under ``<camera>/pinhole`` (8-bit PNG ``EncodedImage``, 255 = person)."""
MASKED_CLASS_ID: int = 1
"""Mask value of a masked pixel (nonzero = drop features there)."""
MASK_ANNOTATION: rr.AnnotationContext = rr.AnnotationContext([(0, "kept", (255, 255, 255, 0)), (MASKED_CLASS_ID, "masked", (40, 110, 255, 140))])
"""Static annotation logged on every mask entity."""
