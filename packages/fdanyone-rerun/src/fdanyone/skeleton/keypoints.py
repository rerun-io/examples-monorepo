"""Minimal MHR70/Goliath40 conditioning schema.

The exact keypoint names/order and no-finger link closure were extracted from
``facebookresearch/sapiens2`` revision
``0e51c12d7c7257d88431b2d50e523a7b03004854`` and remain subject to the
Sapiens2 License. A copy is kept at
``third_party/licenses/SAPIENS2_LICENSE.md``. The RGB palette, per-keypoint
color policy, and renderer implementation are original 4DAnyone material
licensed under Apache-2.0.
"""

from __future__ import annotations

# The exact names/order below are derived from Sapiens2, Copyright (c) Meta
# Platforms, Inc. and affiliates, under the Sapiens2 License noted above.

KEYPOINT_NAMES = (
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-thumb-tip",
    "right-thumb-first-joint",
    "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip",
    "right-index-first-joint",
    "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip",
    "right-middle-first-joint",
    "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip",
    "right-ring-first-joint",
    "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip",
    "right-pinky-first-joint",
    "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip",
    "left-thumb-first-joint",
    "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip",
    "left-index-first-joint",
    "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip",
    "left-middle-first-joint",
    "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip",
    "left-ring-first-joint",
    "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip",
    "left-pinky-first-joint",
    "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon",
    "right-olecranon",
    "left-cubital-fossa",
    "right-cubital-fossa",
    "left-acromion",
    "right-acromion",
    "neck",
)

# First-party 4DAnyone color palette, licensed under Apache-2.0.
BLUE = (116, 192, 252)
GREEN = (130, 186, 129)
ORANGE = (248, 129, 81)
TEAL = (99, 230, 190)
YELLOW = (255, 212, 59)
PINK = (229, 153, 247)
PURPLE = (177, 151, 252)
RED = (255, 135, 135)

# The schema ids and link endpoints are Sapiens2-derived. The RGB assignments
# are first-party 4DAnyone material.
# (schema id, first keypoint, second keypoint, RGB, major)
LINKS = (
    (0, 13, 11, TEAL, True),
    (1, 11, 9, TEAL, True),
    (2, 14, 12, YELLOW, True),
    (3, 12, 10, YELLOW, True),
    (4, 9, 10, BLUE, True),
    (5, 5, 9, GREEN, True),
    (6, 6, 10, ORANGE, True),
    (7, 5, 6, BLUE, True),
    (8, 5, 7, TEAL, True),
    (9, 6, 8, YELLOW, True),
    (10, 7, 62, TEAL, True),
    (11, 8, 41, YELLOW, True),
    (12, 1, 2, BLUE, True),
    (13, 0, 1, GREEN, True),
    (14, 0, 2, ORANGE, True),
    (15, 1, 3, GREEN, True),
    (16, 2, 4, ORANGE, True),
    (17, 3, 5, GREEN, True),
    (18, 4, 6, ORANGE, True),
    (19, 13, 15, TEAL, True),
    (20, 13, 16, TEAL, True),
    (21, 13, 17, TEAL, True),
    (22, 14, 18, YELLOW, True),
    (23, 14, 19, YELLOW, True),
    (24, 14, 20, YELLOW, True),
    (25, 62, 45, YELLOW, False),
    (29, 62, 49, PINK, False),
    (33, 62, 53, PURPLE, False),
    (37, 62, 57, RED, False),
    (41, 62, 61, TEAL, False),
    (45, 41, 24, YELLOW, False),
    (49, 41, 28, PINK, False),
    (53, 41, 32, PURPLE, False),
    (57, 41, 36, RED, False),
    (61, 41, 40, TEAL, False),
    (65, 5, 10, PURPLE, True),
    (66, 6, 9, PURPLE, True),
    (67, 3, 4, PURPLE, True),
)

EXTRA_KEYPOINT_IDS = frozenset(range(63, 70))


def keypoint_color(keypoint_id: int) -> tuple[int, int, int]:
    name = KEYPOINT_NAMES[keypoint_id]
    if name == "nose" or name == "neck":
        return BLUE
    if name in {"left-eye", "left-ear"}:
        return GREEN
    if name in {"right-eye", "right-ear"}:
        return ORANGE
    if "thumb" in name:
        return YELLOW
    if "forefinger" in name or "index" in name:
        return PINK
    if "middle" in name:
        return PURPLE
    if "ring" in name:
        return RED
    if "pinky" in name:
        return TEAL
    return TEAL if name.startswith("left-") else YELLOW if name.startswith("right-") else BLUE


VISIBLE_KEYPOINT_IDS = (
    frozenset({index for _, first, second, _, _ in LINKS for index in (first, second)}) | EXTRA_KEYPOINT_IDS
)
