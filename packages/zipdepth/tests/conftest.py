"""Keep synthetic transform tests independent of the network."""

import os

# Albumentations otherwise contacts PyPI during test collection.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
