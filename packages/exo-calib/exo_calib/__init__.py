import os

# In the dev env beartype checks every annotated local, and Stage B's per-frame loops pay for it:
# memory grows by about 1 MiB per frame and the stage runs about 2x slower (4,000-frame capture).
# Accepted for dev runs; full captures run in the prod env (`-e exo-calib`), which never loads beartype.
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
