"""CLI entrypoint shim. Logic lives in ``live_rerun.apis.oak_live_rerun`` so it is
type-checked by ``beartype_this_package()`` when running under the dev environment."""

import tyro

from live_rerun.apis import oak_live_rerun

if __name__ == "__main__":
    oak_live_rerun.main(tyro.cli(oak_live_rerun.OakLiveRerunConfig, description=oak_live_rerun.__doc__))
