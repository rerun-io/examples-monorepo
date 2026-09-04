# LAMP tracker

This package ports Meta's [LAMP](https://github.com/facebookresearch/LAMP) model
into the examples monorepo. It combines PoseKit 2D person poses from four
synchronized cameras with LAMP's temporal SMPL lifter.

LAMP code and weights use **CC-BY-NC 4.0**. This package is for
non-commercial use only. See `lamptrack/third_party/lamp/LICENSE`.

Download the checkpoint and neutral SMPL body model:

```bash
pixi run -e lamp lamp-download-artifacts
```

The replay fixture is not published with the package. After the reviewer adds
it to `pablovela5620/lamp-fixtures`, download and replay it with:

```bash
pixi run -e lamp _lamp-download-fixture
pixi run -e lamp lamp-replay --rr-config.live --rr-config.no-headless
```
