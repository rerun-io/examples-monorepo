import tyro

from colsfm.apis.evaluate_kitti import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
