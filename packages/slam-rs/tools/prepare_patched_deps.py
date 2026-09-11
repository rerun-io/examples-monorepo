import tyro

from slam_rs.apis.prepare_patched_deps import Config, main

if __name__ == '__main__':
    main(tyro.cli(Config))
