import tyro

from slam_rs.apis.fleet_check import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
