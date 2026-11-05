import tyro

from posekit.apis.body3d_mv_body import Sam3MVBodyDemoConfig, main

if __name__ == "__main__":
    # Every flag goes after the dataset subcommand, so flags appended to a Pixi task parse.
    main(tyro.cli(Sam3MVBodyDemoConfig, config=(tyro.conf.ConsolidateSubcommandArgs,)))
