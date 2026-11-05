import tyro

from posekit.apis.body3d_mv_video import Sam3MVVideoDemoConfig, main

if __name__ == "__main__":
    # Every flag goes after the dataset subcommand, so flags appended to a Pixi task parse.
    main(tyro.cli(Sam3MVVideoDemoConfig, config=(tyro.conf.ConsolidateSubcommandArgs,)))
