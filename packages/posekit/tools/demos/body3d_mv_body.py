import tyro

from posekit.apis.body3d_mv_body import Sam3MVBodyDemoConfig, main

if __name__ == "__main__":
    # Flags follow both dataset and body subcommands; the Pixi task supplies only the dataset.
    main(tyro.cli(Sam3MVBodyDemoConfig, config=(tyro.conf.ConsolidateSubcommandArgs,)))
