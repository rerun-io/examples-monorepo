import tyro

from colsfm.apis.ba_accuracy import Config, MergeConfig, main, merge_main

if __name__ == "__main__":
    command = tyro.extras.subcommand_cli_from_dict({"measure": Config, "merge": MergeConfig})
    (main if isinstance(command, Config) else merge_main)(command)
