import tyro

from exo_calib.apis.blueprint import BlueprintConfig, main

if __name__ == "__main__":
    main(tyro.cli(BlueprintConfig, description="Write the exocalib .rbl blueprint for a catalog dataset."))
