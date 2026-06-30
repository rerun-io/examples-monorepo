import tyro

from simplecv.apis.exoego_forge_catalog import RegisterConfig, register_main

if __name__ == "__main__":
    register_main(tyro.cli(RegisterConfig, description="Register ExoEgo Forge RRD roots into a running catalog (tier 2)."))
