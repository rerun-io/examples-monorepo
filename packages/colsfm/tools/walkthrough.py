import tyro

from colsfm.walkthrough import WalkthroughConfig, run_walkthrough

if __name__ == "__main__":
    run_walkthrough(tyro.cli(WalkthroughConfig))
