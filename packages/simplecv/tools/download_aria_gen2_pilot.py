import tyro

from simplecv.apis.download_aria_gen2_pilot import DownloadConfig, main

if __name__ == "__main__":
    main(tyro.cli(DownloadConfig, description="Download Aria Gen2 Pilot sequences from CDN URL JSON."))
