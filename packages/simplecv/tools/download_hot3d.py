import tyro

from simplecv.apis.download_hot3d import DownloadConfig, main

if __name__ == "__main__":
    main(tyro.cli(DownloadConfig, description="Download HOT3D sequences from CDN URL JSON."))
