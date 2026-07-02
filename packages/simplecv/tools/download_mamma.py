import tyro

from simplecv.apis.download_mamma import DownloadConfig, main

if __name__ == "__main__":
    main(tyro.cli(DownloadConfig, description="Download one MAMMA sequence per subset (needs MAMMA_USERNAME/MAMMA_PASSWORD)."))
