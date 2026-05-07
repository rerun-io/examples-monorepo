import tyro

from egoexo_forge.api.reencode_vids import EncodeConfig, reencode_videos

# Example usage
if __name__ == "__main__":
    reencode_videos(
        tyro.cli(
            EncodeConfig,
            description="Re-encode videos",
        )
    )
