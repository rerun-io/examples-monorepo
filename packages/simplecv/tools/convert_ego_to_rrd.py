import tyro

from simplecv.apis.convert_to_rrd import ConvertEgoConfig, convert_ego

# Example usage
if __name__ == "__main__":
    convert_ego(
        tyro.cli(
            ConvertEgoConfig,
            description="Convert Ego Only dataset to RRD format",
        )
    )
