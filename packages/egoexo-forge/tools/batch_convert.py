import tyro

from egoexo_forge.api.batch_raw_to_rrd import BatchConvertConfig, batch_raw_to_rrd

# Example usage
if __name__ == "__main__":
    batch_raw_to_rrd(
        tyro.cli(
            BatchConvertConfig,
            description="Visualize Ego Only dataset",
        )
    )
