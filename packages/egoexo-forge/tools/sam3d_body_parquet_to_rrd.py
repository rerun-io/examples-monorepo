import tyro

from egoexo_forge.api.sam3d_body_parquet_to_rrd import (
    Sam3dBodyParquetToRrdConfig,
    convert_sam3d_body_parquet_to_rrd,
)


def main(config: Sam3dBodyParquetToRrdConfig) -> None:
    result = convert_sam3d_body_parquet_to_rrd(config)
    print(f"Wrote {result.rrd_path}")
    print(f"rows={result.row_count}")


if __name__ == "__main__":
    main(
        tyro.cli(
            Sam3dBodyParquetToRrdConfig,
            description="Convert one facebook/sam-3d-body-dataset parquet shard into an Exo/Ego-schema Rerun RRD.",
        )
    )
