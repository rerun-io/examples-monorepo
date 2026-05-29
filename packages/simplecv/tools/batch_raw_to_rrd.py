import tyro

from simplecv.apis.batch_raw_to_rrd import BatchConvertConfig, main

if __name__ == "__main__":
    config: BatchConvertConfig = tyro.cli(BatchConvertConfig)
    main(config)
