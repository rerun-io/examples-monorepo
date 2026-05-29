import tyro

from simplecv.apis.convert_to_ns import ConvertConfig, convert_data_to_ns

if __name__ == "__main__":
    convert_data_to_ns(tyro.cli(ConvertConfig))
