from simplecv.rerun_dataloader import RECOMMENDED_FETCH_BLOCK_SIZE

from mvs.apis.live_mesh import CLOUD_CATALOG_URL, CatalogDataConfig


def test_catalog_data_config_defaults() -> None:
    """The default input configuration points at the cloud ARKitScenes dataset."""

    config = CatalogDataConfig()

    assert config.catalog_url == CLOUD_CATALOG_URL
    assert config.dataset_name == "arkitscenes"
    assert config.segments == ("42899799",)
    assert RECOMMENDED_FETCH_BLOCK_SIZE == 1024
