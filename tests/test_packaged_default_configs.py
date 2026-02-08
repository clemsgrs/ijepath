from importlib import resources

import configs


def test_default_profile_config_is_packaged():
    profile = resources.files("configs").joinpath(
        "profiles/ctx1p0_tgt0p5_fov512um_k4.yaml"
    )
    assert profile.is_file()


def test_default_run_config_is_packaged():
    run = resources.files("configs").joinpath(
        "runs/tcga_prad_smoke.yaml"
    )
    assert run.is_file()


def test_helper_api_lists_profile_and_run_defaults():
    profiles = configs.list_default_profile_configs()
    runs = configs.list_default_run_configs()
    assert "ctx1p0_tgt0p5_fov512um_k4.yaml" in profiles
    assert "tcga_prad_smoke.yaml" in runs


def test_helper_api_reads_packaged_config_text():
    text = configs.read_default_config("profiles/ctx1p0_tgt0p5_fov512um_k4.yaml")
    assert "profile_id:" in text
