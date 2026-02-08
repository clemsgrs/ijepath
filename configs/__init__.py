from importlib import resources


def _list_yaml(relative_dir: str) -> list[str]:
    directory = resources.files(__name__).joinpath(relative_dir)
    return sorted(
        item.name
        for item in directory.iterdir()
        if item.is_file() and item.name.endswith(".yaml")
    )


def list_default_profile_configs() -> list[str]:
    return _list_yaml("profiles")


def list_default_run_configs() -> list[str]:
    return _list_yaml("runs")


def read_default_config(relative_path: str) -> str:
    resource = resources.files(__name__).joinpath(relative_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Default config not found: {relative_path}")
    return resource.read_text(encoding="utf-8")
