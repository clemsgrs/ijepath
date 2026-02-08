# CI Workflows

## Release workflows

- `.github/workflows/docker.yaml`: On GitHub Release publish, builds and pushes:
  - `waticlems/ijepath:<tag>`
  - `waticlems/ijepath:latest`
- `.github/workflows/release.yaml`: On GitHub Release publish, builds Python package artifacts and uploads them to PyPI using `PYPI_API_TOKEN`.

## Packaged defaults

- Default config resources are packaged in wheel artifacts under:
  - `configs/profiles/*.yaml`
  - `configs/runs/*.yaml`
- Packaging is defined in `pyproject.toml` and `MANIFEST.in`.

## Required repository secrets

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `PYPI_API_TOKEN`
