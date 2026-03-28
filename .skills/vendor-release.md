# Vendor Renaming and Releasing

This skill performs vendor renaming of Triton specifically for tokenspeed dependency.

## Goals

* Package the latest Triton as Python wheel under tokenspeed vendor naming and releasing schemes
* Enable parallel existence to the canonical Triton pip wheels, which is a dependency to PyTorch

## Tasks

* Update version patch number to 10
* Rename wheel name to `tokenspeed-triton`
* Rename Python module name to `tokenspeed_triton`
* Rename kernel cache directory to `tokenspeed-cache` under `~/.triton/`
* Split Proton related components into its own `tokenspeed-proton` wheel
* `tokenspeed-proton` should depends on `tokenspeed-triton`, and all Proton related components,
  including cupti and vendor toolchain, should move to the former package
* Replace pybind11 with nanobind; build separate wheels for python 3.10, 3.11, 3.12 and above
* Fix core compiler and runtime references to make everything consistent
* Make sure enabling both NVIDIA and AMD backends for both triton and proton wheels
* Make sure using `Release` to build C++ components for wheels
* Trim unnecessary components to generate smallest wheels

Also note that
* Activate local Python venv and perform actions inside
* Don't change any compiler and runtime features or functionality
* Don't need to fix all Python unit test and example imports

## Testing

Build and install locally
* Make sure normal `pip install` works properly at project root directory for both wheels
  * Make sure we don't need to remove existing `triton` pip package, if any
* After installation, make sure LIT tests pass by `cd build/cmake.*/test` and `lit -v .`

Build and install via wheels
* Follow steps you have put in `wheels.yml` to build both triton and proton wheels to make sure
  steps are okay; just need to temporary build the Python version matching current Python
  environment.
* Verify by installing the built wheels to check it's okay. Also don't remove existing
  `triton` pip package, if any.
* After installation, make sure `pip show tokenspeed-triton` and `pip show tokenspeed-proton`
  shows proper information and version in the format of `x.y.z.postYYYYMMDD`
  * Make sure `pip show triton` still works, if any
* After installation, update imports in `python/tutorials/` tutorial 01 to 10 test them to make
  sure it works
  * Remove `~/.triton/cache` and `~/.triton/tokenspeed-cache` before testing and check we generate
    kernel intermediates in `~/.triton/tokenspeed-cache`.
* After installation, update imports in `pytest -n8 python/test/gluon/test_core.py` and test to
  make sure it works
* Afer installation, update imports in `pytest -n8 third_party/proton/test/test_api.py` and test
  to make sure it works

Build and test wheelhouse
* Perform cibuildwheel like `wheels.py` for current Python version to make sure it works.
* Make sure generated triton and proton wheels are both less than 100MB.

## Committing

After all tasks performed and tests succeeded, make a git commit of all changed files.
* Only check in existing/renamed files; don't check in temporary or build intermeidate files
