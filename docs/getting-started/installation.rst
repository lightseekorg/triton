============
Installation
============

For supported platform/OS and supported hardware, review the `Compatibility <https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility>`_ section on Github.

--------------------
Binary Distributions
--------------------

You can install the latest stable release of Triton from pip:

.. code-block:: bash

      pip install triton

Binary wheels are available for CPython 3.10-3.14.

-----------
From Source
-----------

++++++++++++++
Python Package
++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git
      cd triton

      pip install -r python/requirements.txt # build-time dependencies
      pip install -e .

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

On Windows, use a Visual Studio developer shell with the C++ build tools and
Windows SDK installed. The build downloads the pinned Windows LLVM and NVIDIA
toolchain packages and bundles TinyCC for compiling runtime launchers. Both
x64 and ARM64 builds are supported. The default distribution name is
``triton-windows``; the Python import remains ``triton``.

After installing the build-time dependencies, build from that shell with:

.. code-block:: powershell

      python -m pip install -e . --no-build-isolation

Windows does not require ``make``: use ``cmake --build`` with the build directory
reported by ``python.build_helpers.get_cmake_dir()`` for incremental native
builds, and ``python -m pytest -s --tb=short`` for individual Python tests.
Running GPU kernels requires a supported NVIDIA GPU/driver or a compatible
ROCm for Windows installation.

For building with a custom LLVM, review the `Building with a custom LLVM <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ section on Github.

You can then test your installation by running the tests:

.. code-block:: bash

      # One-time setup
      make dev-install

      # To run all tests (requires a GPU)
      make test

      # Or, to run tests without a GPU
      make test-nogpu
