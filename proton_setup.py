"""
Setup script for tokenspeed-proton: the Proton profiler split out from tokenspeed-triton.

Builds libproton.so (nanobind native module) from third_party/proton/ and packages
the profiler Python code. Depends on tokenspeed-triton at runtime.

libproton.so only links against Python3::Module (CUPTI/roctracer loaded via dlopen),
so this build does NOT require LLVM/MLIR.
"""

import contextlib
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

import nanobind

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel

sys.path.insert(0, os.path.dirname(__file__))

from python.build_helpers import get_base_dir


def get_triton_cache_path():
    user_home = os.getenv("TRITON_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_env_with_keys(key: list):
    for k in key:
        if k in os.environ:
            return os.environ[k]
    return ""


def get_base_version():
    """Read the base version (e.g. '3.7.10') from setup_triton.py or setup.py."""
    base_dir = os.path.dirname(__file__)
    for candidate in ["setup_triton.py", "setup.py"]:
        path = os.path.join(base_dir, candidate)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()
        match = re.search(r'TRITON_VERSION\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return "0.0.0"


def get_post_timestamp():
    """Generate .postYYYYMMDD suffix from the git commit timestamp (UTC)."""
    try:
        cmd = ['git', 'show', '--quiet', '--date=format-local:%Y%m%d', '--format=%cd']
        env = {**os.environ, 'TZ': 'UTC0'}
        timestamp = subprocess.check_output(cmd, env=env).strip().decode('utf-8')
        return f'.post{timestamp}'
    except Exception:
        return ""


def get_version():
    """Build the full version string, matching the triton package versioning."""
    base = get_base_version()
    return base + get_post_timestamp()


def ensure_json_headers():
    """Download nlohmann/json headers if not already cached."""
    cache_path = get_triton_cache_path()
    json_root = os.path.join(cache_path, "json")
    json_include = os.path.join(json_root, "include")
    version_file = os.path.join(json_root, "version.txt")

    json_version_path = os.path.join(get_base_dir(), "cmake", "json-version.txt")
    with open(json_version_path) as f:
        version = f.read().strip()

    url = f"https://github.com/nlohmann/json/releases/download/{version}/include.zip"

    if os.path.exists(version_file) and Path(version_file).read_text().strip() == url:
        return json_include

    with contextlib.suppress(Exception):
        shutil.rmtree(json_root)

    os.makedirs(json_root, exist_ok=True)
    archive_path = os.path.join(json_root, "include.zip")
    print(f"Downloading nlohmann/json {version}...")
    urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(json_root)
    os.remove(archive_path)

    with open(version_file, "w") as f:
        f.write(url)

    return json_include


class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class ProtonBdistWheel(bdist_wheel):

    def get_tag(self):
        if check_env_flag("TRITON_STABLE_ABI"):
            return "cp312", "abi3", super().get_tag()[2]
        return super().get_tag()


class CMakeBuildPy(build_py):
    def run(self) -> None:
        self.run_command("build_ext")
        return super().run()


class ProtonCMakeBuild(build_ext):

    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of Triton"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ninja_dir = shutil.which("ninja")
        assert ninja_dir is not None, "ninja not found!"

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        python_include_dir = sysconfig.get_path("platinclude")

        cupti_include_dir = get_env_with_keys(["TRITON_CUPTI_INCLUDE_PATH"])
        if not cupti_include_dir:
            cupti_include_dir = os.path.join(get_base_dir(), "third_party", "nvidia", "backend", "include")

        roctracer_include_dir = get_env_with_keys(["TRITON_ROCTRACER_INCLUDE_PATH"])
        if not roctracer_include_dir:
            roctracer_include_dir = os.path.join(get_base_dir(), "third_party", "amd", "backend", "include")

        json_include_dir = get_env_with_keys(["JSON_INCLUDE_DIR"])
        if not json_include_dir:
            json_include_dir = ensure_json_headers()

        cmake_args = [
            "-G", "Ninja",
            "-DCMAKE_MAKE_PROGRAM=" + ninja_dir,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPython3_INCLUDE_DIR=" + python_include_dir,
            f"-Dnanobind_ROOT={nanobind.cmake_dir()}",
            f"-DCUPTI_INCLUDE_DIR={cupti_include_dir}",
            f"-DROCTRACER_INCLUDE_DIR={roctracer_include_dir}",
            f"-DJSON_INCLUDE_DIR={json_include_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DCMAKE_CXX_FLAGS=-fPIC -std=gnu++17 -ftls-model=global-dynamic",
        ]

        if check_env_flag("TRITON_STABLE_ABI"):
            cmake_args.append("-DTRITON_STABLE_ABI=ON")

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        proton_dir = os.path.join(self.base_dir, "third_party", "proton")
        build_dir = os.path.join(self.build_temp, "proton")
        os.makedirs(build_dir, exist_ok=True)

        env = os.environ.copy()
        subprocess.check_call(["cmake", proton_dir] + cmake_args, cwd=build_dir, env=env)

        max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "-j" + max_jobs, "--target", "proton"],
            cwd=build_dir,
        )


setup(
    name="tokenspeed-proton",
    version=get_version(),
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A profiler for Triton (vendor release for TokenSpeed)",
    long_description="",
    license="MIT",
    install_requires=[
        "tokenspeed-triton",
    ],
    packages=[
        "tokenspeed_triton.profiler",
        "tokenspeed_triton.profiler.hooks",
    ],
    package_dir={
        "tokenspeed_triton.profiler": "third_party/proton/proton",
        "tokenspeed_triton.profiler.hooks": "third_party/proton/proton/hooks",
    },
    entry_points={
        "console_scripts": [
            "proton-viewer = tokenspeed_triton.profiler.viewer:main",
            "proton = tokenspeed_triton.profiler.proton:main",
        ],
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("tokenspeed_triton", "tokenspeed_triton/_C/")],
    cmdclass={
        "bdist_wheel": ProtonBdistWheel,
        "build_ext": ProtonCMakeBuild,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    url="https://github.com/triton-lang/triton/",
    python_requires=">=3.10",
)
