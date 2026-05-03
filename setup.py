# Run this by calling
#     python setup.py sdist bdist_wheel # old way to build a package
#     or
#     python -m build                   # new way to build a package

import os
from setuptools import setup, find_packages


def get_static_files(root):
    out = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            out.append(os.path.join(path, name))
    return [
        d.strip("./pciSeq/")
        for d in out
        if (
            d.endswith(".html")
            or d.endswith(".js")
            or d.endswith(".css")
            or d.endswith(".msi")
            or d.endswith(".so")
            or d.endswith(".json")
            or d.endswith("PotreeConverter")
        )
    ]


install_deps = [
    "numpy_groupies",
    "pandas",
    "dask",
    "scipy",
    "streamlit",
    "altair",
    "scikit-image",
    "scikit-learn",
    "tqdm",
    "flask",
    "flask-socketio",
    "fastremap",
    "numexpr",
    "diplib",
    "pyvips[binary]",
    "natsort",
    "redis",
    "matplotlib",
    "laspy",
    "tomlkit",
    "colorlog",
    "shapely",
    "alphashape",
    "opt_einsum",
    "plotly",
    "numba",
    "pyarrow",
]

# GPU acceleration via CuPy (Linux + Windows; pip skips on Mac since Apple
# dropped CUDA in 2018). At runtime, b_upd auto-falls-back to the numpy version
# if CuPy or a GPU is missing, so the package still imports everywhere.
_gpu_marker = "sys_platform != 'darwin'"
gpu_deps = [
    f"{pkg}; {_gpu_marker}" for pkg in [
        "cupy-cuda12x>=14.0",
        "nvidia-cublas-cu12>=12.4",
        "nvidia-cusparse-cu12>=12.4",
        "nvidia-cufft-cu12>=11.0",
        "nvidia-cusolver-cu12>=11.0",
        "nvidia-curand-cu12>=10.0",
        "nvidia-cuda-runtime-cu12>=12.0",
        "nvidia-cuda-nvrtc-cu12>=12.0",
        "nvidia-nvjitlink-cu12>=12.4",
    ]
]
install_deps += gpu_deps


def get_version():
    """Get version from _version.py and append git commit hash if available."""
    version = None
    with open(os.path.join("pciSeq", "_version.py"), "r") as fid:
        for line in (line.strip() for line in fid):
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip("'\"")  # Strip both ' and "
                break

    if version is None:
        raise RuntimeError("Could not determine version")

    # Try to append git commit hash
    # NOTE: Disabled because +g{commit} format causes pip install failures in CI
    # The version string must be PEP 440 compliant
    # try:
    #     import subprocess
    #     commit = subprocess.check_output(
    #         ['git', 'rev-parse', '--short', 'HEAD'],
    #         stderr=subprocess.DEVNULL,
    #         text=True
    #     ).strip()
    #     version = f"{version}+g{commit}"
    # except Exception:
    #     # No git or not in a git repo - use version as is
    #     pass

    return version


version = get_version()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pciSeq_3d",
    version=version,
    license="BSD",
    author="Dimitris Nicoloutsopoulos",
    author_email="dimitris.nicoloutsopoulos@gmail.com",
    description="Probabilistic cell typing for spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/acycliq/pciSeq_3d",
    packages=find_packages(),
    install_requires=install_deps,
    extras_require={
        "interactive": ["matplotlib>=2.2.0", "jupyter"],
    },
    include_package_data=True,
    package_data={
        "pciSeq": get_static_files(os.path.join("pciSeq", "static"))
        + get_static_files(os.path.join("pciSeq", "src", "realtime_viewer"))
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
