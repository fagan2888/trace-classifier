from setuptools import find_packages
from setuptools import setup


# Parse the version from the module.
with open("trace_classifier/__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break


# Requirements
requirements = [
    "keras==2.2.2",
    "tensorflow==1.10.0",
    "tensorframes==0.2.9",
    "numpy==1.14.5",
    "pandas==0.23.1",
    "matplotlib==3.0.0",
    "h5py==2.8.0",
    "pyspark==2.4.4",
    "pyarrow==0.8.0",
]

# Extra requirements
extra_require = {"test": ["pytest", "pytest-cov", "codecov"]}


url = "https://github.com/mapbox/trace-classifier"

setup(
    name="trace-classifier",
    version=version,
    description="A classifier for location traces",
    long_description="See " + url,
    classifiers=[],
    keywords="",
    author="@lily-chai",
    author_email="lily@mapbox.com",
    url=url,
    license="MIT",
    packages=find_packages(exclude=["ez_setup", "examples", "tests", "notebooks"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    extras_require=extra_require,
    entry_points="""""",
)
