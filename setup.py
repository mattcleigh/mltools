"""Setup for pip installable package."""

import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f]

setuptools.setup(
    name="mltools",
    version="2.0.0",
    author="Matthew Leigh",
    author_email="mattcleigh@gmail.com",
    description="Some common utilities used in my DL projects",
    long_description=long_description,
    url="https://gitlab.cern.ch/mleigh/mltools",
    project_urls={"Bug Tracker": "https://gitlab.cern.ch/mleigh/mltools/issues"},
    license="MIT",
    packages=["mltools", "mltools", "mltools.gnets"],
    install_requires=requirements,
)
