import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mattstools",
    version="0.0.2",
    author="Matthew Leigh",
    author_email="mattcleigh@gmail.com",
    description="Some common utilities used in my DL projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cern.ch/mleigh/mattstools",
    project_urls={"Bug Tracker": "https://gitlab.cern.ch/mleigh/mattstools/issues"},
    license="MIT",
    packages=["mattstools"],
    install_requires=[
        "geomloss",
        "matplotlib",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit_learn",
        "scipy",
        "setuptools",
        "torch",
        "tqdm",
        "typing_extensions",
    ],
)
