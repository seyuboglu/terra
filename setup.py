import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="terra",
    version="0.0.1",
    author="Sabri Eyuboglu",
    author_email="eyuboglu@stanford.edu",
    description="A Python package that transforms free-form research workflows into reproducible pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seyuboglu/terra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["terra=terra.cli:cli"]},
    python_requires=">=3.8",
    # TODO(sabri): test these version lower bounds 
    install_requires=[
        "pandas>=1.1.0",
        "numpy>=1.0",
        "slackclient>=2.0",
        "pytest>=6.0",
        "sqlalchemy>=2.0",
        "click>=7.0.0",
        "torch>=1.0.0",
    ],
)
