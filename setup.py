import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="terra",  # Replace with your own username
    version="0.0.1",
    author="Sabri Eyuboglu",
    author_email="eyuboglu@stanford.edu",
    description="A package for running computational experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seyuboglu/terra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
