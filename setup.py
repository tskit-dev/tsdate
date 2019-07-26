import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tstime-awohns",
    version="0.0.1",
    author="Anthony Wilder Wohns",
    author_email="awohns@gmail.com",
    description="Infer node ages from a tree sequence topology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        'Source': 'https://github.com/awohns/tstime',
        'Bug Reports': 'https://github.com/awohns/tstime/issues',
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
