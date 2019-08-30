import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsdate",
    author="Anthony Wilder Wohns",
    author_email="awohns@gmail.com",
    description="Infer node ages from a tree sequence topology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    setup_requires=['setuptools_scm'],
    install_requires=[
        "numpy",
        "tskit",
        "tqdm"
    ],
    project_urls={
        'Source': 'https://github.com/awohns/tsdate',
        'Bug Reports': 'https://github.com/awohns/tsdate/issues',
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    use_scm_version={"write_to": "tsdate/_version.py"},
)
