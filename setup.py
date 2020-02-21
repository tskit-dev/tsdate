from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tsdate",
    author="Anthony Wilder Wohns and Yan Wong",
    author_email="awohns@gmail.com",
    description="Infer node ages from a tree sequence topology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    entry_points={
        'console_scripts': [
            'tsdate=tsdate.cli:tsdate_main',
        ]
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        "numpy>=1.17.0",
        "tskit>=0.2.3",
        "scipy>1.2.3",
        "numba",
        "tqdm"
    ],
    project_urls={
        'Source': 'https://github.com/awohns/tsdate',
        'Bug Reports': 'https://github.com/awohns/tsdate/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OSX",
    ],
    use_scm_version={"write_to": "tsdate/_version.py"},
)
