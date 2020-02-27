from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tsdate",
    author="Anthony Wilder Wohns and Yan Wong",
    author_email="awohns@gmail.com",
    description="Infer node ages from a tree sequence topology",
    long_description=long_description,
    packages=["tsdate"],
    long_description_content_type="text/markdown",
    url="http://pypi.python.org/pypi/tsdate",
    python_requires='>=3.4',
    entry_points={
        'console_scripts': [
            'tsdate=tsdate.__main__:main',
        ]
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        "numpy",
        "tskit>=0.2.3",
        "scipy>1.2.3",
        "numba>=0.46.0",
        "tqdm",
        "appdirs"
    ],
    project_urls={
        'Source': 'https://github.com/awohns/tsdate',
        'Bug Reports': 'https://github.com/awohns/tsdate/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    use_scm_version={"write_to": "tsdate/_version.py"},
)
