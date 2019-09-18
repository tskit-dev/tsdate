from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext


# Obscure magic required to allow numpy be used as an 'setup_requires'.
class build_ext(_build_ext):
    def finalize_options(self):
        super(build_ext, self).finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tsdate",
    author="Anthony Wilder Wohns",
    author_email="awohns@gmail.com",
    description="Infer node ages from a tree sequence topology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    setup_requires=['setuptools_scm'],
    install_requires=[
        "numpy>=1.17.0",
        "tskit>=0.2.1",
        "tsinfer>=0.1.4",
        "tqdm"
    ],
    project_urls={
        'Source': 'https://github.com/awohns/tsdate',
        'Bug Reports': 'https://github.com/awohns/tsdate/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    use_scm_version={"write_to": "tsdate/_version.py"},
)
