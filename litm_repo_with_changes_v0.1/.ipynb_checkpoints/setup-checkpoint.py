#!/usr/bin/env python3
"""
Mostly taken from https://github.com/rochacbruno/python-project-template/blob/main/setup.py
"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if line.strip() and not line.startswith(('"', "#", "-"))]


setup(
    name="modeling_attn_collapse",
    version="0.1.0",
    description="Repository for modeling attention collapse and hallucination risk in long-context LLMs.",
    url="https://github.com/ivanadebie/Modeling-Attn-Collapse--JMIA",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)