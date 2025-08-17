"""Setup script for EchoQA package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Requirements are now managed in the main pyproject.toml
requirements = []

setup(
    name="echo_qa",
    version="0.1.0",
    author="EchoPrime Team",
    description="Medical Image Caption Processing and Question Answering using OpenAI's GPT models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "echo-qa=echo_qa.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="medical imaging, openai, gpt, question answering, caption processing",
    project_urls={
        "Bug Reports": "https://github.com/echoprime/echo_qa/issues",
        "Source": "https://github.com/echoprime/echo_qa",
    },
)
