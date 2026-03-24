"""Minimal setup.py so the project can be installed in editable mode."""

from setuptools import setup, find_packages

setup(
    name="running_gait_analysis",
    version="0.1.0",
    description="Automated running gait analysis using deep learning and computer vision.",
    packages=find_packages(),
    python_requires=">=3.10",
)
