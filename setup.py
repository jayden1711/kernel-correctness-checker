from setuptools import setup, find_packages

setup(
    name="kernel-correctness-checker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
        "triton-viz",
    ],
    python_requires=">=3.9",
)