from setuptools import setup, find_packages

setup(
    name="graphpy",
    version="0.1.0",
    description="A very specific Python package for plotting graphs the way I like them.",
    author="Orlando Timmerman",
    url="https://github.com/yourusername/graphpy",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.0",
        "numpy>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
