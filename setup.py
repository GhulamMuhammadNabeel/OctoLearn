from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="octolearn",
    version="0.5.1",
    description="Structured AutoML Pipeline with Intelligent Dataset Profiling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ghulam_Muhammad_Nabeel",
    url="https://github.com/ghulam-nabeel/octolearn",
    project_urls={
        "Bug Tracker": "https://github.com/ghulam-nabeel/octolearn/issues",
        "Documentation": "https://github.com/ghulam-nabeel/octolearn",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "optuna>=2.0.0",
        "reportlab>=3.6.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "shap>=0.40.0"
    ],
    extras_require={
        "distributed": [
            "dask[complete]>=2021.9.0",
            "ray[default]>=2.0.0"
        ]
    },
    python_requires=">=3.8",
)
