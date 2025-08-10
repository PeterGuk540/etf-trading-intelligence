"""Setup script for ETF Trading Intelligence System"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="etf-trading-intelligence",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade quantitative trading platform for ETF sector rotation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/etf-trading-intelligence",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "ruff>=0.0.270",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "etf-train=training.train:main",
            "etf-predict=trading.predict:main",
            "etf-backtest=evaluation.backtest:main",
        ],
    },
)