from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="moodlens",
    version="1.0.0",
    author="Your Name",
    author_email="you@example.com",
    description="AI-powered emotion & sentiment analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moodlens",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "nltk>=3.8",
        "textblob>=0.17",
        "transformers>=4.38",
        "torch>=2.1",
        "rich>=13.7",
    ],
    extras_require={
        "api":  ["fastapi>=0.110", "uvicorn[standard]>=0.27", "pydantic>=2.0", "httpx>=0.27"],
        "viz":  ["matplotlib>=3.8", "numpy>=1.26"],
        "full": ["fastapi>=0.110", "uvicorn[standard]>=0.27", "pydantic>=2.0",
                 "matplotlib>=3.8", "numpy>=1.26", "httpx>=0.27"],
        "dev":  ["pytest>=8.0", "pytest-cov>=4.1"],
    },
    entry_points={
        "console_scripts": [
            "moodlens=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="nlp sentiment emotion analysis ai psychology text-mining",
)
