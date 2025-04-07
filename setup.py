from setuptools import setup, find_packages

setup(
    name="textguard-ai",
    version="1.0.0",
    description="TextGuard AI - DeepSeek MCP Client for Text Analysis",
    author="Algorithm07 AI",
    author_email="your-email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.6",
        "requests>=2.31.0",
        "python-multipart>=0.0.6",
        "typing-extensions>=4.8.0",
        "tenacity>=8.2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "httpx>=0.25.1",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    project_urls={
        "Source": "https://github.com/algorithm07-ai/TextGuardAI",
        "Documentation": "https://github.com/algorithm07-ai/TextGuardAI#readme",
    },
) 