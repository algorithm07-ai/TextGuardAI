from setuptools import setup, find_packages

setup(
    name="textguard-ai",
    version="1.0.0",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
        "python-multipart>=0.0.5",
        "httpx>=0.24.1",
        "huggingface-hub>=0.19.3",
        "gradio>=4.19.2",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.2"
    ],
    author="algorithm07-ai",
    author_email="your.email@example.com",
    description="Text classification and spam detection API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/algorithm07-ai/TextGuardAI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
) 