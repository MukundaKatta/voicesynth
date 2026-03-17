from setuptools import setup, find_packages

setup(
    name="voicesynth",
    version="0.1.0",
    description="State-of-the-art Text-to-Speech with voice cloning and emotion control",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MukundaKatta",
    url="https://github.com/MukundaKatta/voicesynth",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "scipy>=1.10", "fastapi>=0.100", "uvicorn>=0.22", "pydantic>=2.0"],
    extras_require={
        "full": ["torch>=2.0", "torchaudio>=2.0", "librosa>=0.10"],
        "dev": ["pytest>=7.0", "httpx>=0.24"],
    },
)
