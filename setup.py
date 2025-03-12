from setuptools import setup, find_packages

setup(
    name="xlsr-transducer",
    version="0.1.0",
    description="XLSR-Transducer: Streaming ASR for Estonian",
    author="Idiap Research Institute",
    author_email="example@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "transformers>=4.25.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.62.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.3",
        "tensorboard>=2.10.0",
    ],
) 