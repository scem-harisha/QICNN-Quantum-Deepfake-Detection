from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="qicnn-deepfake-detection",
    version="0.1.0",
    author="QICNN Contributors",
    description="Quantum-Inspired CNN for Deepfake Detection with Cross-Dataset Generalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scem-harisha/QICNN-Quantum-Deepfake-Detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
