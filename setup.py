from setuptools import setup, find_packages

setup(
    name="jitsi-connect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "psutil>=5.8.0",
        "PyJWT>=2.8.0",
        "aiohttp>=3.8.0",
        "python-socketio>=5.0.0",
        "dataclasses-json>=0.5.0",
        "prometheus_client>=0.11.0",
    ],
    extras_require={
        'cv': ["opencv-python>=4.0.0"],
        'dev': [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "black",
            "flake8",
            "mypy"
        ],
        'all': [
            "opencv-python>=4.0.0",
            "mediapipe>=0.8.0",
            "tensorflow>=2.0.0"
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.15.1',
            'pytest-cov>=4.0.0',
            'pytest-html>=3.0.0'
        ]
    },
    python_requires='>=3.8',
) 