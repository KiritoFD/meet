from setuptools import setup, find_packages

setup(
    name="meet",
    version="0.1.0",
    description="Meet AI Platform",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # 基础依赖
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'opencv-python>=4.5.0',
        'numpy>=1.19.2',
        'pillow>=8.0.0',
        'scipy>=1.6.0',
        'tqdm>=4.50.0',
        'mediapipe>=0.8.9',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.9',
            'isort>=5.0',
        ],
        'full': [
            'open3d>=0.13.0',
            'face-recognition>=1.3.0',
            'moviepy>=1.0.3',
            'huggingface-hub>=0.4.0',
            'transformers>=4.11.0',
        ]
    }
)