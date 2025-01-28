from setuptools import setup, find_packages

setup(
    name="meet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'msgpack',
        'numpy',
        'opencv-python',
        'cryptography',
        'playwright',
        'pytest-asyncio',
        'PyJWT'
    ],
) 