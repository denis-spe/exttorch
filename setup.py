from setuptools import setup, find_packages

setup(
    name='exttorch',
    version='0.8.8',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'pandas',
    ],
    include_package_data=True
)