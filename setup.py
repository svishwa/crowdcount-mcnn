from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = open('requirements.txt').readlines()

setup(
    name='mcnn',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=find_packages(),
    description='Single Image Crowd Counting via Multi Column Convolutional Neural Network',
)
