from setuptools import setup, find_packages

setup(
    name='azureml-sfta',
    version='0.1.0',
    author='SFTA team',
    description='Package for the SFTA',
    packages=find_packages(where='./'),
    package_dir={'': './'},
)
