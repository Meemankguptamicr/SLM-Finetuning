from setuptools import setup, find_packages

setup(
    name='common_utils',
    version='0.1.0',
    author='SFTA team',
    description='Package for the SLM project utilities',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)