from setuptools import setup, find_packages

setup(
    name='slm_project_package',
    version='0.1.0',
    description='Package for the SLM project utilities',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)