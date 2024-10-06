from setuptools import setup, find_packages

setup(
    name='slm_project_package',
    version='0.1.0',
    description='Package for the SLM project',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'azureml-core>=1.37.0',
        'azureml-pipeline-core>=1.37.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'transformers',
        'torch',
        # Add other dependencies required by your code
    ],
    python_requires='>=3.10',
)