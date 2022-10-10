# Imports
from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
    
# Setup
setup(
    # Basic info
    name='bayesian-tensorflow',
    version='1.0.0',
    # Descriptions
    description='Bayesian Neural Networks for TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    # Author info
    author='Jim Beckers',
    author_email='jbeckers@gnhearing.com',
    # Classifiers
    classifiers = [
        'Development Status :: 3 - Alpha'
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # Packages
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires='>=3.7',
    # install_requires=[
    #     'tensorflow>=2.9.0',
    # ],
)