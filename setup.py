"""nccd

https://github.com/papamarkou/nccd
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nccd',
    version='0.0.1',
    description='Nuclear canister corrosion detector',
    long_description=long_description,
    url='https://github.com/papamarkou/nccd',
    author='Theodore Papamarkou, Hayley Guy, Bryce Kroencke, Jordan Miller, Preston Robinette, Daniel Schultz',
    author_email='papamarkout@ornl.gov',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['deep learning', 'convolutional neural networks', 'resnets', 'nuclear canisters', 'corrosion detection'],
    package_dir={'nccd': 'nccd'},
    install_requires=['fastai', 'numpy', 'scikit-learn', 'torch',]
)
