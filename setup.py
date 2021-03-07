from setuptools import setup, find_packages


VERSION = '0.0.0'

DESCRIPTION = 'detection and analysis of eyeballs'

CLASSIFIERS = [
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python',
]

REQUIREMENTS = ['codecov==2.1.11', 'pytest==6.2.2', 'albumentations==0.5.2', 'attrs==20.3.0', 'certifi==2020.12.5', 'chardet==4.0.0', 'codecov==2.1.11', 'coverage==5.5', 'cycler==0.10.0', 'decorator==4.4.2', 'efficientnet-pytorch==0.6.3', 'idna==2.10', 'imageio==2.9.0', 'imgaug==0.4.0', 'imglyb==1.0.0', 'iniconfig==1.1.1', 'jgo==1.0.0', 'joblib==1.0.1', 'JPype1==1.2.1', 'kiwisolver==1.3.1', 'matplotlib==3.3.4', 'mkl-fft==1.3.0', 'mkl-random==1.1.1', 'mkl-service==2.3.0', 'munch==2.5.0', 'nd2reader==3.2.3', 'networkx==2.5', 'olefile==0.46', 'opencv-python==4.5.1.48', 'opencv-python-headless==4.5.1.48', 'packaging==20.9', 'pandas==1.2.3', 'patsy==0.5.1', 'Pillow>=6.2.0', 'PIMS==0.5', 'pluggy==0.13.1', 'pretrainedmodels==0.7.4', 'psutil==5.8.0', 'py==1.10.0', 'pyimagej==1.0.0', 'pyparsing==2.4.7', 'pytest==6.2.2', 'python-dateutil==2.8.1', 'pytz==2021.1', 'PyWavelets==1.1.1', 'PyYAML==5.4.1', 'requests==2.25.1', 'scikit-image==0.18.1', 'scikit-learn==0.24.1', 'scipy==1.6.1', 'scyjava==1.1.0', 'seaborn==0.11.1', 'segmentation-models-pytorch==0.1.3', 'Shapely==1.7.1', 'sklearn==0.0', 'slicerator==1.0.0', 'statsmodels==0.12.2', 'threadpoolctl==2.1.0', 'tifffile==2021.2.26', 'timm==0.3.2', 'toml==0.10.2', 'torch==1.8.0', 'torchaudio==0.8.0', 'torchvision==0.9.0', 'tqdm==4.59.0', 'urllib3==1.26.3', 'xarray==0.17.0', 'xmltodict==0.12.0']

PACKAGES = [
    'autoballs',
    'autoballs.network',
]


SETUP_REQUIRES = ('pytest','pytest-cov', 'pytest-runner', 'codecov')
TESTS_REQUIRES = ('pytest','pytest-cov','codecov')


options = {
    'name': 'autoballs',
    'version': VERSION,
    'author': 'Ryan Greenhalgh',
    'author_email': 'ryan.greenhalgh@hotmail.co.uk, ',
    'description': DESCRIPTION,
    'classifiers': CLASSIFIERS,
    'packages': PACKAGES,
    'install_requires': REQUIREMENTS,
    'setup_requires': SETUP_REQUIRES,
    'test_requires': TESTS_REQUIRES
}


setup(**options)
