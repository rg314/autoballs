from setuptools import setup, find_packages


VERSION = '0.0.0'

DESCRIPTION = 'detection and analysis of eyeballs'

CLASSIFIERS = [
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python',
]

REQUIREMENTS = ['codecov==2.1.11']



PACKAGES = [
    'autoballs',
]

SETUP_REQUIRES = ('pytest-cov', 'pytest-runner','pytest', 'codecov')
TESTS_REQUIRES = ('pytest-cov','codecov')


options = {
    'name': 'autoballs',
    'version': VERSION,
    'author': 'Ryan Greenhalgh',
    'author_email': 'ryan.greenhalgh@hotmail.co.uk, ',
    'description': DESCRIPTION,
    'classifiers': CLASSIFIERS,
    'packages': PACKAGES,
    'install_requires': REQUIREMENTS,
    # 'package_data': DATA,
    'setup_requires': SETUP_REQUIRES,
    'test_requires': TESTS_REQUIRES
}


setup(**options)
