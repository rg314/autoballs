from setuptools import setup, find_packages


VERSION = '0.0.0'

DESCRIPTION = 'detection and analysis of eyeballs'

CLASSIFIERS = [
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python',
]

REQUIREMENTS = ['codecov==2.1.11', 'pytest==6.2.2']
with open("autoballs/requirements.txt") as f:
    for line in f.readlines():
        line = line.partition('#')[0]
        line = line.rstrip()
        if not line:
            continue

        REQUIREMENTS.append(line)

PACKAGES = [
    'autoballs',
    'autoballs.network',
]

DATA = {'autoballs': ['requirements.txt'],}

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
    'package_data': DATA,
    'setup_requires': SETUP_REQUIRES,
    'test_requires': TESTS_REQUIRES
}


setup(**options)
