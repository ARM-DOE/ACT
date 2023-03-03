import glob
from os import path
from setuptools import setup, find_packages
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
act does not support Python {}.{}.
Python {}.{} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *sys.version_info[:2], *min_version
    )
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line for line in requirements_file.read().splitlines() if not line.startswith('#')
    ]


setup(
    name='act-atmos',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Package for working with atmospheric time series datasets',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Adam Theisen',
    author_email='atheisen@anl.gov',
    url='https://github.com/ARM-DOE/ACT',
    packages=find_packages(exclude=['docs']),
    entry_points={'console_scripts': []},
    include_package_data=True,
    package_data={'act': []},
    scripts=glob.glob("scripts/*"),
    install_requires=requirements,
    license='BSD (3-clause)',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
