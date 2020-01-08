===================================
Atmospheric Community Toolkit (ACT)
===================================

|AnacondaCloud| |CondaDownloads|

|Travis|

.. |AnacondaCloud| image:: https://anaconda.org/conda-forge/act-atmos/badges/version.svg
    :target: https://anaconda.org/conda-forge/act-atmos

.. |CondaDownloads| image:: https://anaconda.org/conda-forge/act-atmos/badges/downloads.svg
    :target: https://anaconda.org/conda-forge/act-atmos/files

.. |Travis| image:: https://img.shields.io/travis/ARM-DOE/ACT.svg
        :target: https://travis-ci.org/ARM-DOE/ACT

Python toolkit for working with atmospheric time-series datasets of varying dimensions. The toolkit
is meant to have functions for every part of the scientific process; discovery, IO,
quality control, corrections, retrievals, visualization, and analysis. Initial efforts were
heavily focused on the static visualization aspect of the process, but future efforts will look to
build up the other areas of interest include discovery, corrections, retirevals, and
interactive plots.

* Free software: 3-clause BSD license

Important Links
~~~~~~~~~~~~~~~

* Documentation: https://arm-doe.github.io/ACT/
* Examples: https://arm-doe.github.io/ACT/source/auto_examples/index.html
* Issue Tracker: https://github.com/ARM-DOE/ACT/issues

Dependencies
~~~~~~~~~~~~

* `NumPy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `xarray <https://xarray.pydata.org/en/stable/>`_
* `astral <https://astral.readthedocs.io/en/latest/>`_
* `pandas <https://pandas.pydata.org/>`_
* `dask <https://dask.org/>`_
* `Pint <https://pint.readthedocs.io/en/0.9/>`_
* `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_
* `Boto3 <https://aws.amazon.com/sdk-for-python/>`_
* `PyProj <https://pyproj4.github.io/pyproj/stable/>`_

Installation
~~~~~~~~~~~~

ACT can be installed a few different ways. One way is to install using pip.
When installing with pip, the ACT dependencies found in
`requirements.txt <https://github.com/ARM-DOE/ACT/blob/master/requirements.txt>`_ will also be installed. To install using pip::

    pip install act-atmos

The easiest method for installing ACT is to use the conda packages from
the latest release. To do this you must download and install 
`Anaconda <https://www.anaconda.com/download/#>`_ or 
`Miniconda <https://conda.io/miniconda.html>`_.
With Anaconda or Miniconda install, it is recommended to create a new conda
environment when using ACT or even other packages. To create a new
environment based on the `environment.yml <https://github.com/ARM-DOE/ACT/blob/master/environment.yml>`_::

    conda env create -f environment.yml

Or for a basic environment and downloading optional dependencies as needed::

    conda create -n act_env -c conda-forge python=3.7 act-atmos

Basic command in a terminal or command prompt to install the latest version of
ACT::

    conda install -c conda-forge act-atmos

To update an older version of ACT to the latest release use::

    conda update -c conda-forge act-atmos

If you do not wish to use Anaconda or Miniconda as a Python environment or want
to use the latest, unreleased version of ACT see the section below on 
**Installing from source**.

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

Installing ACT from source is the only way to get the latest updates and
enhancement to the software that have no yet made it into a release.
The latest source code for ACT can be obtained from the GitHub repository,
https://github.com/ARM-DOE/ACT. Either download and unpack the
`zip file <https://github.com/ARM-DOE/ACT/archive/master.zip>`_ of
the source code or use git to checkout the repository::

    git clone https://github.com/ARM-DOE/ACT.git

To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

Contributing
~~~~~~~~~~~~

ACT is an open source, community software project. Contributions to the
package are welcomed from all users.

The latest source code can be obtained with the command::
 
    git clone https://github.com/ARM-DOE/ACT.git

If you are planning on making changes that you would like included in ACT,
forking the repository is highly recommended.

We welcome contributions for all uses of ACT, provided the code can be
distributed under the BSD 3-clause license. A copy of this license is
available in the **LICENSE.txt** file in this directory. For more on
contributing, see the `contributor's guide. <https://github.com/ARM-DOE/ACT/blob/master/CONTRIBUTING.rst>`_

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory (you will need to have pytest installed)::

   $ pytest --pyargs act

In-place installs can be tested using the `pytest` command from within
the source directory.
