========================================
Atmospheric data Community Toolkit (ACT)
========================================

|AnacondaCloud| |Travis| |Coveralls| 

|CondaDownloads| |Zenodo| |ARM|

.. |AnacondaCloud| image:: https://anaconda.org/conda-forge/act-atmos/badges/version.svg
    :target: https://anaconda.org/conda-forge/act-atmos

.. |CondaDownloads| image:: https://anaconda.org/conda-forge/act-atmos/badges/downloads.svg
    :target: https://anaconda.org/conda-forge/act-atmos/files

.. |Travis| image:: https://www.travis-ci.com/ARM-DOE/ACT.svg?branch=master
    :target: https://travis-ci.com/ARM-DOE/ACT

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3855537.svg
    :target: https://doi.org/10.5281/zenodo.3855537

.. |Coveralls| image:: https://coveralls.io/repos/github/ARM-DOE/ACT/badge.svg
    :target: https://coveralls.io/github/ARM-DOE/ACT

.. |ARM| image:: https://img.shields.io/badge/Sponsor-ARM-blue.svg?colorA=00c1de&colorB=00539c
    :target: https://www.arm.gov/


The Atmospheric data Community Toolkit (ACT) is an open source Python toolkit for working with atmospheric time-series datasets of varying dimensions.  The toolkit has functions for every part of the scientific process; discovery, IO, quality control, corrections, retrievals, visualization, and analysis.   It is a community platform for sharing code with the goal of reducing duplication of effort and better connecting the science community with programs such as the `Atmospheric Radiation Measurement (ARM) User Facility <http://www.arm.gov>`_.  Overarching development goals will be updated on a regular basis as part of the `Roadmap <https://github.com/AdamTheisen/ACT/blob/master/guides/ACT_Roadmap.pdf>`_  .

|act|

.. |act| image:: ./docs/source/act_plots.png

Please report any issues or feature requests by sumitting an `Issue <https://github.com/ARM-DOE/ACT/issues>`_.  Additionally, our `discussions boards <https://github.com/ARM-DOE/ACT/discussions>`_ are open for ideas, general discussions or questions, and show and tell!

Announcements
~~~~~~~~~~~~~

Following GitHub standards for a more open community and inclusiveness, ACT main branch will be renamed from master to main.

https://github.com/github/renaming
https://www.git-tower.com/learn/git/faq/git-rename-master-to-main

For those using ACT with anaconda and pip, there will be no changes . If you are using a fork of ACT with GitHub under branch settings on GitHub you can rename the branch to main.

commands to switch naming locally can be found here:
https://www.git-tower.com/learn/git/faq/git-rename-master-to-main

Important Links
~~~~~~~~~~~~~~~

* Documentation: https://arm-doe.github.io/ACT/
* Examples: https://arm-doe.github.io/ACT/source/auto_examples/index.html
* Issue Tracker: https://github.com/ARM-DOE/ACT/issues

Citing
~~~~~~

If you use ACT to prepare a publication, please cite the DOI listed in the badge above, which is updated with every version release to ensure that contributors get appropriate credit.  DOI is provided through Zenodo.

Dependencies
~~~~~~~~~~~~

* `xarray <https://xarray.pydata.org/en/stable/>`_
* `NumPy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `skyfield <https://rhodesmill.org/skyfield/>`_
* `pandas <https://pandas.pydata.org/>`_
* `dask <https://dask.org/>`_
* `Pint <https://pint.readthedocs.io/en/0.9/>`_
* `PyProj <https://pyproj4.github.io/pyproj/stable/>`_
* `Proj <https://proj.org/>`_
* `Six <https://pypi.org/project/six/>`_
* `Requests <https://2.python-requests.org/en/master/>`_

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* `MPL2NC <https://github.com/peterkuma/mpl2nc>`_ Reading binary MPL data.
* `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_  Mapping and geoplots
* `MetPy <https://unidata.github.io/MetPy/latest/index.html>`_ >= V1.0 Skew-T plotting and some stabilities indices calculations
* `scikit-posthocs <https://scikit-posthocs.readthedocs.io/en/latest/>`_ Using interquartile range or generalized Extreme Studentized Deviate quality control tests

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

   $ pytest --mpl --pyargs act

In-place installs can be tested using the `pytest` command from within
the source directory.
