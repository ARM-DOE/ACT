============
Installation
============

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

    conda create -n act_env -c conda-forge python=3.12 act-atmos

Basic command in a terminal or command prompt to install the latest version of
ACT::

    conda install -c conda-forge act-atmos

To update an older version of ACT to the latest release use::

    conda update -c conda-forge act-atmos

If you are using mamba::

    mamba install -c conda-forge act-atmos
