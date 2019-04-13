===================================
Atmospheric Community Toolkit (ACT)
===================================

.. image:: https://img.shields.io/travis/ANL-DIGR/ACT.svg
        :target: https://travis-ci.org/ANL-DIGR/ACT

Package for connecting Atmospheric Radation Measurement (ARM) users to the
data. Has the ability to download, read, and visualize multi-file datasets from
ARM datastreams. Currently, multi-panel timeseries plots are supported. For
more on ARM and ARM datastreams:

* ARM: https://www.arm.gov/
* ARM Data Discovery: https://www.archive.arm.gov/discovery/

* Free software: 3-clause BSD license

Important Links
~~~~~~~~~~~~~~~

* Documentation: https://anl-digr.github.io/ACT/
* Examples: https://anl-digr.github.io/ACT/source/auto_examples/index.html
* Issue Tracker: https://github.com/ANL-DIGR/ACT/issues

Dependencies
~~~~~~~~~~~~

* `NumPy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `xarray <https://xarray.pydata.org/en/stable/>`_
* `astral <https://astral.readthedocs.io/en/latest/>`_
* `pandas <https://pandas.pydata.org/>`_
* `dask <https://dask.org/>`_

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

Installing ACT from source is the only way to get the latest updates and
enhancement to the software that have no yet made it into a release.
The latest source code for ACT can be obtained from the GitHub repository,
https://github.com/ANL-DIGR/ACT. Either download and unpack the
`zip file <https://github.com/ANL-DIGR/ACT/archive/master.zip>`_ of
the source code or use git to checkout the repository::

    git clone https://github.com/ANL-DIGR/ACT.git

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
 
    git clone https://github.com/ANL-DIGR/ACT.git

If you are planning on making changes that you would like included in ACT,
forking the repository is highly recommended.

We welcome contributions for all uses of ACT, provided the code can be
distributed under the BSD 3-clause license. A copy of this license is
available in the **LICENSE.txt** file in this directory. For more on
contributing, see the `contributor's guide. <https://github.com/ANL-DIGR/ACT/blob/master/CONTRIBUTING.rst>`_

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory (you will need to have pytest installed)::

   $ pytest --pyargs act

In-place installs can be tested using the `pytest` command from within
the source directory.
