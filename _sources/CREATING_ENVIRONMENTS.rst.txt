Setting up an Environment
=========================


Anaconda
~~~~~~~~

Creating environments using Anaconda is recommended due to the ability to
create more than one environment. It is also recommended because you can
keep dependencies separate from one another that might conflict if you had
them all in your root environment. For example, if you had all the dependencies
for a Pandas environment and all the dependencies for a Cartopy environment in
your root environment, there might be conflicts between channels and packages.
So Anaconda allows you to create multiple environments to avoid these issues.

To download and install `Anaconda <https://www.anaconda.com/download/#>`_.

While Anaconda is downloading, it will ask if you want to set a path to it, or
let Anaconda set a default path. After choosing, Anaconda should finish
downloading. After it is done, exit the terminal and open a new one to make
sure the environment path is set. If conda command is not found, there is help
on running conda and fixing the environment path, found here:

* `How to Run Conda <https://stackoverflow.com/questions/18675907/how-to-run-conda>`_

Setting a Channel
~~~~~~~~~~~~~~~~~

Anaconda has a cloud that stores many of its packages. It is recommended, at
times, to use the conda-forge channel instead. Conda-Forge is a community led
collection of packages, and typically contains the most recent versions of the
packages required for ACT. Having packages in an environment, within the same
channel, helps avoid conflict issues. To add conda-forge as the priority
channel, simply do::

        conda config --add channels conda-forge

You can also just flag the channel when conda install packages such as::

        conda install -c conda-forge numpy

More on managing channels can be found here:

* `Managing Channels <https://conda.io/docs/user-guide/tasks/manage-channels.html>`_

Creating an Environment
~~~~~~~~~~~~~~~~~~~~~~~

There are a few ways to create a conda environment for using ACT or other
packages. One way is to use the environment file, found here:

* https://github.com/ARM-DOE/ACT/blob/master/environment.yml

To create an environment using this file, use the command::

        conda env create -f environment.yml

This will then create an environment called act_env that can be activated
by::

        source activate act_env

or deactivated after use::

        source deactivate act_env

Once the environment is created and activated, you can install more packages
into the environment by simply conda installing them. An example of this is,
if you want Jupyter Notebook to run in that enviroment with those packages,
do this step while the environment is activate::

        conda install -c conda-forge jupyter notebook

Another way to create a conda environment is by doing it from scratch using
the conda create command. An example of this::

        conda create -n act_env -c conda-forge python=3.7 numpy pandas astral
        scipy matplotlib dask xarray

After activating the environment with::

        source activate act_env

Go into the ACT directory and run::

        python setup.py install

or if in development mode::

        pip install -e .

This will also create an environment called act_env that can be activated the
same way, as mentioned above with the environment.yml file. To then run your
coding editor within the environment, run in the command line::

        python

or::

        ipython

or::

        jupyter notebook

or even::

        spyder

depending on what you installed in your environment and want to use for coding.

More Information
~~~~~~~~~~~~~~~~

For more an conda and help with conda:

* https://conda.io/docs/
* https://gitter.im/conda/conda
