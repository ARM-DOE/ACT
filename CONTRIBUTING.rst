============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ARM-DOE/ACT/issues

If you are reporting a bug, please include:

* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Atmospheric data Community Toolkit could always use more documentation, whether
as part of the official Atmospheric data Community Toolkit docs, in docstrings,
or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ARM-DOE/ACT/issues

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's are a few steps we will go over for contributing
to `act`.

1. Fork the `arm-community-toolkit` repo on GitHub and clone your fork locally.

2. Install your local copy into an Anaconda environment. Assuming you have
   anaconda installed.

3. Create a branch for local development.

4. Create or modified code so that it produces doc string and follows standards.

5. Install your `pre-commit <https://pre-commit.com>` hooks, by using `pre-commit install`

6. Set up environment variables (Optional)

7. Local unit testing using Pytest.

8. Commit your changes and push your branch to GitHub and submit a pull
   request through the GitHub website.

Fork and Cloning the ACT Repository
-----------------------------------
To start, you will first fork the `arm-community-toolkit` repo on GitHub by
clicking the fork icon button found on the main page here:

- https://github.com/ARM-DOE/ACT

After your fork is created, git clone your fork. I would not clone the main
repository link unless your just using the package as an install and not
development. The main master is used as a remote for upstream for grabbing
changes as others contribute.

To clone and set up an upstream, use::

    git clone https://github.com/yourusername/ACT.git

or if you have ssh key setup::

    git clone git@github.com:yourusername/ACT.git

After that, from within the ACT directory, do::

    git remote add upstream https://github.com/ARM-DOE/ACT.git

Install
-------

The easiest method for using ACT and is dependencies is by using:
`Anaconda <https://www.anaconda.com/download/#>`_ or
`Miniconda <https://conda.io/miniconda.html>`_.
From within the ACT directory, you can use::

    pip install -e .

This downloads ACT in development mode. Do this preferably in a conda
environment. For more on Anaconda and environments:

- https://arm-doe.github.io/ACT/CREATING_ENVIRONMENTS.html

Working with Git Branches
-------------------------

When contributing to ACT, the changes created should be in a new branch
under your forked repository. Let's say the user is adding a new plot display.
Instead of creating that new function in your master branch. Create a new
branch called ‘wind_rose_plot’. If everything checks out and the admin
accepts the pull request, you can then merge the master branch and
wind_rose_plot branch.

To delete a branch both locally and remotely, if done with it::

                git push origin --delete <branch_name>
                git branch -d <branch_name>

or in this case::

                git push origin --delete wind_rose_plot
                git branch -d wind_rose_plot


To create a new branch::

                git checkout -b <branch_name>

If you have a branch with changes that have not been added to a pull request
but you would like to start a new branch with a different task in mind. It
is recommended that your new branch is based on your master. First::

                git checkout master

Then::

                git checkout -b <branch_name>

This way, your new branch is not a combination of your other task branch and
the new task branch, but is based on the original master branch.

Typing `git status` will not only inform the user of what files have been
modified and untracked, it will also inform the user of which branch they
are currently on.

To switch between branches, simply type::

                git checkout <branch_name>

Python File Setup
-----------------

When adding a new function to ACT, add the function in the __init__.py
for the submodule so it can be included in the documentation.

Following the introduction code, modules are then added. To follow pep8
standards, modules should be added in the order of:

        1. Standard library imports.
        2. Related third party imports.
        3. Local application/library specific imports.

For example:

.. code-block:: python

    import glob
    import os

    import numpy as np
    import numpy.ma as ma

    from .dataset import ACTAccessor

Following the main function def line, but before the code within it, a doc
string is needed to explain arguments, returns, references if needed, and
other helpful information. These documentation standards follow the NumPy
documentation style.

For more on the NumPy documentation style:

- https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

An example:

.. code-block:: python

    def read_arm_netcdf(filenames, variables=None):

        """
        Returns `xarray.Dataset` with stored data and metadata from a
        user-defined query of standard netCDF files from a single
        datastream.

        Parameters
        ----------
        filenames : str or list
            Name of file(s) to read
        variables : list, optional
            List of variable name(s) to read

        Returns
        -------
        act_obj : Object
            ACT dataset

        Examples
        --------
        This example will load the example sounding data used for unit
        testing.

        .. code-block:: python

            import act

            the_ds, the_flag = act.io.arm.read_arm_netcdf(
                act.tests.sample_files.EXAMPLE_SONDE_WILDCARD)
            print(the_ds.act.datastream)
        """

As seen, each argument has what type of object it is, an explanation of
what it is, mention of units, and if an argument has a default value, a
statement of what that default value is and why.

Private or smaller functions and classes can have a single line explanation.

An example:

.. code-block:: python

    def _get_value(self):
        """Gets a value that is used in a public function."""

Code Style
----------

Py-ART uses pre-commit for linting, which applies a variety of pep8 and other
code style rules.

For more on pep8 style:

- https://www.python.org/dev/peps/pep-0008/

To install pre-commit hooks for the Py-ART repo::

        pre-commit install

Afterwards, pre-commit will run on every commit to the repository. It will
re-format files as neccessary.

Naming Convention
-----------------

Discovery
~~~~~~~~~
When adding discovery modules or functions please adhere to the following
* Filenames should just include the name of the organization (arm) or portal (airnow) and no other filler words like get or download
* Functions should follow [get/download]_[org/portal]_[data/other description].  If it is getting data but not downloading a file, it should start with get, like get_asos_data. If it downloads a file, it should start with download.  The other description can vary depending on what you are retrieving.  Please check out the existing functions for ideas.

IO
~~
Similarly, for the io modules, the names should not have filler and just be the organization or portal name.  The functions should clearly indicate what it is doing like read_arm_netcdf instead of read_netcdf if the function is specific to ARM files.

Adding Secrets and Environment Variables
----------------------------------------
In some cases, unit tests (as noted in the next section), need some username/password/token information
and that is not something that is good to make public.  For these instances, it is recommended that users
set up environment variables for testing.  The following environment variables should be set on the user's
local machine using the user's own credentials for all tests to run properly.

Atmospheric Radiation Measurement User Facility - https://adc.arm.gov/armlive/

    ARM_USERNAME

    ARM_PASSWORD

Environmental Protection Agency AirNow - https://docs.airnowapi.org/

    AIRNOW_API

If adding tests that require new environment variables to be set, please reach out to the ACT development
team through the pull request.  The ACT development team will need to do the following to ensure it works
properly when merged in.  Note, due to security purposes these secrets are not available to the actions in
a pull request but will be available once merged it.

1.) Add a GitHub Secret to ACT settings that's the same as that in the test file

2.) Add this name to the "env" area of the GitHub Workflow yml files in .github/workflows/*

3.) If the amount of code will impact the decrease in coverage during testing, update the threshold in coveralls

4.) Upon merge, this should automatically pull in the secrets for the testing but there have been quirks.
Ensure that tests run properly


Unit Testing
------------

When adding a new function to ACT it is important to add your function to
the __init__.py file under the corresponding ACT folder.

Create a test for your function and have assert from numpy testing test the
known values to the calculated values. If changes are made in the future to
ACT, pytest will use the test created to see if the function is still valid
and produces the same values. It works that, it takes known values that are
obtained from the function, and when pytest is ran, it takes the test
function and reruns the function and compares the results to the original.

An example:

.. code-block:: python

    import act
    import numpy as np
    import xarray as xr


    def test_correct_ceil():
        # Make a fake dataset to test with, just an array with 1e-7
        # for half of it.
        fake_data = 10 * np.ones((300, 20))
        fake_data[:, 10:] = -1
        arm_obj = {}
        arm_obj["backscatter"] = xr.DataArray(fake_data)
        arm_obj = act.corrections.ceil.correct_ceil(arm_obj)
        assert np.all(arm_obj["backscatter"].data[:, 10:] == -7)
        assert np.all(arm_obj["backscatter"].data[:, 1:10] == 1)

Pytest is used to run unit tests in ACT.

It is recommended to install ACT in “editable” mode for pytest testing.
From within the main ACT directory::

        pip install -e .

This lets you change your source code and rerun tests at will.

To install pytest::

        conda install -c conda-forge pytest

To run all tests in pyart with pytest from outside the pyart directory::

        pytest --pyargs act

All test with increase verbosity::

        pytest -v

Just one file::

        pytest filename

Note: When an example shows filename as such::

        pytest filename

filename is the filename and location, such as::

        pytest /home/user/act/act/tests/test_correct.py

Relative paths can also be used::

        cd ACT
        pytest ./act/tests/test_correct.py

For more on pytest:

- https://docs.pytest.org/en/latest/

Note: When testing ACT, the unit tests will download files from different
datastreams as part of the tests. These files will download to the directory
from where the tests were ran. These files will need to be added to the
.gitignore if they are in a location that isn't caught by the .gitignore.
More on using git can be seen below.


Adding Changes to GitHub
------------------------

Once your done updating a file, and want the changes on your remote branch.
Simply add it by using::

        git add <file_name.py>

When commiting to GitHub, start the statement with a acronym such as
‘ADD:’ depending on what your commiting, could be ‘MAINT:’ or
‘BUG:’ or more. Then following should be a short statement such as
“ADD: Adding new wind rose display.”, but after the short statement, before
finishing the quotations, hit enter and in your terminal you can then type
a more in depth description on what your commiting.

A set of recommended acronymns can be found at:

- https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html

If you would like to type your commit in the terminal and skip the default
editor::

	git commit -m "STY: Removing whitespace from plot.py pep8."

To use the default editor(in Linux, usually VIM), simply type::

	git commit

One thing to keep in mind is before doing a pull request, update your
branches with the original upstream repository.

This could be done by::

	git fetch upstream

After fetching, a git merge is needed to pull in the changes.

This is done by::

        git merge upstream/master

To prevent a merge commit::

        git merge --ff-only upstream/master

or a rebase can be done with::

        git pull --rebase upsteam master

Rebase will take commits you missed and stack your changes on top of them.

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.7, 3.6, 3.7 for PyPy. Check
   https://travis-ci.org/ARM-DOE/ACT
   and make sure that the tests pass for all supported Python versions.

After creating a pull request through GitHub, and outside checker TravisCI
will determine if the code past all checks. If the code fails the tests, as
the pull request sits, make changes to fix the code and when pushed to GitHub,
the pull request will automatically update and TravisCI will automatically
rerun.

For more on Git:

- https://git-scm.com/book/en/v2
