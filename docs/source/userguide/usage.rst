=====
Usage
=====

Start by importing Atmospheric data Community Toolkit.

.. code-block:: python

    import act

The Atmospheric data Community Toolkit comes with modules for loading ARM datasets.
The main dataset object that is used in ACT is based off of an extension of
the `xarray.Dataset<http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`
object. In particular ACT adds a DatasetAccessor that stores the additional
properties required by act in the .act property of a Dataset. For example,
if we want to access the name of the datastream, we simply do:

.. code-block:: python

    import act

    the_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_SONDE_WILDCARD)
    print(the_ds.act.datastream)

To load ARM-standard files into, the ``arm.io.arm.read_arm_netcdf`` routine is used.
This takes in a string with wildcards allowed or a list of files for ACT to read.
Currently, there is support for ACT to concatenate multiple netCDF files along a ``time``
dimension if all of the files follow the same format. This allows for the easy
reading of multi-file datasets, such as the examples provided in the
:ref:`sphx_glr_source_auto_examples_plot_sonde.py`.

In addition, ACT has a TimeSeriesDisplay object that makes plotting the data
in a timeseries easy. The TimeSeriesDisplay object supports multipanel plots
with ease. The following code will plot a 3 panel time series plot from
the dataset in the code snippet above.

.. code-block:: python

    display = act.plotting.TimeSeriesDisplay(met)
    display.add_subplots((3,), figsize=(15, 10))
    display.plot("alt", subplot_index=(0,))
    display.plot("temp_mean", subplot_index=(1,))
    display.plot("rh_mean", subplot_index=(2,))
    plt.show()

In addition, the figure and axes handles of each subplot are stored in the
`TimeSeriesDisplay` object as `TimeSeriesDisplay.fig` and
`TimeSeriesDisplay.ax`. Therefore, standard matplotlib routines can then
be used to modify the properties of each plot if the user desires further
customization.

Finally, ACT is able to download data from the ARM archive given that a
user's username and token are provided.

.. code-block:: python

    act.discovery.get_data(
        "userName", "XXXXXXXXXXXXXXXX", "sgpmetE13.b1", "2017-01-14", "2017-01-20"
    )

The preceding example will download the sgpmetE13.b1 dataset in netCDF
format from 2017-01-14 to 2017-01-20 and store the dataset in an output
folder named 'sgpmetE13.b1.' This output folder can also be specified
by the user.
