.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_source_auto_examples_qc_example.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_source_auto_examples_qc_example.py:


===========================================================
Example for working with embedded quality control variables
===========================================================

This is an example of how to use existing or create new quality
control varibles. All the tests are located in act/qc/qctests.py
file but called under the qcfilter method.


.. code-block:: default


    from act.io.armfiles import read_netcdf
    from act.tests import EXAMPLE_IRT25m20s
    from act.qc.qcfilter import parse_bit
    import numpy as np


    # Read a data file that does not have any embedded quality control
    # variables. This data comes from the example dataset within ACT.
    # Can also read data that has existing quality control variables
    # and add, manipulate or use those variables the same.
    ds_object = read_netcdf(EXAMPLE_IRT25m20s)

    # The name of the data variable we wish to work with
    var_name = 'inst_up_long_dome_resist'

    # Since there is no embedded quality control varible one will be
    # created for us.
    # We can start with adding where the data are set to missing value.
    # First we will change the first value to NaN to simulate where
    # a missing value exist in the data file.
    data = ds_object[var_name].values
    data[0] = np.nan
    ds_object[var_name].values = data

    # Add a test for where the data are set to missing value.
    # Since a quality control variable does not exist in the file
    # one will be created as part of adding this test.
    result = ds_object.qcfilter.add_missing_value_test(var_name)

    # The returned dictionary will contain the information added to the
    # quality control varible for direct use now. Or the information
    # can be looked up later for use.
    print('\nresult =', result)

    # We can add a second test where data is less than a specified value.
    result = ds_object.qcfilter.add_less_test(var_name, 7.8)

    # Next we add a test to indicate where a value is greater than
    # or equal to a specified number. We also set the assessement
    # to a user defined word. The default assessment is "Bad".
    result = ds_object.qcfilter.add_greater_equal_test(var_name, 12,
                                                       test_assessment='Suspect')

    # We can now get the data as a numpy masked array with a mask set
    # where the third test we added (greater than or equal to) using
    # the result dictionary to get the test number created for us.
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    print('\nData type =', type(data))

    # Or we can get the masked array for all tests that use the assessment
    # set to "Bad".
    data = ds_object.qcfilter.get_masked_data(var_name, rm_assessments=['Bad'])

    # If we prefer to mask all data for both Bad or Suspect we can list
    # as many assessments as needed
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_assessments=['Suspect', 'Bad'])
    print('\ndata =', data)

    # We can convert the masked array into numpy array and choose the fill value.
    data = data.filled(fill_value=np.nan)
    print('\ndata filled with masked array fill_value =', data)

    # We can create our own test by creating an array of indexes of where
    # we want the test to be set and call the method to create our own test.
    # We can allow the method to pick the test number (next available)
    # or set the test number we wan to use. This example uses test number
    # 5 to demonstrate how not all tests need to be used in order.
    data = ds_object.qcfilter.get_masked_data(var_name)
    diff = np.diff(data)
    max_difference = 0.04
    data = np.ma.masked_greater(diff, max_difference)
    index = np.where(data.mask is True)[0]
    result = ds_object.qcfilter.add_test(
        var_name, index=index,
        test_meaning='Difference is greater than {}'.format(max_difference),
        test_assessment='Suspect',
        test_number=5)

    # If we prefer to work with numpy arrays directly we can return the
    # data array converted to a numpy array with masked values set
    # to NaN. Here we are requesting both Suspect and Bad data be masked.
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_assessments=['Suspect', 'Bad'],
                                              return_nan_array=True)
    print('\nData type =', type(data))
    print('data =', data)

    # We can see how the quality control data is stored and what assessments,
    # or test descriptions are set. Some of the tests have also added attributes to
    # store the test limit values.
    qc_varialbe = ds_object[result['qc_variable_name']]
    print('\nQC Variable =', qc_varialbe)

    # The test numbers are not the flag_masks numbers. The flag masks numbers
    # are bit-paked numbers used to store what bit is set. To see the test
    # numbers we can unpack the bits.
    print('\nmask : test')
    print('-' * 11)
    for mask in qc_varialbe.attrs['flag_masks']:
        print(mask, ' : ', parse_bit(mask))

    # We can also just use the get_masked_data() method to get data
    # the same as using ".values" method on the xarray dataset. If we don't
    # request any tests or assessments to mask the returned masked array
    # will not have any mask set. The returned value is a numpy masked array
    # where the raw numpy array is accessable with .data property.
    data = ds_object.qcfilter.get_masked_data(var_name)
    print('\nNormal numpy array data values:', data.data)
    print('Mask associated with values:', data.mask)

    # We can use the get_masked_data() method to return a masked array
    # where the test is set in the quality control varialbe, and use the
    # masked array method to see if any of the values have the test set.
    data = ds_object.qcfilter.get_masked_data(var_name, rm_tests=3)
    print('\nAt least one less than test set =', data.mask.any())
    data = ds_object.qcfilter.get_masked_data(var_name, rm_tests=4)
    print('At least one difference test set =', data.mask.any())


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_source_auto_examples_qc_example.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: qc_example.py <qc_example.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: qc_example.ipynb <qc_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
