===============
Transforming Data
===============

The ``act.transform`` subpackage moves data from one coordinate grid onto another --
most often resampling a datastream onto a different time base -- while carrying
quality control information through the resampling.

Two things distinguish it from the resampling tools already in xarray:

#. **Quality control is propagated.** Input QC is honored, so samples flagged bad are
   excluded from the computation rather than silently averaged in. Output QC is then
   *generated*, recording which output points were interpolated, extrapolated, built
   from partly bad input, or otherwise deserve suspicion.
#. **Bin bounds are respected.** ``bin_average`` treats each sample as covering a cell
   with a start and an end, using CF ``bounds`` attributes when they are present, rather
   than assuming an instantaneous point measurement.

.. contents:: Contents
   :local:
   :depth: 2


When to use this instead of xarray
==================================

Use ``xarray``'s ``.resample()`` and ``.interp()`` for ordinary resampling and
interpolation. Use ``act.transform`` when one of the following is true:

* **You have a companion QC variable and the answer must respect it.** ``ds['temp'].resample(time='30min').mean()``
  averages every sample it is given. If a sensor reported an obviously wrong value and the
  file's ``qc_temp`` variable says so, that value still lands in the mean. ``act.transform``
  takes the QC variable as an argument and drops the flagged samples.

* **You need to know how trustworthy each output point is.** xarray gives you back data.
  ``act.transform`` gives you back data *and* a QC variable describing how each output
  point was produced, which you can then filter on.

* **Your samples are averages over an interval, not instantaneous points.** Many ARM
  datastreams record a value accompanied by a ``time_bounds`` variable saying which
  interval it summarizes. ``bin_average`` weights each input sample by how much of its
  cell overlaps the output cell. ``.resample()`` assigns each timestamp to exactly one
  bin, so a sample straddling a bin edge is attributed entirely to one side.

* **You want to declare when a bin had too little or too noisy coverage to trust.** The
  ``std_*`` and ``goodfrac_*`` thresholds flag output points whose input was more variable,
  or sparser, than you are willing to accept.

If none of those apply, plain xarray is usually sufficient.


Choosing a transform
====================

Three transforms are available. All three take the same first three arguments
(``data``, ``target``, ``dim``) and all three return a ``(result, result_qc)`` tuple of
:class:`xarray.DataArray` objects.

.. list-table::
   :header-rows: 1
   :widths: 18 30 34 18

   * - Transform
     - What it does
     - Use it when
     - Nearest xarray analogue
   * - :func:`~act.transform.bin_average`
     - Weighted average of every input sample overlapping each output cell.
     - Going to a **coarser** grid, and the output should summarize the interval.
     - ``.resample().mean()``
   * - :func:`~act.transform.interpolate`
     - Piecewise linear interpolation between the two bracketing samples.
     - Going to a **finer** grid, or onto offset timestamps, for a continuous quantity.
     - ``.interp()``
   * - :func:`~act.transform.subsample`
     - Selects the nearest input sample within ``t_range``.
     - The value must stay an **actual observed sample** -- codes, flags, states, and
       other quantities an average would render meaningless.
     - ``.reindex(method='nearest')``

The choice matters for more than accuracy. Bin-averaging a
present-weather *code* produces a number that means nothing; subsampling is correct
there. Subsampling a temperature onto a much coarser grid throws away most of the
record; bin-averaging is correct there.

Only ``bin_average`` accepts bounds and the coverage thresholds. Only ``interpolate``
and ``subsample`` accept ``t_range``, the maximum distance from an output point to an
input sample beyond which no value is produced (it defaults to the median input
spacing).


A first example
===============

:func:`~act.transform.make_coord` builds the target coordinate. It accepts datetime-like
endpoints with a pandas frequency string, or numeric endpoints with a numeric step:

.. code-block:: python

    import act
    from arm_test_data import DATASETS

    filename = DATASETS.fetch('gucmetM1.b1.20230301.000000.cdf')
    ds = act.io.arm.read_arm_netcdf(filename, cleanup_qc=True)

    # The file is 1-minute data; build a 30-minute target coordinate.
    target = act.transform.make_coord('2023-03-01', '2023-03-02', '30min', name='time')

    result, result_qc = act.transform.bin_average(ds['temp_mean'], target, dim='time')

``result`` is a ``DataArray`` on the new coordinate, keeping the input's ``attrs``,
``encoding``, and any coordinates that do not depend on ``dim``. ``result_qc`` is an
integer QC ``DataArray`` of the same shape.

Any dimension works, not only time, and the array may be N-dimensional -- the transform
is applied along ``dim`` and every other axis is iterated over.


Quality control, end to end
===========================

Passing input QC
----------------

Pass the companion QC variable as ``qc``. By default, ``qc_mask=None`` selects every
QC bit whose ``flag_assessments`` metadata is ``"Bad"``; samples matching those bits
are excluded from the computation. This assessment-aware default is preferred over
hardcoding a datastream-specific bit value:

.. code-block:: python

    var_name = 'tbrg_precip_total_corr'

    filtered, filtered_qc = act.transform.bin_average(
        ds[var_name], target, dim='time', qc=ds['qc_' + var_name]
    )

For the ``gucmetM1.b1`` file above, this excludes the ``fail_max`` QC test that flags
the raw data's 7999 mm spikes. You can choose a different assessment by passing its
name as ``qc_mask``:

.. code-block:: python

    # Exclude every input assessed as Indeterminate instead.
    filtered, filtered_qc = act.transform.bin_average(
        ds[var_name],
        target,
        dim='time',
        qc=ds['qc_' + var_name],
        qc_mask='Indeterminate',
    )

The three supported forms are:

* ``None`` (the default): derive a mask for ``"Bad"`` from the QC variable's
  ``flag_assessments`` and ``flag_masks`` metadata. If no QC variable is supplied,
  no QC bits are excluded.
* A string such as ``'Bad'`` or ``'Indeterminate'``: derive a mask for that exact
  assessment from the QC metadata. An explicit assessment string requires a QC
  variable with both metadata attributes.
* A list of strings such as ``['Bad', 'Suspect']``: combine the masks for all listed
  assessments. This is useful when several assessment categories should be excluded.
* An integer such as ``4``: use that bitmask directly. This is useful when working
  with a non-CF QC array or when an exact bit-level selection is required. Remember
  that ``qc_mask`` is a bitmask, not a bit number: bit 3 is ``1 << 2``, or ``4``.

The same ``qc_mask`` options apply to ``interpolate``, ``subsample``,
``transform_dataset``, and the Dataset accessor methods. The transform's output QC
still records any input QC bits that were not selected for exclusion.

Reading the output QC
---------------------

The returned QC variable is bit-packed. The bit constants live in
``act.transform.constants`` and are re-exported at package level:

.. code-block:: python

    from act.transform.constants import QC_SOME_BAD_INPUTS

    affected = (filtered_qc.values & QC_SOME_BAD_INPUTS).astype(bool)
    print(affected.sum())  # number of output bins that lost input to QC

The full set of bits, and which transforms can set them:

.. list-table::
   :header-rows: 1
   :widths: 30 8 14 48

   * - Constant
     - Value
     - Assessment
     - Meaning
   * - ``QC_BAD``
     - 1
     - Bad
     - No value could be produced. Set alongside ``QC_OUTSIDE_RANGE`` or
       ``QC_ALL_BAD_INPUTS``; it is *not* set by the threshold bits below.
   * - ``QC_INDETERMINATE``
     - 2
     - Indeterminate
     - An input sample carried a QC bit that was *not* in ``qc_mask``, so it was used
       but was not clean.
   * - ``QC_INTERPOLATE``
     - 4
     - Indeterminate
     - ``interpolate`` had to skip past a bad or missing neighbor to find a usable one.
   * - ``QC_EXTRAPOLATE``
     - 8
     - Indeterminate
     - The output point lies beyond the input samples used.
   * - ``QC_NOT_USING_CLOSEST``
     - 16
     - Indeterminate
     - ``subsample`` rejected the nearest sample (bad or missing) and took a farther one.
   * - ``QC_SOME_BAD_INPUTS``
     - 32
     - Indeterminate
     - ``bin_average`` excluded some, but not all, candidate samples from this bin.
   * - ``QC_ZERO_WEIGHT``
     - 64
     - Indeterminate
     - Total weight for this output point was zero.
   * - ``QC_OUTSIDE_RANGE``
     - 128
     - Bad
     - The output point falls outside the input coordinate range, or beyond ``t_range``.
   * - ``QC_ALL_BAD_INPUTS``
     - 256
     - Bad
     - Every candidate input sample was bad or missing; no value could be produced.
   * - ``QC_BAD_STD``
     - 512
     - Bad
     - Within-bin standard deviation exceeded ``std_bad_max``.
   * - ``QC_INDETERMINATE_STD``
     - 1024
     - Indeterminate
     - Within-bin standard deviation exceeded ``std_ind_max``.
   * - ``QC_BAD_GOODFRAC``
     - 2048
     - Bad
     - Good coverage fraction fell below ``goodfrac_bad_min``.
   * - ``QC_INDETERMINATE_GOODFRAC``
     - 4096
     - Indeterminate
     - Good coverage fraction fell below ``goodfrac_ind_min``.

Note the distinction between ``qc_mask`` and ``QC_INDETERMINATE``: bits you name in
``qc_mask`` cause a sample to be *excluded*, while any other bit set on a used sample
raises ``QC_INDETERMINATE`` on the output. Input QC you did not ask to be treated as bad
is therefore not discarded -- it is reported.

Composing with ``ds.qcfilter``
------------------------------

Every output QC variable is stamped with ARM-style ``flag_masks``,
``flag_meanings``, ``flag_assessments``, ``flag_comments``, and
``standard_name='quality_flag'`` by
:func:`act.transform.constants.add_arm_qc_attrs`. That means ARM metadata and
ACT's existing QC machinery work on transform output without any translation step:

.. code-block:: python

    import xarray as xr

    out_ds = xr.Dataset({filtered.name: filtered, filtered_qc.name: filtered_qc})

    # Mask output points that lost input to QC.
    masked = out_ds.qcfilter.get_masked_data(
        var_name, rm_assessments=['Bad', 'Indeterminate']
    )

``qcfilter`` finds the QC variable through the data variable's ``ancillary_variables``
attribute. ARM files already set it, and the transform copies the input's ``attrs``
onto the output, so this normally works without extra setup; if your input lacks the
attribute, set ``out_ds[var_name].attrs['ancillary_variables'] = filtered_qc.name``
first. Note also that most transform bits are assessed ``Indeterminate`` rather than
``Bad``, so filtering only ``rm_assessments=['Bad']`` will not remove points flagged
``QC_SOME_BAD_INPUTS``.


Bounds awareness and coverage thresholds
========================================

``bin_average`` works on cells rather than points. Each input sample and each output bin
has a start and an end, and an input sample contributes to an output bin in proportion to
how much of it overlaps.

Bounds come from three places, in order of preference:

#. **Explicitly**, via ``input_bounds`` and ``output_bounds`` -- each an ``(n, 2)`` array
   of ``[start, end]`` pairs.
#. **From CF metadata.** :func:`~act.transform.transform_dataset` reads the coordinate's
   ``bounds`` attribute and picks up the named variable automatically.
#. **Inferred from midpoints.** With no bounds available, each cell runs from the midpoint
   to its previous neighbor to the midpoint to its next, with the outermost edges
   extrapolated.

.. code-block:: python

    # ARM met files carry a time_bounds variable; use it explicitly.
    result, result_qc = act.transform.bin_average(
        ds['temp_mean'], target, dim='time', input_bounds=ds['time_bounds'].values
    )

Two sets of thresholds let a caller declare when a bin is not trustworthy. Both compare
against quantities ``bin_average`` computes per output bin, and both have an
"indeterminate" and a harder "bad" level:

* ``std_ind_max`` / ``std_bad_max`` -- the weighted standard deviation of the input
  samples in the bin. High values mean the bin is averaging over real variability, so the
  single output number is a poor summary.
* ``goodfrac_ind_min`` / ``goodfrac_bad_min`` -- the fraction of the bin's span covered by
  *good* input. Low values mean the average rests on only part of the interval, because
  input was missing or QC-excluded.

.. code-block:: python

    from act.transform.constants import QC_BAD_STD, QC_INDETERMINATE_STD

    # Wind speed over 30-minute bins: flag gusty bins, reject the worst.
    result, result_qc = act.transform.bin_average(
        ds['wspd_arith_mean'], target, dim='time',
        std_ind_max=0.8, std_bad_max=1.5,
    )

    print((result_qc.values & QC_INDETERMINATE_STD).astype(bool).sum())
    print((result_qc.values & QC_BAD_STD).astype(bool).sum())

Both default to a no-op -- the ``std_*`` thresholds to infinity and the ``goodfrac_*``
thresholds to zero -- so the bits only appear once you set a threshold. The right values
are a property of your instrument and your tolerance, not something the library can
guess.


Transforming a whole Dataset
============================

:func:`~act.transform.transform_dataset` applies a transform to every variable in a
Dataset that has ``dim``:

.. code-block:: python

    new_ds = act.transform.transform_dataset(
        ds, target=target, dim='time', transform='bin_average'
    )

``transform_dataset`` handles QC pairing, bounds lookup, and pass-through variables:

* **QC auto-pairing.** For each variable ``foo``, if ``qc_foo`` exists in the Dataset it is
  passed as that variable's ``qc`` automatically. The prefix is configurable with
  ``qc_prefix``, which defaults to ``'qc_'``. QC variables are not themselves transformed
  as data; each appears in the output as the transformed variable's companion.
* **CF bounds auto-detection.** For ``bin_average``, the coordinate's ``bounds`` attribute
  is followed and the named variable used as ``input_bounds``. The bounds variable itself
  is skipped rather than transformed, since it describes coordinate cells rather than
  measured data.
* **Pass-through.** Variables without ``dim`` are copied across unchanged, and the
  Dataset's global ``attrs`` are preserved.

Because one transform is rarely right for every variable in a file, per-variable
overrides are supported:

.. code-block:: python

    new_ds = act.transform.transform_dataset(
        ds,
        target=target,
        dim='time',
        transform='bin_average',          # default for most variables
        # qc_mask=None (the default) excludes QC bits assessed as Bad.
        per_var_transform={
            # Averaging a present-weather code is meaningless; take a real sample.
            'pwd_pw_code_inst': 'subsample',
        },
        per_var_kwargs={
            'wspd_arith_mean': {'std_ind_max': 0.8},
        },
    )

You can also give ``target_ds=`` another Dataset instead of ``target=`` to transform onto
an existing file's coordinate -- convenient for putting two datastreams on a common time
base. The two arguments are mutually exclusive.


Functions or accessor
=====================

Each transform has a function form and a Dataset-accessor form. Functions accept
``DataArray`` objects; accessors accept variable names and delegate to the same
implementation.

.. code-block:: python

    # Function form -- takes DataArrays.
    result, result_qc = act.transform.bin_average(
        ds['temp_mean'], target, dim='time', qc=ds['qc_temp_mean']
    )

    # Accessor form -- takes names. Identical result.
    result, result_qc = ds.transform.bin_average(
        'temp_mean', target, dim='time', qc_var_name='qc_temp_mean'
    )

Note the argument name changes with the form: the functions take ``qc=<DataArray>``,
while the accessor takes ``qc_var_name=<str>``. ``ds.transform.transform_dataset(...)``
works the same way, taking every argument except ``ds`` itself.


Datetime coordinates
====================

Datetime coordinates are handled directly; there is no need to convert to numeric time
first. ``datetime64`` coordinates, and the ``cftime`` objects that
:func:`act.io.arm.read_arm_netcdf` produces for CF ``bounds`` variables, are all
normalized to nanoseconds internally, so values from coordinates and bounds recorded in
different units stay mutually comparable. The output coordinate is returned as a datetime,
not as a number.

For ``interpolate`` and ``subsample``, ``t_range`` may be given as a timedelta when the
coordinate is a datetime:

.. code-block:: python

    import numpy as np

    result, result_qc = act.transform.subsample(
        ds['temp_mean'], target, dim='time', t_range=np.timedelta64(5, 'm')
    )

Numeric coordinates continue to use the same behavior.


The numba dependency
====================

The numeric kernels are JIT-compiled with `numba <https://numba.pydata.org/>`_. If numba
cannot be imported, cannot compile a kernel, or raises an error while running a kernel,
that kernel falls back to a pure-Python implementation of the same logic
(see ``act/transform/_numba_support.py``) and emits a ``RuntimeWarning`` once for each failing kernel type, naming the reason:

.. code-block:: text

    RuntimeWarning: act.transform: numba JIT unavailable or failed for the
    'bin_average' kernel (...); falling back to a slower pure-Python
    implementation. Results are unaffected.

The fallback uses the same Python kernel as the JIT path, so it preserves results but
may run more slowly.

Further reading
===============

* :ref:`transform_examples` -- runnable gallery examples.
* :mod:`act.transform` -- the generated API reference.
