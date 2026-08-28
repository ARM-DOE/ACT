import numpy as np
import pytest
import xarray as xr

import act
from act.io.icartt import Icartt, read_icartt, write_icartt

# A minimal but spec-legal FFI 1001 normal comments section. Includes free-form
# text ahead of the first keyword, a keyword value spanning several lines, one
# entry per dependent variable for the uncertainty and limit of detection
# keywords, and two revisions. Every one of those is allowed by ESDS-RFC-029v2
# and each broke the previous index-based reader.
SAMPLE_NCOM = [
    'Free-form note about this file.',
    'A second free-form line.',
    'PI_CONTACT_INFO: Address: Somewhere ; email: jane@example.org',
    'PLATFORM: Test Platform',
    'LOCATION: Somewhere',
    'ASSOCIATED_DATA: N/A',
    'INSTRUMENT_INFO: A thermometer',
    'DATA_INFO: reported at STP',
    'UNCERTAINTY: 0.5, 1.0',
    'ULOD_FLAG: -7777',
    'ULOD_VALUE: 100, 2000',
    'LLOD_FLAG: -8888',
    'LLOD_VALUE: -50, 0',
    'DM_CONTACT_INFO: dm@example.org',
    'PROJECT_INFO: Test project',
    'STIPULATIONS_ON_USE: None',
    'OTHER_COMMENTS: first line of comments',
    'continued without leading space',
    '  and a space-indented continuation',
    'REVISION: R1',
    'R1: second revision',
    'R0 : first revision',
    'Start_UTC,temperature,pressure',
]

SAMPLE_DATA = ['0,20.5,1013.2', '1,-9999,1012.8', '2,21.0,-8888']


def build_ict(tmp_path, name='TEST_20240315_R1.ict', ncom=None, data=None, nlhead=None):
    """Write a synthetic FFI 1001 file and return its path."""
    ncom = list(SAMPLE_NCOM if ncom is None else ncom)
    data = list(SAMPLE_DATA if data is None else data)
    nv, nscoml = 2, 0
    if nlhead is None:
        nlhead = 14 + nv + nscoml + len(ncom)
    header = [
        f'{nlhead}, 1001',
        'Doe, Jane',
        'Test Org',
        'Test Instrument',
        'TESTMISSION',
        '1, 1',
        '2024,03,15,2024,03,16',
        '1.0',
        'Start_UTC, seconds',
        str(nv),
        '1, 1',
        '-9999, -8888',
        'temperature, degC',
        'pressure, hPa',
        str(nscoml),
        str(len(ncom)),
    ]
    path = tmp_path / name
    path.write_text('\n'.join(header + ncom + data) + '\n')
    return str(path)


def test_read_icartt():
    result = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)
    assert 'pitch' in result
    assert len(result['time'].values) == 14087
    assert result['true_airspeed'].units == 'm/s'
    assert 'Revision' in result.attrs
    np.testing.assert_almost_equal(result['static_pressure'].mean(), 708.75, decimal=2)


def test_read_icartt_lazy_loader():
    # The public act.io namespace exposes the reader, writer and container.
    assert act.io.read_icartt is read_icartt
    assert act.io.write_icartt is write_icartt
    assert act.io.Icartt is Icartt


def test_read_icartt_structure():
    ds = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)
    # The independent variable is retained alongside the time coordinate.
    assert 'start_time' in ds.data_vars
    assert len(ds.data_vars) == 39
    assert ds['time'].dtype == np.dtype('datetime64[ns]')
    assert str(ds['time'].values[0]) == '2018-11-04T13:04:36.000000000'
    assert str(ds['time'].values[-1]) == '2018-11-04T16:59:22.000000000'
    # qc_flag is renamed on read.
    assert 'quality_flag' in ds.data_vars
    assert 'qc_flag' not in ds.data_vars
    # Missing values become NaN rather than the file's -9999 flag.
    assert np.isnan(ds['drift'].values).sum() == 1181
    assert np.isnan(ds['vert_wind_speed'].values).sum() == 7778


def test_read_icartt_global_attributes():
    attrs = read_icartt(act.tests.EXAMPLE_AAF_ICARTT).attrs
    assert attrs['PI'] == 'ARM Aerial Facility Team'
    assert attrs['PI_Affiliation'] == 'ARM PNNL'
    assert attrs['Mission'] == 'N/A'
    assert attrs['DateOfCollection'] == '(2018, 11, 4)'
    assert attrs['DateOfRevision'] == '(2018, 11, 4)'
    assert attrs['Data_Interval'] == '[1.0]'
    assert attrs['Independent_Var'] == 'start_time,seconds'
    assert attrs['Dependent_Var_Num'] == 38
    assert attrs['_datastream'] == 'AAFNAV'
    # Keyword values keep everything after the keyword, colons included, and are
    # not split on every colon as the previous implementation did.
    assert attrs['PI_Contact'] == 'Address: PNNL ; email: armaaf@arm.gov'
    assert attrs['Platform'] == 'Department of Energy ARM Aerial Facility Gulfstream'
    assert attrs['Revision'] == 'R0'
    assert attrs['Comments'].startswith('command_line:aafnaviwg_ingest')
    assert attrs['Revision_Comments'].startswith('created by user dsmgr')
    # Keywords with no value get the standard's N/A stand-in.
    assert attrs['Associated_Data'] == 'N/A'
    assert attrs['Instrument_Info'] == 'N/A'
    assert attrs['Project_Info'] == 'N/A'


def test_read_icartt_variable_attributes():
    ds = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)
    attrs = ds['static_pressure'].attrs
    assert attrs['units'] == 'hPa'
    assert attrs['mvc'] == -9999.0
    assert attrs['scale_factor'] == 1.0
    assert attrs['ULOD_Flag'] == '-7777'
    assert attrs['LLOD_Flag'] == '-8888'
    # This file supplies no per-variable uncertainty or LOD values.
    assert attrs['uncertainty'] == 'N/A'
    assert attrs['ULOD_Value'] == 'N/A'
    assert attrs['LLOD_Value'] == 'N/A'


def test_read_icartt_freeform_comments(tmp_path):
    # Free-form text shifts every normal comment line, which silently corrupted
    # every attribute under positional lookup.
    ds = read_icartt(build_ict(tmp_path))
    ict = Icartt.from_file(build_ict(tmp_path))
    assert ict.freeform == ['Free-form note about this file.', 'A second free-form line.']
    assert ds.attrs['PI_Contact'] == 'Address: Somewhere ; email: jane@example.org'
    assert ds.attrs['Location'] == 'Somewhere'
    assert ds.attrs['DM_Contact'] == 'dm@example.org'
    assert ds.attrs['Stipulations'] == 'None'


def test_read_icartt_multiline_keyword(tmp_path):
    ds = read_icartt(build_ict(tmp_path))
    assert ds.attrs['Comments'] == (
        'first line of comments\n'
        'continued without leading space\n'
        'and a space-indented continuation'
    )


def test_read_icartt_per_variable_values(tmp_path):
    # One entry per dependent variable, matched by name rather than by a counter
    # that used to start on the independent variable.
    ds = read_icartt(build_ict(tmp_path))
    assert ds['temperature'].attrs['uncertainty'] == '0.5'
    assert ds['pressure'].attrs['uncertainty'] == '1.0'
    assert ds['temperature'].attrs['ULOD_Value'] == '100'
    assert ds['pressure'].attrs['ULOD_Value'] == '2000'
    assert ds['temperature'].attrs['LLOD_Value'] == '-50'
    assert ds['pressure'].attrs['LLOD_Value'] == '0'
    # A single file-wide flag still applies to everything.
    assert ds['temperature'].attrs['ULOD_Flag'] == '-7777'
    assert ds['pressure'].attrs['LLOD_Flag'] == '-8888'
    # The independent variable has no uncertainty or LOD entry.
    assert ds['Start_UTC'].attrs['uncertainty'] == 'N/A'


def test_read_icartt_missing_values(tmp_path):
    # Each dependent variable uses its own missing value flag.
    ds = read_icartt(build_ict(tmp_path))
    np.testing.assert_array_equal(ds['temperature'].values, [20.5, np.nan, 21.0])
    np.testing.assert_array_equal(ds['pressure'].values, [1013.2, 1012.8, np.nan])


def test_read_icartt_revisions(tmp_path):
    # Revision comments follow the REVISION keyword, not a fixed line offset.
    ds = read_icartt(build_ict(tmp_path))
    assert ds.attrs['Revision'] == 'R1'
    assert ds.attrs['Revision_Comments'] == 'second revision'


def test_read_icartt_times(tmp_path):
    ds = read_icartt(build_ict(tmp_path))
    expected = np.array(
        ['2024-03-15T00:00:00', '2024-03-15T00:00:01', '2024-03-15T00:00:02'],
        dtype='datetime64[ns]',
    )
    np.testing.assert_array_equal(ds['time'].values, expected)


def test_icartt_nlhead(tmp_path):
    ict = Icartt.from_file(build_ict(tmp_path))
    assert ict.NLHEAD == 14 + ict.NV + ict.NSCOML + ict.NNCOML
    assert ict.NLHEAD == ict.declared_nlhead
    assert ict.NV == 2
    assert ict.NNCOML == len(SAMPLE_NCOM)


def test_icartt_nlhead_mismatch_warns(tmp_path):
    path = build_ict(tmp_path, nlhead=99)
    with pytest.warns(UserWarning, match='declares 99 header lines'):
        Icartt.from_file(path)


def test_icartt_missing_keyword_warns(tmp_path):
    ncom = [line for line in SAMPLE_NCOM if not line.startswith('PROJECT_INFO')]
    with pytest.warns(UserWarning, match='PROJECT_INFO'):
        read_icartt(build_ict(tmp_path, ncom=ncom))


def test_write_icartt_roundtrip(tmp_path):
    ds1 = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)
    out = tmp_path / 'AAFNAV_COR_20181104_R0.ict'
    write_icartt(ds1, out)

    ds2 = read_icartt(str(out))
    assert list(ds1.data_vars) == list(ds2.data_vars)
    np.testing.assert_array_equal(ds1['time'].values, ds2['time'].values)
    xr.testing.assert_allclose(ds1, ds2, rtol=1e-9)
    assert ds1.attrs == ds2.attrs
    for name in ds1.data_vars:
        assert ds1[name].attrs == ds2[name].attrs
        assert np.isnan(ds1[name].values).sum() == np.isnan(ds2[name].values).sum()


def test_write_icartt_nlhead(tmp_path):
    # The written header count is recomputed from the content.
    ds = read_icartt(build_ict(tmp_path))
    out = tmp_path / 'OUT_20240315_R1.ict'
    write_icartt(ds, out)

    declared = int(out.read_text().splitlines()[0].split(',')[0])
    ict = Icartt.from_file(str(out))
    assert declared == 14 + ict.NV + ict.NSCOML + ict.NNCOML
    assert declared == ict.NLHEAD


def test_write_icartt_roundtrip_synthetic(tmp_path):
    ds1 = read_icartt(build_ict(tmp_path))
    out = tmp_path / 'OUT_20240315_R1.ict'
    write_icartt(ds1, out)
    ds2 = read_icartt(str(out))

    xr.testing.assert_allclose(ds1, ds2, rtol=1e-9)
    assert ds2.attrs['Revision'] == 'R1'
    assert ds2.attrs['Revision_Comments'] == 'second revision'
    assert ds2['temperature'].attrs['uncertainty'] == '0.5'
    assert ds2['pressure'].attrs['uncertainty'] == '1.0'
    np.testing.assert_array_equal(ds2['temperature'].values, [20.5, np.nan, 21.0])


def test_read_icartt_missing_file(tmp_path):
    missing = str(tmp_path / 'does_not_exist.ict')
    assert read_icartt(missing, return_None=True) is None
    with pytest.raises(FileNotFoundError):
        read_icartt(missing)


def test_read_icartt_truncated_header(tmp_path):
    path = tmp_path / 'TRUNC_20240315_R0.ict'
    path.write_text('12, 1001\nDoe, Jane\n')
    with pytest.raises(ValueError, match='Unexpected end of file'):
        read_icartt(str(path))


def test_read_icartt_bad_format_line(tmp_path):
    path = tmp_path / 'BAD_20240315_R0.ict'
    path.write_text('not a header\nDoe, Jane\n')
    with pytest.raises(ValueError, match='file format line'):
        read_icartt(str(path))


def test_read_icartt_unsupported_ffi(tmp_path):
    path = tmp_path / 'FFI_20240315_R0.ict'
    path.write_text('12, 2110\nDoe, Jane\n')
    with pytest.raises(NotImplementedError, match='FFI 1001'):
        read_icartt(str(path))
    with pytest.raises(NotImplementedError, match='FFI 1001'):
        read_icartt(act.tests.EXAMPLE_AAF_ICARTT, ict_format=2110)


def test_read_icartt_variable_count_mismatch(tmp_path):
    path = tmp_path / 'MISMATCH_20240315_R0.ict'
    path.write_text(
        '\n'.join(
            [
                '20, 1001',
                'Doe, Jane',
                'Test Org',
                'Test Instrument',
                'TESTMISSION',
                '1, 1',
                '2024,03,15,2024,03,16',
                '1.0',
                'Start_UTC, seconds',
                '2',
                '1',
                '-9999, -9999',
            ]
        )
        + '\n'
    )
    with pytest.raises(ValueError, match='scale factor line'):
        read_icartt(str(path))
