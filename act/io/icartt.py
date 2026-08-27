"""
Modules for Reading/Writing the International Consortium for Atmospheric
Research on Transport and Transformation (ICARTT) file format standards V2.0

This module implements the ICARTT FFI 1001 format directly, so no third-party
ICARTT library is required.

References:
    ICARTT V2.0 Standards/Conventions:
    - https://www.earthdata.nasa.gov/s3fs-public/imported/ESDS-RFC-029v2.pdf

"""

import ast
import re
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# Deprecated. ICARTT support is built in now, so this is always True. Retained so
# existing ``skipif`` guards and downstream references keep working.
_ICARTT_AVAILABLE = True

#: Field delimiter for the ICARTT format (ESDS-RFC-029v2 section 2.3.2).
DEFAULT_FIELD_DELIM = ','

#: Numeric format used when writing data records.
DEFAULT_NUM_FORMAT = '%.10g'

#: Scale factor and missing value assumed for the independent variable, which
#: carries neither in the header (ESDS-RFC-029v2 section 2.3.2.12).
DEFAULT_SCALE_FACTOR = 1.0
DEFAULT_MISSING_VALUE = -9999.0

#: Required normal-comment keywords, in the order given by ESDS-RFC-029v2 Table 1.
REQUIRED_KEYWORDS = (
    'PI_CONTACT_INFO',
    'PLATFORM',
    'LOCATION',
    'ASSOCIATED_DATA',
    'INSTRUMENT_INFO',
    'DATA_INFO',
    'UNCERTAINTY',
    'ULOD_FLAG',
    'ULOD_VALUE',
    'LLOD_FLAG',
    'LLOD_VALUE',
    'DM_CONTACT_INFO',
    'PROJECT_INFO',
    'STIPULATIONS_ON_USE',
    'OTHER_COMMENTS',
    'REVISION',
)

# Revision keywords are the current and all previous revision identifiers, e.g.
# "R0", "RA", "R12" (ESDS-RFC-029v2 Table 1, row 17).
_REVISION_RE = re.compile(r'^R[A-Za-z0-9]{1,2}$')


class IcarttVariable:
    """
    A single ICARTT variable description.

    Parameters
    ----------
    shortname : str
        Variable short name, used as the data column header.
    units : str
        Variable units, or 'none' if unitless.
    standardname : str, optional
        Variable standard name from the controlled list.
    longname : str, optional
        Free-form descriptive name.
    scale : str or float, optional
        Scale factor for the variable.
    miss : str or float, optional
        Missing data flag for the variable.

    """

    __slots__ = ('shortname', 'units', 'standardname', 'longname', 'scale', 'miss')

    def __init__(
        self,
        shortname,
        units,
        standardname=None,
        longname=None,
        scale=DEFAULT_SCALE_FACTOR,
        miss=DEFAULT_MISSING_VALUE,
    ):
        self.shortname = shortname
        self.units = units
        self.standardname = standardname
        self.longname = longname
        self.scale = scale
        self.miss = miss

    @classmethod
    def from_desc(cls, parts, **kwargs):
        """
        Build a variable from a split header description line.

        Per ESDS-RFC-029v2 section 2.3.2.13 the line is
        ``shortname, units, standardname, [longname]``. The long name may itself
        contain commas, so any trailing fields are rejoined into it.

        """
        parts = [p.strip() for p in parts]
        shortname = parts[0] if parts else ''
        units = parts[1] if len(parts) > 1 else ''
        standardname = parts[2] if len(parts) > 2 else None
        longname = DEFAULT_FIELD_DELIM.join(parts[3:]) if len(parts) > 3 else None
        return cls(shortname, units, standardname, longname, **kwargs)

    def desc(self, delimiter=DEFAULT_FIELD_DELIM):
        """Variable description string as it appears in an ICARTT file."""
        out = [str(self.shortname), str(self.units)]
        if self.standardname is not None:
            out.append(str(self.standardname))
        if self.longname is not None:
            out.append(str(self.longname))
        return delimiter.join(out)

    def __str__(self):
        return self.desc()

    def __repr__(self):
        return f'IcarttVariable({self.shortname!r}, {self.units!r})'


class Icartt:
    """

    Container for an ICARTT FFI 1001 file: the full header model plus the data
    records. Reads and writes the format described by ESDS-RFC-029v2.

    Attributes are named after the fields in the standard, so the header can be
    inspected and edited directly before writing.

    Examples
    --------
    .. code-block :: python

        from act.io.icartt import Icartt

        ict = Icartt.from_file('AAFNAV_COR_20181104_R0.ict')
        print(ict.NV, ict.keywords['PLATFORM'])
        ds = ict.to_xarray()

    """

    def __init__(self):
        # Line 1 - file format information.
        self.FFI = 1001
        self.version = None
        # Number of header lines declared by the file, kept for validation only.
        # The authoritative value is the computed ``NLHEAD`` property.
        self.declared_nlhead = None

        # Lines 2-5 - originator, affiliation, data source, mission.
        self.ONAME = ''
        self.ORG = ''
        self.SNAME = ''
        self.MNAME = ''

        # Line 6 - file volume number, total number of file volumes.
        self.IVOL = 1
        self.VVOL = 1

        # Line 7 - collection and revision dates as (yyyy, mm, dd) tuples.
        self.DATE = (1970, 1, 1)
        self.RDATE = (1970, 1, 1)

        # Line 8 - data interval code(s).
        self.DX = [1.0]

        # Line 9 - independent variable definition.
        self.XNAME = None

        # Lines 10 to 12+NV - dependent variable definitions.
        self.VNAME = []

        # Special comments.
        self.SCOM = []

        # Normal comments, split into the three parts of section 2.3.2.17.
        self.freeform = []
        self.keywords = OrderedDict((k, '') for k in REQUIRED_KEYWORDS)
        self.shortnames = []

        # Data records, keyed by variable short name.
        self.data = {}

        # Source or destination path.
        self.name = ''

    # ------------------------------------------------------------------
    # Derived header fields
    # ------------------------------------------------------------------

    @property
    def NV(self):
        """Number of dependent variables (header line 10)."""
        return len(self.VNAME)

    @property
    def VSCAL(self):
        """Scale factors, one per dependent variable (header line 11)."""
        return [v.scale for v in self.VNAME]

    @property
    def VMISS(self):
        """Missing data flags, one per dependent variable (header line 12)."""
        return [v.miss for v in self.VNAME]

    @property
    def NSCOML(self):
        """Number of special comment lines."""
        return len(self.SCOM)

    @property
    def NCOM(self):
        """
        Normal comment lines, rebuilt from the parsed parts.

        Ordered as free-form text, then the keyword block, then the variable
        short name list, which must always be the last line.

        """
        lines = list(self.freeform)
        for key, value in self.keywords.items():
            body = value if value else 'N/A'
            lines.extend(f'{key}: {body}'.split('\n'))
        lines.append(DEFAULT_FIELD_DELIM.join(self.shortnames))
        return lines

    @property
    def NNCOML(self):
        """Number of normal comment lines."""
        return len(self.NCOM)

    @property
    def NLHEAD(self):
        """
        Number of header lines.

        Computed rather than stored, per ESDS-RFC-029v2 section 2.3.2.1: 14
        fixed lines plus one line per dependent variable, special comment and
        normal comment.

        """
        return 14 + self.NV + self.NSCOML + self.NNCOML

    @property
    def variables(self):
        """All variables, independent first, keyed by short name."""
        out = OrderedDict()
        if self.XNAME is not None:
            out[self.XNAME.shortname] = self.XNAME
        for var in self.VNAME:
            out[var.shortname] = var
        return out

    @property
    def times(self):
        """
        Time steps of the data as a ``numpy.datetime64[ns]`` array.

        The independent variable is seconds since UTC midnight of the collection
        date (ESDS-RFC-029v2 section 2.3.2.9).

        """
        ref = np.datetime64(datetime(*self.DATE), 'ns')
        values = np.asarray(self.data[self.XNAME.shortname], dtype=np.float64)
        return ref + (values * 10**9).astype('timedelta64[ns]')

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filename, delimiter=DEFAULT_FIELD_DELIM):
        """
        Read an ICARTT FFI 1001 file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the file to read.
        delimiter : str, optional
            Field delimiter. The standard mandates a comma.

        Returns
        -------
        ict : Icartt

        """
        obj = cls()
        obj.name = str(filename)
        with open(filename, encoding='utf-8', errors='replace') as fh:
            obj._read_header(fh, delimiter)
            obj._read_data(fh, delimiter)
        return obj

    def _read_header(self, fh, delimiter):
        """Read the header, following the line order of section 2.3.2."""

        def readline(split=True):
            line = fh.readline()
            if line == '':
                raise ValueError(
                    f'Unexpected end of file while reading the ICARTT header of {self.name}'
                )
            line = line.rstrip('\r\n')
            if split:
                return [part.strip() for part in line.split(delimiter)]
            return line

        # Line 1 - number of header lines, file format index, optional version.
        first = readline()
        try:
            self.declared_nlhead = int(first[0])
            self.FFI = int(first[1])
        except (IndexError, ValueError) as err:
            raise ValueError(
                f'Could not parse the ICARTT file format line of {self.name}: {first!r}'
            ) from err
        if len(first) > 2 and first[2]:
            self.version = first[2]

        if self.FFI != 1001:
            raise NotImplementedError(
                f'ACT supports the ICARTT FFI 1001 format only, this file declares {self.FFI}'
            )

        # Lines 2-5.
        self.ONAME = readline(False)
        self.ORG = readline(False)
        self.SNAME = readline(False)
        self.MNAME = readline(False)

        # Line 6 - file volume number, total number of file volumes.
        volumes = readline()
        self.IVOL = int(volumes[0])
        self.VVOL = int(volumes[1])

        # Line 7 - collection date, revision date.
        dates = readline()
        if len(dates) < 6:
            raise ValueError(
                f'ICARTT date line of {self.name} needs six fields, found {len(dates)}'
            )
        self.DATE = tuple(int(x) for x in dates[:3])
        self.RDATE = tuple(int(x) for x in dates[3:6])

        # Line 8 - data interval code.
        self.DX = [float(x) for x in readline()]

        # Line 9 - independent variable definition.
        self.XNAME = IcarttVariable.from_desc(readline())

        # Line 10 - number of dependent variables.
        nvar = int(readline()[0])

        # Lines 11-12 - scale factors and missing value flags.
        vscal = readline()
        vmiss = readline()
        for label, values in (('scale factor', vscal), ('missing value', vmiss)):
            if len(values) != nvar:
                raise ValueError(
                    f'ICARTT {label} line of {self.name} has {len(values)} entries '
                    f'but the file declares {nvar} dependent variables'
                )

        # Lines 13 to 12+NV - dependent variable definitions.
        self.VNAME = [
            IcarttVariable.from_desc(readline(), scale=vscal[idx], miss=vmiss[idx])
            for idx in range(nvar)
        ]

        # Special comments.
        nscoml = int(readline()[0])
        self.SCOM = [readline(False) for _ in range(nscoml)]

        # Normal comments.
        nncoml = int(readline()[0])
        self._ingest_normal_comments([readline(False) for _ in range(nncoml)])

        # Validate against the counts the file itself declared. The NLHEAD
        # property is the canonical count for writing, which can legitimately
        # differ here when a required keyword was absent and gets restored.
        parsed_nlhead = 14 + nvar + nscoml + nncoml
        if self.declared_nlhead != parsed_nlhead:
            warnings.warn(
                f'ICARTT file {self.name} declares {self.declared_nlhead} header lines '
                f'but {parsed_nlhead} were parsed',
                stacklevel=2,
            )

    def _ingest_normal_comments(self, raw):
        """
        Split the normal comments into free-form text, keywords and short names.

        Follows ESDS-RFC-029v2 section 2.3.2.17: free-form text runs until the
        first required keyword, keyword values continue until the next keyword
        line, and the final line is always the variable short name list.

        """
        raw = list(raw)
        if not raw:
            raise ValueError(
                f'ICARTT file {self.name} has an empty normal comments section, but the '
                'variable short name line is required'
            )

        # The last line is always the comma separated list of short names.
        self.shortnames = [name.strip() for name in raw.pop().split(DEFAULT_FIELD_DELIM)]

        buffers = OrderedDict((key, []) for key in REQUIRED_KEYWORDS)
        self.freeform = []
        current = None

        for line in raw:
            keyword = None
            # Keywords start the line with no leading whitespace and are followed
            # by a colon. The space before the colon seen in some revision lines
            # is tolerated.
            if ':' in line and not line[:1].isspace():
                head = line.split(':', 1)[0].rstrip()
                if head in buffers or _REVISION_RE.match(head):
                    keyword = head

            if keyword is not None:
                current = keyword
                buffers.setdefault(current, [])
                buffers[current].append(line.split(':', 1)[1].strip())
            elif current is None:
                self.freeform.append(line)
            else:
                # Continuation of the previous keyword's value.
                buffers[current].append(line.strip())

        missing = [key for key in REQUIRED_KEYWORDS if not buffers[key]]
        if missing:
            warnings.warn(
                f'ICARTT file {self.name} is missing required normal comment '
                f"keywords: {', '.join(missing)}",
                stacklevel=3,
            )

        self.keywords = OrderedDict((key, '\n'.join(val)) for key, val in buffers.items())

    def _read_data(self, fh, delimiter):
        """Read the data records into ``self.data``, missing values as NaN."""
        names = list(self.variables)
        missing = {name: var.miss for name, var in self.variables.items()}

        with warnings.catch_warnings():
            # genfromtxt warns on an empty file; an empty dataset is legal here.
            warnings.simplefilter('ignore')
            records = np.genfromtxt(
                fh,
                names=names,
                dtype=[(name, np.float64) for name in names],
                missing_values=missing,
                usemask=True,
                delimiter=delimiter,
                deletechars='',
            ).filled(fill_value=np.nan)

        self.data = {name: np.atleast_1d(records[name]) for name in names}

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _keyword(self, key):
        """Keyword value, with the standard's 'N/A' stand-in for an empty one."""
        value = self.keywords.get(key, '').strip()
        return value if value else 'N/A'

    def _revision_comments(self):
        """Comments for the revision named by the REVISION keyword."""
        revision = self.keywords.get('REVISION', '').strip()
        if revision in self.keywords:
            return self.keywords[revision].strip()
        for key, value in self.keywords.items():
            if key not in REQUIRED_KEYWORDS and _REVISION_RE.match(key):
                return value.strip()
        return 'N/A'

    def _per_variable_values(self, key):
        """
        Map a keyword holding one entry per dependent variable onto short names.

        Sized against NV, not the total variable count: the independent variable
        has no uncertainty or limit of detection entry (sections 2.3.2.12,
        2.1.4.3). Returns an empty mapping when the counts do not line up, which
        means the file did not supply per-variable values.

        """
        raw = self.keywords.get(key, '').strip()
        if not raw:
            return {}
        parts = [part.strip() for part in raw.split(DEFAULT_FIELD_DELIM)]
        if len(parts) != self.NV:
            return {}
        return {var.shortname: value for var, value in zip(self.VNAME, parts)}

    def _per_variable_flags(self, key):
        """
        Map a limit of detection flag keyword onto short names.

        Section 2.1.4.3 allows either a single flag for the whole file or one per
        dependent variable.

        """
        raw = self.keywords.get(key, '').strip()
        if not raw:
            return {}
        parts = [part.strip() for part in raw.split(DEFAULT_FIELD_DELIM)]
        if len(parts) == self.NV and self.NV != 1:
            return {var.shortname: value for var, value in zip(self.VNAME, parts)}
        return {name: raw for name in self.variables}

    def to_xarray(self):
        """
        Convert to an `xarray.Dataset` with a ``time`` coordinate.

        Returns
        -------
        ds : xarray.Dataset

        """
        times = self.times

        uncertainty = self._per_variable_values('UNCERTAINTY')
        ulod_value = self._per_variable_values('ULOD_VALUE')
        llod_value = self._per_variable_values('LLOD_VALUE')
        ulod_flag = self._per_variable_flags('ULOD_FLAG')
        llod_flag = self._per_variable_flags('LLOD_FLAG')

        ds = xr.Dataset()
        for name, var in self.variables.items():
            # Short name for a quality flag is standardised on read.
            out_name = 'quality_flag' if name == 'qc_flag' else name
            da = xr.DataArray(
                self.data[name],
                coords=dict(time=times),
                name=out_name,
                dims=['time'],
            )
            da.attrs['uncertainty'] = uncertainty.get(name, 'N/A')
            da.attrs['ULOD_Value'] = ulod_value.get(name, 'N/A')
            da.attrs['LLOD_Value'] = llod_value.get(name, 'N/A')
            da.attrs['units'] = var.units
            da.attrs['mvc'] = var.miss
            da.attrs['scale_factor'] = var.scale
            da.attrs['ULOD_Flag'] = ulod_flag.get(name, 'N/A')
            da.attrs['LLOD_Flag'] = llod_flag.get(name, 'N/A')
            ds[out_name] = da

        ds.attrs['PI'] = self.ONAME
        ds.attrs['PI_Affiliation'] = self.ORG
        ds.attrs['Platform'] = self._keyword('PLATFORM')
        ds.attrs['Mission'] = self.MNAME
        ds.attrs['DateOfCollection'] = str(self.DATE)
        ds.attrs['DateOfRevision'] = str(self.RDATE)
        ds.attrs['Data_Interval'] = str(self.DX)
        ds.attrs['Independent_Var'] = str(self.XNAME)
        ds.attrs['Dependent_Var_Num'] = self.NV
        ds.attrs['PI_Contact'] = self._keyword('PI_CONTACT_INFO')
        ds.attrs['Location'] = self._keyword('LOCATION')
        ds.attrs['Associated_Data'] = self._keyword('ASSOCIATED_DATA')
        ds.attrs['Instrument_Info'] = self._keyword('INSTRUMENT_INFO')
        ds.attrs['Data_Info'] = self._keyword('DATA_INFO')
        ds.attrs['DM_Contact'] = self._keyword('DM_CONTACT_INFO')
        ds.attrs['Project_Info'] = self._keyword('PROJECT_INFO')
        ds.attrs['Stipulations'] = self._keyword('STIPULATIONS_ON_USE')
        ds.attrs['Comments'] = self._keyword('OTHER_COMMENTS')
        ds.attrs['Revision'] = self._keyword('REVISION')
        ds.attrs['Revision_Comments'] = self._revision_comments()

        # Additional ARM metadata.
        ds.attrs['_datastream'] = Path(self.name).name.split('_')[0]

        return ds

    @classmethod
    def from_xarray(cls, ds, filename=''):
        """
        Build an `Icartt` from an `xarray.Dataset` produced by :func:`read_icartt`.

        Reverses the mapping applied by `to_xarray`, including the
        ``qc_flag`` to ``quality_flag`` rename.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to convert.
        filename : str or pathlib.Path, optional
            Name to record on the object.

        Returns
        -------
        ict : Icartt

        """

        def tuple_attr(key, default):
            try:
                return tuple(ast.literal_eval(str(ds.attrs[key])))
            except (KeyError, ValueError, SyntaxError, TypeError):
                return default

        obj = cls()
        obj.name = str(filename)
        obj.ONAME = str(ds.attrs.get('PI', 'N/A'))
        obj.ORG = str(ds.attrs.get('PI_Affiliation', 'N/A'))
        obj.SNAME = str(ds.attrs.get('Platform', 'N/A'))
        obj.MNAME = str(ds.attrs.get('Mission', 'N/A'))
        obj.DATE = tuple_attr('DateOfCollection', (1970, 1, 1))
        obj.RDATE = tuple_attr('DateOfRevision', (1970, 1, 1))

        try:
            obj.DX = [float(x) for x in ast.literal_eval(str(ds.attrs['Data_Interval']))]
        except (KeyError, ValueError, SyntaxError, TypeError):
            obj.DX = [1.0]

        independent = str(ds.attrs.get('Independent_Var', 'Start_UTC,seconds'))
        obj.XNAME = IcarttVariable.from_desc(independent.split(DEFAULT_FIELD_DELIM))
        ivar = obj.XNAME.shortname

        for out_name in ds.data_vars:
            name = 'qc_flag' if out_name == 'quality_flag' else str(out_name)
            attrs = ds[out_name].attrs
            values = np.asarray(ds[out_name].values, dtype=np.float64)
            if name == ivar:
                obj.XNAME.units = str(attrs.get('units', obj.XNAME.units))
                obj.data[ivar] = values
                continue
            obj.VNAME.append(
                IcarttVariable(
                    name,
                    str(attrs.get('units', 'none')),
                    scale=attrs.get('scale_factor', DEFAULT_SCALE_FACTOR),
                    miss=attrs.get('mvc', DEFAULT_MISSING_VALUE),
                )
            )
            obj.data[name] = values

        if ivar not in obj.data:
            # The independent variable was dropped from the Dataset, so rebuild it
            # as seconds since UTC midnight of the collection date.
            ref = np.datetime64(datetime(*obj.DATE), 'ns')
            delta = ds['time'].values.astype('datetime64[ns]') - ref
            obj.data[ivar] = delta.astype('timedelta64[ns]').astype(np.float64) / 1e9

        obj.shortnames = [ivar] + [var.shortname for var in obj.VNAME]

        keyword_attrs = (
            ('PI_CONTACT_INFO', 'PI_Contact'),
            ('PLATFORM', 'Platform'),
            ('LOCATION', 'Location'),
            ('ASSOCIATED_DATA', 'Associated_Data'),
            ('INSTRUMENT_INFO', 'Instrument_Info'),
            ('DATA_INFO', 'Data_Info'),
            ('DM_CONTACT_INFO', 'DM_Contact'),
            ('PROJECT_INFO', 'Project_Info'),
            ('STIPULATIONS_ON_USE', 'Stipulations'),
            ('OTHER_COMMENTS', 'Comments'),
            ('REVISION', 'Revision'),
        )
        for keyword, attr in keyword_attrs:
            obj.keywords[keyword] = str(ds.attrs.get(attr, 'N/A'))

        # Per-variable metadata is reconstructed from the variable attributes when
        # every dependent variable carries the same value, matching how the reader
        # broadcasts a single file-wide entry.
        for keyword, attr in (
            ('UNCERTAINTY', 'uncertainty'),
            ('ULOD_FLAG', 'ULOD_Flag'),
            ('ULOD_VALUE', 'ULOD_Value'),
            ('LLOD_FLAG', 'LLOD_Flag'),
            ('LLOD_VALUE', 'LLOD_Value'),
        ):
            values = []
            for var in obj.VNAME:
                out_name = 'quality_flag' if var.shortname == 'qc_flag' else var.shortname
                values.append(str(ds[out_name].attrs.get(attr, 'N/A')))

            if not values:
                obj.keywords[keyword] = 'N/A'
            elif len(set(values)) == 1:
                obj.keywords[keyword] = values[0]
            else:
                obj.keywords[keyword] = DEFAULT_FIELD_DELIM.join(values)

        revision = obj.keywords['REVISION'].strip()
        if _REVISION_RE.match(revision):
            obj.keywords[revision] = str(ds.attrs.get('Revision_Comments', 'N/A'))

        return obj

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, filename=None, fmt=DEFAULT_NUM_FORMAT, delimiter=DEFAULT_FIELD_DELIM):
        """
        Write the object to an ICARTT FFI 1001 file.

        ``NLHEAD`` is recomputed from the content, so the header count is always
        consistent with what is written.

        Parameters
        ----------
        filename : str or pathlib.Path, optional
            Destination path. Defaults to the object's ``name`` attribute.
        fmt : str, optional
            Numeric format for the data records.
        delimiter : str, optional
            Field delimiter. The standard mandates a comma.

        """
        if filename is None:
            filename = self.name
        if not filename:
            raise ValueError('No filename given and the Icartt object has no name set')
        if self.XNAME is None:
            raise ValueError('Cannot write an Icartt object with no independent variable')

        ivar = self.XNAME.shortname
        names = [ivar] + [var.shortname for var in self.VNAME]
        for name in names:
            if name not in self.data:
                raise ValueError(f'No data present for the variable {name!r}')

        # Missing values go back out as the file's own flag rather than NaN.
        columns = [np.asarray(self.data[ivar], dtype=np.float64)]
        for var in self.VNAME:
            column = np.array(self.data[var.shortname], dtype=np.float64, copy=True)
            try:
                column[np.isnan(column)] = float(var.miss)
            except (TypeError, ValueError):
                column[np.isnan(column)] = DEFAULT_MISSING_VALUE
            columns.append(column)

        header = [f'{self.NLHEAD}{delimiter} {self.FFI}']
        if self.version:
            header[0] += f'{delimiter} {self.version}'
        header.append(self.ONAME)
        header.append(self.ORG)
        header.append(self.SNAME)
        header.append(self.MNAME)
        header.append(f'{self.IVOL}{delimiter} {self.VVOL}')
        header.append(delimiter.join(f'{part:d}' for part in (*self.DATE, *self.RDATE)))
        header.append(delimiter.join(str(x) for x in self.DX))
        header.append(self.XNAME.desc(delimiter + ' '))
        header.append(str(self.NV))
        header.append(delimiter.join(str(x) for x in self.VSCAL))
        header.append(delimiter.join(str(x) for x in self.VMISS))
        header.extend(var.desc(delimiter + ' ') for var in self.VNAME)
        header.append(str(self.NSCOML))
        header.extend(self.SCOM)
        header.append(str(self.NNCOML))
        header.extend(self.NCOM)

        with open(filename, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(header))
            fh.write('\n')
            np.savetxt(fh, np.column_stack(columns), fmt=fmt, delimiter=delimiter)

        self.name = str(filename)


def read_icartt(filename, format=1001, return_None=False, **kwargs):
    """

    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ICARTT from a single datastream. Has some procedures to ensure
    time is correctly fomatted in returned Dataset.

    Parameters
    ----------
    filename : str
        Name of file to read.
    format : int or str
        ICARTT format to read. Only FFI 1001 is supported.
    return_None : bool, optional
        Catch IOError exception when file not found and return None.
        Default is False.
    **kwargs : keywords
        keywords to pass on through to Icartt.from_file.

    Returns
    -------
    ds : xarray.Dataset (or None)
        ACT Xarray dataset (or None if no data file(s) found).
    """
    if str(format) not in ('1001', 'FFI1001', 'Formats.FFI1001'):
        raise NotImplementedError(f'ACT supports the ICARTT FFI 1001 format only, got {format!r}')

    try:
        ict = Icartt.from_file(filename, **kwargs)
    except (FileNotFoundError, OSError) as exception:
        if not return_None:
            raise
        if isinstance(exception, FileNotFoundError):
            return None
        if exception.args and exception.args[0] == 'no files to open':
            return None
        raise

    return ict.to_xarray()


def write_icartt(ds, filename, **kwargs):
    """

    Write an `xarray.Dataset` to an ICARTT FFI 1001 file.

    Intended as the inverse of :func:`read_icartt`, so a Dataset produced by it
    round-trips back to a valid ICARTT file. Header metadata is taken from the
    Dataset attributes, and anything absent falls back to 'N/A'.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write. Must have a ``time`` coordinate.
    filename : str or pathlib.Path
        Destination path.
    **kwargs : keywords
        keywords to pass on through to Icartt.write, such as ``fmt``.
    """
    Icartt.from_xarray(ds, filename=filename).write(filename, **kwargs)
