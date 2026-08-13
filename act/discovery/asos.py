"""
Utilities for downloading ASOS data from the Iowa Environmental Mesonet (IEM).
"""

import json
import random
import time
import warnings
from datetime import datetime
from functools import lru_cache
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

IEM_BASE_URL = "https://mesonet.agron.iastate.edu"
IEM_ASOS_URL = f"{IEM_BASE_URL}/cgi-bin/request/asos.py"
USER_AGENT = "ARM-ACT/ASOS (https://github.com/ARM-DOE/ACT)"

MAX_ATTEMPTS = 6
INITIAL_RETRY_DELAY = 5.0
MAX_RETRY_DELAY = 120.0
STATION_BATCH_SIZE = 50


def get_asos_data(
    time_window,
    lat_range=None,
    lon_range=None,
    station=None,
    regions=None,
    variables=None,
):
    """
    Retrieve ASOS observations from the Iowa Environmental Mesonet.

    Data can be requested either by station identifier, geographic bounds, or region.

    Parameters
    ----------
    time_window : tuple
        Two-member sequence containing the start and end times as Python
        ``datetime`` objects.
    lat_range : tuple, optional
        Latitude range ``(min_latitude, max_latitude)`` used to discover ASOS
        stations.
    lon_range : tuple, optional
        Longitude range ``(min_longitude, max_longitude)`` used to discover
        ASOS stations.
    station : str or sequence of str, optional
        One station identifier or a sequence of station identifiers.
        Examples: ``"ORD"`` or ``["ORD", "MDW", "DPA"]``.
    regions : str or sequence of str, optional
        Region/network identifiers. Examples: ``"IL"`` or
        ``["IL", "WI", "IN"]``. Regions can be used alone to retrieve all
        stations in the requested region(s), or together with latitude and
        longitude bounds to narrow geographic station discovery.
    variables : sequence of str, optional
        IEM ASOS variables to request. If omitted, ``data=all`` is used.

    Returns
    -------
    asos_ds : dict of xarray.Dataset
        Dictionary keyed by ASOS station identifier.

    Examples
    --------
    Retrieve data for a single ASOS station.

    .. code-block:: python

        from datetime import datetime

        import act

        time_window = [
            datetime(2020, 2, 4, 2, 0),
            datetime(2020, 2, 4, 4, 0),
        ]

        asos = act.discovery.get_asos_data(
            time_window,
            station="ORD",
        )

    Retrieve multiple ASOS stations in a single request.

    .. code-block:: python

        asos = act.discovery.get_asos_data(
            time_window,
            station=["ORD", "MDW", "DPA"],
        )

    Retrieve all available ASOS stations within a latitude and longitude
    bounding box.

    .. code-block:: python

        lat_range = (41.3, 42.3)
        lon_range = (-88.2, -87.1)

        asos = act.discovery.get_asos_data(
            time_window,
            lat_range=lat_range,
            lon_range=lon_range,
        )

    Retrieve all available ASOS stations from a region.

    .. code-block:: python

        asos = act.discovery.get_asos_data(
            time_window,
            regions="IL",
        )

    Multiple regions can also be requested.

    .. code-block:: python

        asos = act.discovery.get_asos_data(
            time_window,
            regions=["IL", "IN", "WI"],
        )

    Limit the request to selected ASOS variables.

    .. code-block:: python

        asos = act.discovery.get_asos_data(
            time_window,
            station="ORD",
            variables=["tmpf", "dwpf", "relh", "drct", "sknt"],
        )

    Regions can optionally be combined with a geographic bounding box to
    limit station discovery to selected IEM networks.

    .. code-block:: python

        asos = act.discovery.get_asos_data(
            time_window,
            lat_range=lat_range,
            lon_range=lon_range,
            regions=["IL", "IN"],
        )

    Notes
    -----
    Explicit station requests do not require metadata-network lookups. Latitude,
    longitude, and elevation are requested directly from the ASOS endpoint.

    Geographic requests do not require ``regions``. When ``regions`` is not
    supplied, ACT uses a single global IEM METAR/ASOS metadata request and
    filters stations by latitude/longitude. Supplying ``regions`` alone
    retrieves all stations in those regional networks. Large station sets are
    automatically split into smaller batched ASOS requests.

    Code was updated using ChatGPT following testing and feedback from the
    developer.  Code was reviewed and tested by the developer before the PR.

    """
    # Validate the requested period once before performing any discovery or
    # network requests.
    start_time, end_time = _validate_time_window(time_window)

    # There are three supported station-selection modes:
    #   1. Explicit station ID(s)
    #   2. A latitude/longitude bounding box
    #   3. One or more IEM regions/networks
    #
    # Explicit station requests can skip the metadata-discovery step entirely.
    if station is not None:
        stations = _normalize_station_list(station)
        metadata = {}

    # For a geographic request, discover the stations that fall inside the
    # bounding box. If regions are supplied, they are used to narrow the
    # metadata search; otherwise the global station catalog is used.
    elif lat_range is not None and lon_range is not None:
        stations, metadata = _discover_asos_stations(
            lat_range=lat_range,
            lon_range=lon_range,
            regions=regions,
        )

    # A region-only request retrieves all stations belonging to the requested
    # regional ASOS network(s).
    elif regions is not None:
        stations, metadata = _discover_region_stations(regions)

    else:
        raise ValueError("Specify station, regions, or both lat_range and lon_range.")

    if not stations:
        warnings.warn("No ASOS stations matched the request.", UserWarning)
        return {}

    # Large station lists are split into batches. This avoids very long URLs
    # and very large responses while still using far fewer HTTP requests than
    # downloading one station at a time.
    asos_ds = {}
    batches = list(_chunk_stations(stations, STATION_BATCH_SIZE))

    for batch_index, batch in enumerate(batches):
        # Build one IEM request containing every station in this batch.
        uri = _build_asos_url(
            stations=batch,
            start_time=start_time,
            end_time=end_time,
            variables=variables,
        )

        data = _download_data(uri)

        # IEM returns all requested stations in one CSV response. Split that
        # response back into one xarray Dataset per station.
        batch_ds = _parse_asos_response(
            data=data,
            requested_stations=batch,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        )

        asos_ds.update(batch_ds)

        # IEM rate-limits requests by IP. Pause between batches so normal
        # regional downloads do not immediately trigger HTTP 429 responses.
        if batch_index < len(batches) - 1:
            time.sleep(1.0)

    return asos_ds


def _validate_time_window(time_window):
    """Validate and return the requested start/end datetimes."""
    if len(time_window) != 2:
        raise ValueError("time_window must contain exactly two datetime objects.")

    start_time, end_time = time_window

    if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
        raise TypeError("time_window values must be Python datetime objects.")

    if end_time < start_time:
        raise ValueError("The end of time_window must be after the start.")

    return start_time, end_time


def _normalize_station_list(station):
    """Normalize station input to a de-duplicated list of strings."""
    # Treat a single station string and a sequence of station strings the
    # same way internally.
    if isinstance(station, str):
        stations = [station]
    else:
        stations = list(station)

    stations = [str(item).strip() for item in stations if str(item).strip()]

    if not stations:
        raise ValueError("station must contain at least one valid station ID.")

    # dict preserves insertion order, giving us a simple ordered de-duplicate.
    return list(dict.fromkeys(stations))


def _normalize_regions(regions):
    """Normalize region input to a de-duplicated list."""
    if isinstance(regions, str):
        region_list = regions.split()
    else:
        region_list = list(regions)

    region_list = [str(item).strip() for item in region_list if str(item).strip()]
    return list(dict.fromkeys(region_list))


def _discover_region_stations(regions):
    """
    Discover all stations in one or more requested regional ASOS networks.

    The AWOS network is also included so regional requests retain the behavior
    of ACT's historical regional discovery.
    """
    region_list = _normalize_regions(regions)

    if not region_list:
        raise ValueError("regions must contain at least one region identifier.")

    # Include the general AWOS network in addition to each regional ASOS
    # network. This preserves stations that historically appeared in ACT's
    # geographic searches but are not part of a state ASOS network.
    networks = ["AWOS"]
    networks.extend(_region_to_network(region) for region in region_list)

    stations = []
    metadata = {}

    # Metadata requests are cached by _get_network_metadata(), so repeated
    # calls for the same network during one Python session do not hit IEM again.
    for network in dict.fromkeys(networks):
        try:
            jdict = _get_network_metadata(network)
        except HTTPError as exp:
            warnings.warn(
                f"Unable to retrieve IEM network metadata for {network}: {exp}",
                UserWarning,
            )
            continue

        for site in jdict.get("features", []):
            geometry = site.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []
            properties = site.get("properties") or {}

            station_id = properties.get("sid")
            if not station_id:
                continue

            # Preserve the IEM station properties as dataset-level metadata.
            # Latitude/longitude are added explicitly because they come from the
            # GeoJSON geometry rather than the properties dictionary.
            station_metadata = {
                "iem_network": network,
            }

            if len(coordinates) >= 2:
                lon, lat = coordinates[:2]
                station_metadata["site_latitude"] = lat
                station_metadata["site_longitude"] = lon

            station_metadata.update(properties)

            elevation = station_metadata.get("elevation")
            if elevation is not None and not isinstance(elevation, str):
                try:
                    station_metadata["elevation"] = f"{float(elevation):f} meter"
                except (TypeError, ValueError):
                    pass

            metadata[station_id] = station_metadata
            stations.append(station_id)

    return list(dict.fromkeys(stations)), metadata


def _chunk_stations(stations, batch_size):
    """Yield station lists in fixed-size batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for index in range(0, len(stations), batch_size):
        yield stations[index : index + batch_size]


def _discover_asos_stations(lat_range, lon_range, regions=None):
    """
    Discover stations inside a bounding box.

    If ``regions`` is omitted, a single IEM global METAR/ASOS GeoJSON feed is
    used. If regions are supplied, only the corresponding regional ASOS
    networks (plus the AWOS network) are queried.

    Network responses are cached for the lifetime of the Python process.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    if lat_min > lat_max:
        raise ValueError("lat_range must be ordered as (min_latitude, max_latitude).")

    if lon_min > lon_max:
        raise ValueError("lon_range must be ordered as (min_longitude, max_longitude).")

    if regions is None:
        # No region was supplied, so use one global metadata request and do the
        # bounding-box filtering locally. This replaces the older ACT behavior
        # that queried many individual ASOS networks.
        #
        # AZOS is IEM's global METAR/ASOS station metadata feed. Using it
        # avoids querying every individual state/country ASOS network.
        network_metadata = [("AZOS", _get_network_metadata("AZOS"))]
    else:
        # If the caller already knows the relevant regions, query only those
        # smaller network catalogs rather than the global catalog.
        region_list = _normalize_regions(regions)

        if not region_list:
            raise ValueError("regions must contain at least one region identifier.")

        networks = ["AWOS"]
        networks.extend(_region_to_network(region) for region in region_list)

        network_metadata = []
        for network in dict.fromkeys(networks):
            try:
                network_metadata.append((network, _get_network_metadata(network)))
            except HTTPError as exp:
                warnings.warn(
                    f"Unable to retrieve IEM network metadata for " f"{network}: {exp}",
                    UserWarning,
                )

    stations = []
    metadata = {}

    for network, jdict in network_metadata:
        for site in jdict.get("features", []):
            geometry = site.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []

            if len(coordinates) < 2:
                continue

            lon, lat = coordinates[:2]

            # Station filtering happens locally after the metadata are
            # downloaded; no additional IEM request is needed per station.
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue

            properties = site.get("properties") or {}
            station_id = properties.get("sid")

            if not station_id:
                continue

            station_metadata = {
                "site_latitude": lat,
                "site_longitude": lon,
                "iem_network": network,
            }
            station_metadata.update(properties)

            elevation = station_metadata.get("elevation")
            if elevation is not None and not isinstance(elevation, str):
                try:
                    station_metadata["elevation"] = f"{float(elevation):f} meter"
                except (TypeError, ValueError):
                    pass

            metadata[station_id] = station_metadata
            stations.append(station_id)

    return list(dict.fromkeys(stations)), metadata


def _region_to_network(region):
    """Convert a short region name to the corresponding ASOS network."""
    region = region.strip()

    if region.endswith("_ASOS"):
        return region

    return f"{region}_ASOS"


# Network station lists change infrequently compared with how often users may
# call get_asos_data() in a notebook or analysis workflow. Cache them in memory
# for the lifetime of the Python process.
@lru_cache(maxsize=128)
def _get_network_metadata(network):
    """Retrieve and cache IEM GeoJSON metadata for one network."""
    uri = f"{IEM_BASE_URL}/geojson/network/{network}.geojson"
    return _download_json(uri)


def _build_asos_url(stations, start_time, end_time, variables=None):
    """Build a batched IEM ASOS request URL."""
    # Build the query as key/value pairs and let urllib handle escaping.
    # Repeated "station" and "data" parameters are supported by the IEM API.
    params = [
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("elev", "yes"),
        ("sts", start_time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("ets", end_time.strftime("%Y-%m-%dT%H:%M:%SZ")),
    ]

    # Preserve ACT's historical behavior by requesting every available
    # variable unless the caller explicitly asks for a subset.
    if variables is None:
        params.append(("data", "all"))
    else:
        variables = list(dict.fromkeys(str(v).strip() for v in variables if str(v).strip()))

        if not variables:
            raise ValueError("variables must contain at least one variable name.")

        for variable in variables:
            params.append(("data", variable))

    # Add every station in the current batch to the same HTTP request.
    for station_id in stations:
        params.append(("station", station_id))

    return f"{IEM_ASOS_URL}?{urlencode(params)}"


def _parse_asos_response(
    data,
    requested_stations,
    metadata,
    start_time,
    end_time,
):
    """Parse one batched IEM CSV response into station-specific datasets."""
    if not data:
        warnings.warn("No ASOS data were returned by IEM.", UserWarning)
        return {}

    buf = StringIO(data)

    # "onlycomma" normally returns a standard CSV without the five comment
    # lines used by the older IEM comma format.
    try:
        df = pd.read_csv(buf, na_values="M")
    except pd.errors.EmptyDataError:
        warnings.warn(
            "No ASOS data were returned for the requested time period.",
            UserWarning,
        )
        return {}
    finally:
        buf.close()

    if df.empty:
        warnings.warn(
            "No ASOS data were returned for the requested time period.",
            UserWarning,
        )
        return {}

    # Some IEM formats may include comment/header lines. If the normal read did
    # not produce the expected columns, fall back to the historic ACT behavior.
    if "station" not in df.columns or "valid" not in df.columns:
        buf = StringIO(data)
        try:
            df = pd.read_csv(buf, skiprows=5, na_values="M")
        except pd.errors.EmptyDataError:
            return {}
        finally:
            buf.close()

    if "station" not in df.columns or "valid" not in df.columns:
        raise ValueError(
            "Unexpected IEM ASOS response: required columns 'station' and "
            "'valid' were not present."
        )

    # Convert IEM's UTC valid time to a timezone-naive datetime coordinate,
    # matching the style used by the existing ACT discovery routines.
    df["time"] = pd.to_datetime(
        df["valid"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    df = df.dropna(subset=["time"])

    asos_ds = {}

    # The response contains all stations in the batch. Grouping here converts
    # that single network response back into ACT's dictionary-of-Datasets API.
    for station_id, station_df in df.groupby("station", sort=False):
        station_df = station_df.copy()

        station_df = station_df.set_index("time")
        station_df = station_df.drop(
            columns=["valid", "station"],
            errors="ignore",
        )

        if station_df.empty:
            continue

        ds = station_df.to_xarray()

        # Geographic/region discovery already provides station metadata.
        # Explicit station requests leave this empty and populate basic
        # latitude/longitude/elevation attributes from the returned CSV below.
        ds.attrs = metadata.get(station_id, {}).copy()
        ds.attrs["_datastream"] = station_id

        # Populate metadata directly from returned observation columns when
        # explicit station requests did not require a network metadata lookup.
        _populate_returned_station_metadata(ds)

        _set_asos_attributes(ds)

        asos_ds[station_id] = ds

    # IEM may know about a station that has no observations during the
    # requested period. Report those stations without failing the whole batch.
    missing = [station for station in requested_stations if station not in asos_ds]

    for station_id in missing:
        warnings.warn(
            f"No data available at station {station_id} between "
            f"{start_time:%Y-%m-%d %H:%M:%S} and "
            f"{end_time:%Y-%m-%d %H:%M:%S}.",
            UserWarning,
        )

    return asos_ds


def _populate_returned_station_metadata(ds):
    """
    Populate dataset attributes from lat/lon/elevation returned by IEM.

    The data variables are retained because they may be useful to downstream
    workflows, while representative values are also copied into attributes.
    """
    if "lat" in ds and ds["lat"].size:
        value = _first_valid_scalar(ds["lat"].values)
        if value is not None:
            ds.attrs.setdefault("site_latitude", value)

    if "lon" in ds and ds["lon"].size:
        value = _first_valid_scalar(ds["lon"].values)
        if value is not None:
            ds.attrs.setdefault("site_longitude", value)

    for elevation_name in ("elevation", "elev"):
        if elevation_name in ds and ds[elevation_name].size:
            value = _first_valid_scalar(ds[elevation_name].values)
            if value is not None:
                ds.attrs.setdefault("elevation", f"{value} meter")
                break


def _first_valid_scalar(values):
    """Return the first non-null scalar from an array-like object."""
    array = np.asarray(values).ravel()

    for value in array:
        if pd.notna(value):
            if hasattr(value, "item"):
                try:
                    return value.item()
                except ValueError:
                    pass
            return value

    return None


def _set_asos_attributes(ds):
    """Set units, long names, and derived ASOS variables."""
    _set_attrs(ds, "lon", "degree", "Longitude")
    _set_attrs(ds, "lat", "degree", "Latitude")

    _set_attrs(ds, "tmpf", "degrees Fahrenheit", "Temperature in degrees Fahrenheit")
    if "tmpf" in ds:
        # Keep the original Fahrenheit variable and add an ACT-friendly Celsius
        # representation rather than replacing the source observation.
        ds["temp"] = (ds["tmpf"] - 32.0) * (5.0 / 9.0)
        _set_attrs(ds, "temp", "degrees Celsius", "Temperature in degrees Celsius")

    _set_attrs(
        ds,
        "dwpf",
        "degrees Fahrenheit",
        "Dewpoint temperature in degrees Fahrenheit",
    )
    if "dwpf" in ds:
        ds["dwpc"] = (ds["dwpf"] - 32.0) * (5.0 / 9.0)
        _set_attrs(
            ds,
            "dwpc",
            "degrees Celsius",
            "Dewpoint temperature in degrees Celsius",
        )

    _set_attrs(ds, "relh", "percent", "Relative humidity")
    _set_attrs(ds, "drct", "degrees", "Wind direction")
    _set_attrs(ds, "sknt", "knots", "Wind speed in knots")

    if "sknt" in ds:
        ds["spdms"] = ds["sknt"] * 0.514444
        _set_attrs(ds, "spdms", "m s-1", "Wind speed in meters per second")

    if "drct" in ds and "spdms" in ds:
        # Meteorological wind direction is the direction FROM which the wind
        # blows, hence the negative signs in the u/v conversion.
        ds["u"] = -np.sin(np.deg2rad(ds["drct"])) * ds["spdms"]
        _set_attrs(ds, "u", "m s-1", "Zonal component of surface wind")

        ds["v"] = -np.cos(np.deg2rad(ds["drct"])) * ds["spdms"]
        _set_attrs(ds, "v", "m s-1", "Meridional component of surface wind")

    _set_attrs(ds, "mslp", "mb", "Mean Sea Level Pressure")
    _set_attrs(ds, "alti", "in Hg", "Atmospheric pressure in inches of Mercury")
    _set_attrs(ds, "vsby", "mi", "Visibility")

    if "vsby" in ds:
        ds["vsbykm"] = ds["vsby"] * 1.60934
        _set_attrs(ds, "vsbykm", "km", "Visibility")

    if "gust" in ds:
        ds["gust"] = ds["gust"] * 0.514444
        _set_attrs(ds, "gust", "m s-1", "Wind gust speed")

    for level in range(1, 5):
        skyc = f"skyc{level}"
        skyl = f"skyl{level}"

        _set_attrs(ds, skyc, None, f"Sky level {level} coverage")

        if skyl in ds:
            ds[skyl] = ds[skyl] * 0.3048
            _set_attrs(ds, skyl, "meter", f"Sky level {level} altitude")

    _set_attrs(ds, "wxcodes", None, "Weather code")

    for hours in (1, 3, 6):
        variable = f"ice_accretion_{hours}hr"
        if variable in ds:
            ds[variable] = ds[variable] * 2.54
            _set_attrs(ds, variable, "cm", f"{hours} hour ice accretion")

    if "peak_wind_gust" in ds:
        ds["peak_wind_gust"] = ds["peak_wind_gust"] * 0.514444
        _set_attrs(ds, "peak_wind_gust", "m s-1", "Peak wind gust speed")

    _set_attrs(ds, "peak_wind_drct", "degree", "Peak wind gust direction")

    if "peak_wind_drct" in ds and "peak_wind_gust" in ds:
        ds["u_peak"] = -np.sin(np.deg2rad(ds["peak_wind_drct"])) * ds["peak_wind_gust"]
        _set_attrs(
            ds,
            "u_peak",
            "m s-1",
            "Zonal component of peak surface wind",
        )

        ds["v_peak"] = -np.cos(np.deg2rad(ds["peak_wind_drct"])) * ds["peak_wind_gust"]
        _set_attrs(
            ds,
            "v_peak",
            "m s-1",
            "Meridional component of peak surface wind",
        )

    _set_attrs(ds, "metar", None, "Raw METAR code")


def _set_attrs(ds, variable, units=None, long_name=None):
    """Safely set variable attributes when the variable exists."""
    if variable not in ds:
        return

    if units is not None:
        ds[variable].attrs["units"] = units

    if long_name is not None:
        ds[variable].attrs["long_name"] = long_name


def _download_json(uri):
    """Download and decode an IEM JSON response."""
    return json.loads(_request_url(uri))


def _download_data(uri):
    """Download ASOS CSV data."""
    return _request_url(uri, reject_error_response=True)


def _request_url(
    uri,
    max_attempts=MAX_ATTEMPTS,
    initial_delay=INITIAL_RETRY_DELAY,
    max_delay=MAX_RETRY_DELAY,
    reject_error_response=False,
):
    """
    Download data from IEM with retry, exponential backoff, and jitter.

    HTTP 429 responses honor the ``Retry-After`` header when supplied.
    HTTP 5xx responses and temporary network errors are retried.
    """
    # Identify ACT to the remote service rather than relying on Python's
    # default urllib User-Agent.
    headers = {"User-Agent": USER_AGENT}
    last_exception = None

    for attempt in range(max_attempts):
        try:
            request = Request(uri, headers=headers)

            with urlopen(request, timeout=300) as response:
                data = response.read().decode("utf-8")

            if reject_error_response and (not data or data.lstrip().startswith("ERROR")):
                raise RuntimeError("IEM returned an empty or ERROR response.")

            return data

        except HTTPError as exp:
            last_exception = exp

            # Retry only failures that are likely to be temporary:
            #   429       - IEM rate limit
            #   500-599   - server-side errors
            if exp.code == 429 or 500 <= exp.code < 600:
                delay = _retry_delay(
                    attempt=attempt,
                    retry_after=exp.headers.get("Retry-After"),
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                )

                if attempt < max_attempts - 1:
                    warnings.warn(
                        f"IEM request returned HTTP {exp.code}. "
                        f"Retrying in {delay:.1f} seconds "
                        f"(attempt {attempt + 1}/{max_attempts}).",
                        UserWarning,
                    )
                    time.sleep(delay)
                    continue

            raise

        except (URLError, TimeoutError, RuntimeError) as exp:
            last_exception = exp
            delay = _retry_delay(
                attempt=attempt,
                retry_after=None,
                initial_delay=initial_delay,
                max_delay=max_delay,
            )

            if attempt < max_attempts - 1:
                warnings.warn(
                    f"IEM request failed: {exp}. "
                    f"Retrying in {delay:.1f} seconds "
                    f"(attempt {attempt + 1}/{max_attempts}).",
                    UserWarning,
                )
                time.sleep(delay)
                continue

    raise RuntimeError(
        f"Unable to download IEM ASOS data after {max_attempts} attempts: {uri}"
    ) from last_exception


def _retry_delay(attempt, retry_after, initial_delay, max_delay):
    """Calculate a retry delay using Retry-After or exponential backoff."""
    delay = None

    # Prefer the server-provided Retry-After value when available.
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = None

    # Otherwise progressively increase the wait time after each failure.
    if delay is None:
        delay = initial_delay * (2**attempt)

    delay = min(delay, max_delay)

    # Add jitter so multiple ACT clients that were throttled at the same time
    # do not all retry again at exactly the same instant.
    delay += random.uniform(0.0, max(0.25, delay * 0.25))

    return delay
