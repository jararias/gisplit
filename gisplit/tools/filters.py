
import warnings
import functools

import numpy as np
import pandas as pd


def guess_time_step(df):
    time_steps, counts = np.unique(np.diff(df.index), return_counts=True)
    time_step = time_steps[np.argsort(counts)[-1]]  # get the modal time step
    return pd.Timedelta(time_step.astype('timedelta64[s]'), 'S')


def fill_data(df):
    def previous_midnight(dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    time_step = guess_time_step(df)
    n_periods = (df.index[0] - previous_midnight(df.index[0])) // time_step
    start_date = df.index[0] - n_periods*time_step
    next_midnight = previous_midnight(df.index[-1]) + pd.Timedelta('1D')
    n_periods = (next_midnight - df.index[-1]) // time_step
    end_date = df.index[-1] + n_periods*time_step
    return df.reindex(pd.date_range(start_date, end_date, freq=time_step))


def get_site_from_index(df):
    if not isinstance(df.index, pd.MultiIndex):
        return None
    if 'site' not in df.index.names:
        return None
    sites = df.index.get_level_values('site')
    if sites.str.contains('/').all():
        sites, _ = zip(*sites.str.split('/'))
    return sites


def get_network_from_index(df):
    if not isinstance(df.index, pd.MultiIndex):
        return None
    if 'site' not in df.index.names:
        return None
    network = None
    site_and_network = df.index.get_level_values('site')
    if site_and_network.str.contains('/').all():
        _, network = zip(*site_and_network.str.split('/'))
    return network


def add_site_to_index(df, site=None):
    site_values = site
    if site_values is None:
        if ('site' in df) and ('network' in df):
            site_values = df['site'].str.cat(df['network'], sep='/')
        else:
            site_values = df['site']
    if isinstance(site_values, str):
        site_values = [site_values]*len(df)
    level_names = ['times_utc', 'site']
    index = pd.MultiIndex.from_arrays([df.index, site_values], names=level_names)
    if isinstance(df, pd.DataFrame):
        return df.set_index(index).drop(columns=['site'], errors='ignore')
    return df.set_axis(index, axis='index')


def drop_site_from_index(df):
    if isinstance(df.index, pd.MultiIndex):
        if isinstance(df, pd.DataFrame):
            return df.set_index(df.index.get_level_values('times_utc'))
        return df.set_axis(df.index.get_level_values('times_utc'), axis='index')
    return df


def add_from_metadata(df, metadata, variable):
    if 'site' in df:
        sites = df.sites
    elif 'site' in df.index.names:
        sites = df.index.get_level_values('site')
    else:
        raise ValueError('unknown site')

    def get_site_values(site):
        site_name, network = site.split('/') if '/' in site else (site, None)
        if network is None:
            site_metadata = (
                metadata.get('bsrn').get(site_name, None) or
                metadata.get('pvps').get(site_name, None))
        else:
            site_metadata = metadata.get(network).get(site_name, None)
        return site_metadata.get(variable)

    values = [[get_site_values(site)]*len(subset) for site, subset in df.groupby(sites)]
    return df.assign(**{variable: functools.reduce(lambda a, b: a+b, values)})


def add_climate(df, metadata):
    return add_from_metadata(df, metadata, 'climate').pipe(
        lambda df: df.assign(climate=df.climate.str.__getitem__(0))
    )


def add_longitude(df, metadata):
    return add_from_metadata(df, metadata, 'longitude')


def add_latitude(df, metadata):
    return add_from_metadata(df, metadata, 'latitude')


def rolling_mean(series, window='1H', **kwargs):
    kwargs.setdefault('min_periods', 1)
    kwargs.setdefault('center', True)
    return series.rolling(window, **kwargs).mean()


def rolling_sum(series, window='1H', **kwargs):
    kwargs.setdefault('min_periods', 1)
    kwargs.setdefault('center', True)
    return series.rolling(window, **kwargs).sum()


def close_radiation(df):
    """Calculate ghi, dif and dni, when possible"""

    if 'dni' not in df and 'sza' not in df:
        warnings.warn('missing `sza`. Cannot compute `dni`')

    if 'ghi' not in df:

        if 'dif' in df and 'dni' in df and 'sza' in df:
            df['ghi'] = df.eval('dif + dni*cos(0.017453292519943295*sza)')

        elif 'dif' in df and 'dir' in df:
            df['ghi'] = df.eval('dif + dir')
            if 'sza' in df:
                df['dni'] = df.eval('dir / cos(0.017453292519943295*sza)')
        else:
            warnings.warn('cannot close solar radiation', UserWarning)

    if 'dif' not in df:

        if 'ghi' in df and 'dni' in df and 'sza' in df:
            df['dif'] = df.eval('ghi - dni*cos(0.017453292519943295*sza)')
        elif 'ghi' in df and 'dir' in df:
            df['dif'] = df.eval('ghi - dir')
            if 'dni' not in df and 'sza' in df:
                df['dni'] = df.eval('dir / cos(0.017453292519943295*sza)')
        else:
            warnings.warn('cannot close solar radiation', UserWarning)

    if 'dni' not in df:

        if 'ghi' in df and 'dif' in df and 'sza' in df:
            df['dni'] = df.eval('(ghi - dif) / cos(0.017453292519943295*sza)')
        elif 'ghi' not in df and 'dif' in df and 'dir' in df:
            df['ghi'] = df.eval('dif + dir')
            if 'sza' in df:
                df['dni'] = df.eval('dir / cos(0.017453292519943295*sza)')
        elif 'ghi' in df and 'dif' not in df and 'dir' in df:
            df['dif'] = df.eval('ghi - dir')
            if 'sza' in df:
                df['dni'] = df.eval('dir / cos(0.017453292519943295*sza)')
        else:
            warnings.warn('cannot close solar radiation', UserWarning)

    for variable in ('ghi', 'dif', 'dir', 'dni'):
        if variable in df:
            df[variable] = df[variable].clip(0., np.inf)

    return df
