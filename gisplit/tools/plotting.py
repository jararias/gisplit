
from datetime import datetime

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib as mpl


def stretch_xlim(ax, margin):
    try:
        start_time = np.datetime64(int(ax.get_xlim()[0]//1), 'D')
        end_time = np.datetime64(int(ax.get_xlim()[1]//1), 'D')
        index, values = ax.lines[0].get_data()
        domain = (index >= start_time) & (index < end_time)
        start_time, end_time = index[domain & ~np.isnan(values)][[0, -1]]
        ax.set_xlim(start_time - margin, end_time + margin)
    except IndexError:
        return


def onscroll_daily_step(event):
    # ax = event.inaxes
    ax = event.canvas.figure.axes[0]
    one_day = np.timedelta64(1, 'D')
    cur_start_time = np.datetime64(int(ax.get_xlim()[0]//1), 'D')
    start_time = cur_start_time - event.step * one_day
    ax.set_xlim(start_time, start_time + one_day)
    stretch_xlim(ax, margin=np.timedelta64(1, 'h'))
    ax.autoscale_view(tight=True, scalex=False)
    event.canvas.draw_idle()


def datemap(variable, data, rc_params=None, colorbar=None, ax=None, lst=True, longitude=None, **kwargs):

    if lst is True and longitude is None:
        raise ValueError('`longitude` is required to convert to local solar time')

    delta_hours = longitude // 15 if lst is True else 0
    df = data.set_index(data.index + pd.Timedelta(int(delta_hours), 'H'))

    index = pd.MultiIndex.from_arrays([df.index.date, df.index.time], names=['date', 'time'])
    df = df.set_index(index).unstack(level=0)

    x = df[variable].columns
    y = df[variable].index.map(lambda t: datetime.combine(datetime.today(), t))
    z = df[variable].to_numpy()

    default_rc_params = {
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large'
    }

    title = kwargs.pop('title', None)
    default_kwargs = {'cmap': 'jet', 'shading': 'auto'}

    with mpl.rc_context(default_rc_params | (rc_params or {})):
        if ax is None:
            _, ax = pl.subplots(1, 1, figsize=(18, 6), constrained_layout=True)
        pc = ax.pcolormesh(x, y, z, **(default_kwargs | (kwargs or {})))
        if colorbar is not None:
            pl.colorbar(pc, **colorbar)
        ax.xaxis_date()
        ax.yaxis_date()
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d-%m-%Y'))
        ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        ax.set(xlabel='Date', ylabel='Time (LST)' if lst is True else 'Time', title=title or variable)

    return ax
