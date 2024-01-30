
import pandas as pd

from .skyclass import CAELUS_SKY_TYPE_CLASSES as SkyType


def calculate_sky_edges(sky_type):
    sky_type_diff = sky_type.diff()
    sky_type_diff = sky_type_diff.where(sky_type_diff.notna(), 0.)
    sky_edges = (sky_type_diff != 0) & (sky_type >= SkyType.OVERCAST)

    sky_edges_shift = sky_edges.shift(-1)
    sky_edges_shift = sky_edges_shift.where(sky_edges_shift.notna(), False)
    sky_pre_transit = pd.Series(index=sky_type.index, data=float('nan'))
    sky_pre_transit[sky_edges] = sky_type[sky_edges_shift].values

    sky_post_transit = pd.Series(index=sky_type.index, data=float('nan'))
    sky_post_transit[sky_edges] = sky_type[sky_edges].to_numpy()

    edge_transits = []
    for from_sky_t in SkyType.iterate(skip_unknown=True):
        for to_sky_t in SkyType.iterate(skip_unknown=True):
            cond = (
                (sky_pre_transit == from_sky_t) &
                (sky_post_transit == to_sky_t))
            if not cond.sum():
                continue
            edge_transits.append([from_sky_t, to_sky_t, cond])
            # print(f'{from_sky_t.name}->{to_sky_t.name}: {cond.sum()}')
    return sky_edges, edge_transits


def nudge_transitions(series, sky_type, transitions, half_width=5):
    sky_edges, edge_transits = calculate_sky_edges(sky_type)

    edges_to_smooth = []
    for transit in edge_transits:
        if (transit[0], transit[1]) in transitions:
            edges_to_smooth.append(transit[2])

    if len(edges_to_smooth) == 0:
        return series

    edges_to_smooth = pd.concat(
        edges_to_smooth, axis=1).sum(axis=1).astype(bool)

    halo = pd.concat(
        [edges_to_smooth.shift(step)
         for step in range(-half_width, half_width+1, 1)], axis=1
    ).sum(axis=1).astype(bool)

    smooth = series.rolling(f'{2*half_width+1}T', center=True).mean()
    return series.where(~halo, other=smooth.loc[halo])
