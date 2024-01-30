
import abc
import json
from pathlib import Path
import copy

from loguru import logger

import numpy as np
import pylab as pl
import pandas as pd

from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

from .egboost import (
    Dataset,
    init_booster_model,
    fit_booster_model
)


logger.disable(__name__)


class BaseModel(metaclass=abc.ABCMeta):

    REQUIRED_TO_PREDICT = ['sza', 'eth', 'ghi', 'ghi_mean', 'ghics', 'difcs']
    REQUIRED_TO_FIT = ['sza', 'eth', 'ghi', 'ghi_mean', 'ghics', 'difcs', 'dif']

    @abc.abstractmethod
    def fit(self, features, target, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, features, **kwargs):
        pass

    @abc.abstractmethod
    def write(self, fname):
        pass

    @classmethod
    @abc.abstractmethod
    def from_file(cls, fname):
        pass


class XGBSplitter(BaseModel):

    def __init__(self, **kwargs):
        kwargs.setdefault('booster', 'gbtree')
        kwargs.setdefault('n_estimators', 100)
        kwargs.setdefault('max_depth', 10)
        kwargs.setdefault('learning_rate', 0.1)
        kwargs.setdefault('objective', 'reg:squarederror')
        kwargs.setdefault('eval_metric', 'rmse')
        self._xgb = init_booster_model(**kwargs)

    @property
    def _feature_names(self):
        return self._xgb.feature_names_in_

    def fit(self, features, target, **kwargs):
        features = features.drop(columns='K', errors='ignore')
        df = pd.concat([features, target.to_frame(name='K')], axis=1)
        data = Dataset(df, features=self._feature_names, target='K')
        test_size = kwargs.pop('test_size', 0.2)
        data.random_split(test_size=test_size)
        fit_booster_model(self._xgb, data, who=f'Fitting {self.__class__.__name__}')

    def predict(self, features, **kwargs):
        K_pred = self._xgb.predict(features[self._feature_names], **kwargs)
        return pd.Series(index=features.index, data=K_pred, name='K')

    def write(self, fname):
        self._xgb.save_model(fname)

    @classmethod
    def from_file(cls, fname):
        xgb = init_booster_model()
        xgb.load_model(fname)
        this = cls()
        this._xgb = copy.deepcopy(xgb)
        return this


class AMPiecewiseModel(BaseModel):
    """
    A AMPiecewiseModel fits/predicts in AM intervals. The target variable
    MUST BE diffuse fraction. It is a pandas series. Features is a pandas
    dataframe that holds the predictors. It can have more predictors than
    the ones that are requried. _basefunc only picks the required ones
    """

    _n_coeffs = None

    def __init__(self, am_bounds=None):
        if am_bounds is None:
            # 12 log intervals from 1 to 10
            self._am_bounds = np.logspace(0., 1., 13)
        else:
            self._am_bounds = np.array(am_bounds)
        self._ai_coeffs = None

    def set_am_bounds(self, am_bounds):
        self._am_bounds = np.array(am_bounds)

    @property
    def am_bounds(self):
        return self._am_bounds

    @property
    def am_centers(self):
        return 0.5 * (self._am_bounds[:-1] + self._am_bounds[1:])

    @property
    def n_am_intervals(self):
        return len(self._am_bounds) - 1

    @property
    def n_coeffs(self):
        return self._n_coeffs

    @property
    def ai_coeffs(self):
        return self._ai_coeffs

    def fit(self, features, target, **kwargs):
        """
        Target must be diffuse fraction, K
        """
        features = features.drop(columns='K', errors='ignore')
        ai_coeffs = np.full((self.n_am_intervals, self._n_coeffs), np.nan)
        for n, am_domain, msg in self._iterate_am_intervals(features['am']):
            fit_result = self._base_fit(
                features[am_domain], target[am_domain], **kwargs)
            ai_coeffs[n] = fit_result.x
            # show the r-square of the fit
            target_pred = self._basefunc(fit_result.x, features[am_domain])
            r2 = np.corrcoef(target[am_domain], target_pred)[0, 1]**2
            logger.debug(f'{msg}, R2: {r2:.4f}')

        self._ai_coeffs = ai_coeffs

    def predict(self, features, **kwargs):
        """
        Return diffuse fraction, K
        """
        kwargs.setdefault('axis', 0)
        kwargs.setdefault('kind', 2)
        kwargs.setdefault('fill_value', 'extrapolate')
        notna = np.all(~np.isnan(self.ai_coeffs), axis=1)
        am_centers = self.am_centers[notna]
        ai_coeffs = self.ai_coeffs[notna]
        ai_interp = interp1d(am_centers, ai_coeffs, **kwargs)(features['am'])
        return self._basefunc(ai_interp.T, features)

    def write(self, fname):

        if self.ai_coeffs is None:
            raise ValueError('this base model cannot be serialized '
                             'because it has not been fitted yet')

        assert Path(fname).suffix == '.json'

        model_data = {
            'am_bounds': self.am_bounds.tolist(),
            'ai_coeffs': self.ai_coeffs.tolist(),
        }
        with Path(fname).open(mode='w', encoding='utf-8') as fh:
            json.dump(model_data, fh)

    @classmethod
    def from_file(cls, fname):
        with Path(fname).open(mode='r', encoding='utf-8') as fh:
            model_data = json.load(fh)
        model = cls(model_data['am_bounds'])
        model._ai_coeffs = np.array(model_data['ai_coeffs'])
        return model

    @staticmethod
    def _basefunc(p, X):
        raise NotImplementedError

    def _base_fit(self, X, y, **kwargs):
        weights = kwargs.pop('weights', np.ones(len(X)))
        kwargs.setdefault('x0', np.ones(self._n_coeffs))
        kwargs.setdefault('method', 'trf')
        kwargs.setdefault('jac', '3-point')
        notna = (np.all(~np.isnan(X), axis=1)
                 & ~np.isnan(y) & ~np.isnan(weights))
        fit_result = least_squares(
            lambda p, X, y, w: (self._basefunc(p, X) - y)*w,
            args=(X[notna], y[notna], weights[notna]), **kwargs
        )
        return fit_result

    def _iterate_am_intervals(self, am):
        am_intervals = zip(self.am_bounds[:-1], self.am_bounds[1:])
        for n_bin, (min_am, max_am) in enumerate(am_intervals):
            am_domain = (am >= min_am) & (am < max_am)
            n_samples = am_domain.sum()
            heading = f'AM range: {min_am:5.2f} <= AM < {max_am:5.2f}:'
            if n_samples < 5:
                logger.warning(f'{heading} {n_samples:6d} samples (< 5)!!')
                continue
            yield n_bin, am_domain, f'{heading} {n_samples:6d} samples'

    def show_diagnostics(self, features, target=None):
        nrows = max(1, int((self.n_am_intervals/3)**0.5))
        ncols = 3*nrows

        fig, axes = pl.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        fig.canvas.manager.set_window_title(f'Diagnostics in {self.__class__.__name__}')

        if 'Kt' in features:
            Kt = features['Kt']
        else:
            Kt = features['ghi'].divide(features['eth']).clip(0.)

        K = target
        if K is None:
            K = features['dif'].divide(features['ghi']).clip(0., 1.)

        K_pred = self.predict(features)

        kwargs = {'gridsize': 100, 'mincnt': 1}

        for n, domain, msg in self._iterate_am_intervals(features['am']):
            j, i = np.unravel_index(n, (nrows, ncols))
            ax = axes[j, i]

            domain_ = domain & K.notna()

            hb = ax.hexbin(Kt[domain_], K[domain_], cmap='copper_r', **kwargs)
            norm = LogNorm(vmin=hb.get_clim()[0], vmax=hb.get_clim()[1])
            ax.hexbin(Kt[domain_], K_pred[domain_], cmap='jet', norm=norm, **kwargs)
            ax.set_title(msg, fontsize=9, y=.99, va='bottom')
            ax.set_xlabel('Clearness index, Kt', fontsize=10)
            ax.set_ylabel('Diffuse fraction, K', fontsize=10)

            residue = K_pred[domain_] - K[domain_]
            mbe, rmse = residue.mean(), (residue**2).mean()**0.5
            r2 = np.corrcoef(K[domain_], K_pred[domain_])[0, 1]**2
            text = f'MBe={mbe:+.2f}  RMSe={rmse:.2f}  R2={r2:.4f}'
            text_kwargs = {'ha': 'left', 'va': 'bottom', 'fontsize': 9}
            ax.text(0.01, 0.01, text, transform=ax.transAxes, **text_kwargs)
            ax.axis([0., 1.25, 0., 1.10])

        fig.tight_layout()
        return fig


class OvercastModel(AMPiecewiseModel):

    _n_coeffs = 2

    def fit(self, features, target, **kwargs):
        K = target
        dif = K * features['ghi']
        super().fit(features, dif, **kwargs)

    def predict(self, features, **kwargs):
        dif_pred = super().predict(features, **kwargs)
        return dif_pred.divide(features['ghi']).clip(0., 1.)

    @staticmethod
    def _basefunc(p, X):
        return p[0] + p[1]*X['ghi']


class ThickCloudsModel(AMPiecewiseModel):

    _n_coeffs = 2

    def fit(self, features, target, **kwargs):
        K = target
        dif = K * features['ghi']
        super().fit(features, dif, **kwargs)

    def predict(self, features, **kwargs):
        dif_pred = super().predict(features, **kwargs)
        return dif_pred.divide(features['ghi']).clip(0., 1.)

    @staticmethod
    def _basefunc(p, X):
        return p[0] + p[1]*X['ghi']


class ScatterCloudsModel(AMPiecewiseModel):

    _n_coeffs = 7

    @staticmethod
    def _basefunc(p, X):
        F = (
            p[1]
            + p[2]*X['Kt']
            + p[3]*X['KT']
            + p[4]*X['Kcs']
            + p[5]*X['Kds']
        )
        return p[0] + (1. - p[0]) / (1. + np.exp(F)) + p[6]*X['Kde']


class ThinCloudsModel(AMPiecewiseModel):

    _n_coeffs = 6

    @staticmethod
    def _basefunc(p, X):
        F = (
            p[0]
            + p[1]*X['Kt']
            + p[2]*X['KT']
            + p[3]*X['Kcs']
            + p[4]*X['Kds']
        )
        return 1. / (1. + np.exp(F)) + p[5]*X['Kde']


class CloudlessModel(AMPiecewiseModel):

    _n_coeffs = 5

    @staticmethod
    def _basefunc(p, X):
        F = (
            p[0]
            + p[1]*X['Kt']
            + p[2]*X['KT']
            + p[3]*X['Kcs']
            + p[4]*X['Kds']
        )
        return 1. / (1. + np.exp(F))


class MostlyClearModel(AMPiecewiseModel):

    _n_coeffs = 6

    @staticmethod
    def _basefunc(p, X):
        F = (
            p[0]
            + p[1]*X['Kt']
            + p[2]*X['KT']
            + p[3]*X['Kcs']
            + p[4]*X['Kds']
        )
        return 1. / (1. + np.exp(F)) + p[5]*X['Kde']


class CloudEnhancementModel(AMPiecewiseModel):

    _n_coeffs = 5

    @staticmethod
    def _basefunc(p, X):
        F = (
            p[0]
            + p[1]*X['Kt']
            + p[2]*X['KT']
            + p[3]*X['Kcs']
            + p[4]*X['Kds']
        )
        return 1. / (1. + np.exp(F))


class AllSkyModel(AMPiecewiseModel):

    _n_coeffs = 7

    @staticmethod
    def _basefunc(p, X):
        F = (
            p[0]
            + p[1]*X['Kt']
            + p[2]*X['KT']
            + p[3]*X['Kcs']
            + p[4]*X['Kds']
            + p[5]*X['Kde']
            + p[6]*X['DKt']
        )
        return 1. / (1. + np.exp(F))
