import copy
import json
import zipfile
import fnmatch
import tempfile
import warnings
import importlib
from pathlib import Path

from loguru import logger

import numpy as np
import pylab as pl
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.exceptions import NotFittedError

from . import skyclass
from .skyclass import (
    BaseSkyType,
    BaseSkyClassifier
)
from . import base_models
from .base_models import (
    BaseModel,
    AMPiecewiseModel,
    XGBSplitter
)
# from .tools import classproperty
from .tools.filters import (
    rolling_mean,
    add_site_to_index,
    drop_site_from_index,
    guess_time_step
)

from .nudge import nudge_transitions


logger.disable(__name__)


def assert_required(columns, data):
    if missing := set(columns).difference(data.columns):
        raise ValueError(f'missing required columns {list(missing)}')


def safe_multisite(func, df, **kwargs):
    if 'site' in df.index.names:
        sites = df.index.get_level_values('site')
        grouper = drop_site_from_index(df).groupby(sites)
        return add_site_to_index(
            pd.concat([func(subset, **kwargs) for _, subset in grouper], axis=0), sites)
    return func(df, **kwargs)


class GISPLIT:

    def __init__(self, engine='xgb', climate=None, **kwargs):
        """
        Creates a GISPLIT instance

        Parameters:
        -----------
        engine: {'xgb', 'reg'}
          'xgb' to use extreme gradient boosting GHI splitters, or
          'reg' to use conventional regression models
        kwargs:
          extra arguments not important for a regular use of the model
        """

        # default configuration values...

        self._metadata = {}
        self._sky_classes = {}
        self._sky_type = BaseSkyType('SkyType', {})
        self._sky_classifier = None
        self._splitting_models = {}
        self._splitter_specs = {
            'ghi_mean_rolling_window': kwargs.pop('ghi_mean_rolling_window', '30min')
        }

        time_step = '1min'

        assert engine in ('xgb', 'reg')
        self._engine = engine

        if climate is not None:
            assert climate in 'ABCDE'
        self._climate = climate

        if time_step is None:
            logger.debug('returning an empty GISPLIT instance')
            return None

        # determine parameters file from input arguments...

        def get_param_file_name(engine=None, climate=None):
            engine_label = engine or self._engine
            clim_label = '' if climate is None else f'_kg{climate}'
            file_name = f'gisplit_for_obs_01min_caelus_{engine_label}{clim_label}.zip'
            return file_name

        parameters_file_name = (
            self.get_param_dir() /
            get_param_file_name(self._engine, self._climate))

        if not parameters_file_name.exists():
            warnings.warn(f'missing parameters file `{parameters_file_name.absolute()}`', UserWarning)
            logger.warning('returning an empty GISPLIT instance')
            return None

        self._param_file_name = parameters_file_name
        logger.debug(f'Parameters file name: {parameters_file_name.absolute()}')

        # read parameters file and initialize object...
        with zipfile.ZipFile(parameters_file_name, 'r') as zip_file:
            members = zip_file.namelist()

            # read `metadata.json`, if it exists
            if 'metadata.json' in members:
                self._metadata = json.loads(zip_file.read('metadata.json'))

            if 'parameters/splitter/specs.json' in members:
                self._splitter_specs = json.loads(
                    zip_file.read('parameters/splitter/specs.json'))

            # classifier model...
            if len(clf_file_name := fnmatch.filter(members, r'parameters/classifier/*.zip')) > 0:
                clf_file_name = Path(clf_file_name[0])
                clf_constructor = getattr(skyclass, clf_file_name.stem)
                if issubclass(clf_constructor.__class__, skyclass.BaseSkyClassifier.__class__):
                    with tempfile.TemporaryDirectory() as temp_dirname:
                        zip_file.extract(clf_file_name.as_posix(), path=temp_dirname)
                        self._sky_classifier = clf_constructor.from_file(
                            Path(temp_dirname) / clf_file_name)
                else:
                    cls_name = clf_constructor.__class__.__name__
                    warnings.warn(f'unexpected instance of {cls_name}', UserWarning)
                    logger.warning('returning a GISPLIT instance without classifier!!')

            assert self._sky_classifier._time_step == self._metadata.get('time_step')

            self._sky_type = self._sky_classifier._sky_type

            # GHI separation model...
            for sky_type in self._sky_type.iterate(skip_unknown=True):
                if (sky_file_name := f'parameters/splitter/{sky_type.name}.zip') not in members:
                    warnings.warn(f'missing sky-type parameters file {sky_file_name}', UserWarning)
                    logger.warning('returning a GISPLIT instance without GHI separation model!!')
                    break

                try:
                    with tempfile.TemporaryDirectory() as temp_dirname:
                        zip_file.extract(sky_file_name, path=temp_dirname)
                        sm = SplittingModel.from_file(Path(temp_dirname) / sky_file_name, sky_type)
                        self._splitting_models[sky_type.name] = copy.deepcopy(sm)
                except KeyError as exc:
                    raise ValueError(f'{exc.args[0]} {parameters_file_name}') from exc

    @staticmethod
    def get_param_dir():
        # param_dir = Path(__file__).parent / 'parameters/'
        # param_dir = importlib.resources.path('gisplit', 'parameters')
        try:
            param_dir = importlib.resources.files('gisplit.parameters').joinpath('')
        except StopIteration:
            # to prevent an apparent regression problem with MultiPlexedPath
            param_dir = importlib.resources.files('gisplit.parameters')._paths[0]
        if not param_dir.exists():
            raise ValueError(f'missing parameters directory `{param_dir.absolute()}`')
        return param_dir

    @property
    def param_file_name(self):
        return self._param_file_name

    @property
    def description(self):
        return self._metadata.get('description', 'unknown')

    @property
    def sky_type(self):
        return self._sky_type

    @property
    def metadata(self):
        return self._metadata

    @property
    def engine(self):
        return self._engine

    @property
    def time_step(self):
        return self._metadata.get('time_step', self._metadata.get('time_step', 'unknown'))

    @property
    def sky_classifier(self):
        return self._sky_classifier

    def get_splitting_model(self, sky_type):
        return self._splitting_models.get(sky_type.name, None)

    def set_metadata(self, metadata):
        self._metadata = dict(metadata)

    def set_sky_type(self, sky_type):
        self._sky_classes = sky_type.as_dict()
        self._sky_type = BaseSkyType('SkyType', self._sky_classes)

    def set_sky_classifier(self, sky_classifier):
        actual = sky_classifier.__class__
        if not issubclass(actual, BaseSkyClassifier):
            raise ValueError(f'expected a `BaseSkyClassifier`. Got {actual.__name__}')

        self._sky_classifier = sky_classifier
        self.set_sky_type(self.sky_classifier.sky_type)
        time_step = self.sky_classifier._time_step
        self._metadata['time_step'] = time_step
        if 'time_step' in self._metadata:
            self._metadata['time_step'] = time_step

    def set_splitting_model(self, **kwargs):
        for sky_type_name, base_model in kwargs.items():
            actual = base_model.__class__
            if not issubclass(actual, BaseModel):
                raise ValueError(f'expected a `BaseModel`. Got {actual.__name__}')
            sky_type = self.sky_type[sky_type_name]
            sm = SplittingModel(base_model, sky_type)
            self._splitting_models[sky_type.name] = copy.deepcopy(sm)

    def update_metadata(self, metadata):
        self._metadata.update(metadata)

    def _check_and_prepare_input_data(self, data, sky_type_or_func=None, **kwargs):

        # A MULTISITE DATAFRAME MUST HAVE MULTIINDEX WITH TWO LEVELS
        # WITH NAMES (times_utc, site) !!!

        # check for MUST-HAVE data
        assert_required(kwargs.pop('required', []), data)
        data_ready = data.copy()

        # calculate ghi_mean, if it is not already in the input data. It is used
        # in the splitter model, so it is needed by the SplittingModel instances
        if 'ghi_mean' not in data_ready:
            window = self._splitter_specs.get('ghi_mean_rolling_window')
            data_ready['ghi_mean'] = safe_multisite(
                lambda df: rolling_mean(df, window=window), data_ready['ghi'])

        # check/calculate sky-type !!
        if sky_type_or_func is None:
            data_ready['sky_type'] = safe_multisite(self.sky_classifier.predict, data_ready, **kwargs)
        elif callable(sky_type_or_func) is True:
            data_ready['sky_type'] = safe_multisite(sky_type_or_func, data_ready, **kwargs)
        elif isinstance(sky_type_or_func, str):
            if sky_type_or_func in data_ready.columns:
                data_ready['sky_type'] = data_ready[sky_type_or_func]
        else:
            data_ready['sky_type'] = sky_type_or_func

        return data_ready

    def fit(self, data, sky_type_or_func=None, **kwargs):
        """
        `data must be a continuous monotonic increasing time series !!!
        This is required to compute indices from a moving window, such as KT
        (which requires `ghi_mean`) for the splitting model or KC and KV if,
        eventually, the sky-type classification is performed also here
        """
        # prepare the required data to compute all features
        fit_data = self._check_and_prepare_input_data(data, sky_type_or_func, **kwargs)

        if 'site' in fit_data.index.names:
            fit_data = drop_site_from_index(fit_data)

        for sky_t in self.sky_type.iterate(skip_unknown=True):
            domain = fit_data['sky_type'] == sky_t.value
            if domain.sum() == 0:
                raise ValueError(f'missing data to fit {sky_t.name} sky type')
            self.get_splitting_model(sky_t).fit(fit_data[domain])

    def predict(self, data, sky_type_or_func=None, nudge_1min=True, nudge_half_width=7, **kwargs):
        """
        `data must be a continuous monotonic increasing time series !!!
        This is required to compute indices from a moving window, such
        as KT (which requires `ghi_mean`) for the splitting model or KC
        and KV if, eventually, the sky-type classification is performed
        also here
        """
        input_time_step = guess_time_step(pd.DataFrame(data).pipe(drop_site_from_index))
        if pd.Timedelta(self.time_step) != input_time_step:
            message = f'expected {self.time_step} time step. Got {input_time_step}'
            warnings.warn(message, UserWarning)
            logger.warning(message)

        # prepare the required data to compute all features
        datain = self._check_and_prepare_input_data(data, sky_type_or_func, **kwargs)

        def splitter(df):
            pred_ = pd.DataFrame(index=df.index, columns=['dif', 'dir', 'dni'], dtype='f4')
            for sky_t in self.sky_type.iterate(skip_unknown=True):
                domain = df['sky_type'] == sky_t.value
                if domain.sum() == 0:
                    continue
                logger.debug(f'Predicting {sky_t.name} sky: {domain.sum()} instances')
                pred_.loc[domain] = (self.get_splitting_model(sky_t)
                                     .predict(df.loc[domain]).astype(float))
            return pred_

        pred = safe_multisite(splitter, datain)

        if (nudge_1min is True) and (self.time_step == '1min'):
            sky_transitions = [
                (self.sky_type.SCATTER_CLOUDS, self.sky_type.THIN_CLOUDS),
                (self.sky_type.THIN_CLOUDS, self.sky_type.SCATTER_CLOUDS)
            ]

            kwargs = dict(transitions=sky_transitions, half_width=nudge_half_width)
            dif = nudge_transitions(pred['dif'], datain['sky_type'], **kwargs)
            pred['dif'] = dif.clip(0., datain['ghi'])
            pred['dir'] = datain['ghi'].sub(pred['dif']).clip(0., datain['ghi'])

        return pred

    def write(self, file_name):
        # pylint: disable=protected-access

        if self.sky_classifier is None:
            raise ValueError('missing sky classifier')

        for sky_t in self.sky_type.iterate(skip_unknown=True):
            base_model = self.get_splitting_model(sky_t).base_model

            actual = base_model.__class__
            if not issubclass(actual, BaseModel):
                raise ValueError(f'expected a `BaseModel`, but got {actual.__name__}')

            if issubclass(actual, AMPiecewiseModel):
                if base_model.ai_coeffs is None:
                    raise ValueError(f'unfitted `BaseModel` of type {actual.__name__}')

            if isinstance(base_model, XGBSplitter):
                try:
                    base_model._xgb.get_booster()
                except NotFittedError as exc:
                    raise ValueError(f'unfitted `BaseModel` of type {actual.__name__}') from exc

        with zipfile.ZipFile(file_name, 'w') as zipf:
            zipf.writestr('metadata.json', json.dumps(self.metadata))
            # zipf.writestr('sky_classes.json', json.dumps(self._sky_classes))

            clf = self.sky_classifier
            clf_specs = {
                'required_to_fit': clf.REQUIRED_TO_FIT,
                'required_to_predict': clf.REQUIRED_TO_PREDICT}
            zipf.writestr('parameters/classifier/specs.json', json.dumps(clf_specs))
            with tempfile.TemporaryDirectory() as temp_dirname:
                clf_file_name = Path(temp_dirname) / f'{clf.__class__.__name__}.zip'
                clf.write(clf_file_name)
                zipf.write(clf_file_name, arcname='parameters/classifier/' + clf_file_name.name)

            spl_specs = {
                'required_to_fit': BaseModel.REQUIRED_TO_FIT,
                'required_to_predict': BaseModel.REQUIRED_TO_PREDICT}
            spl_specs = spl_specs | self._splitter_specs
            zipf.writestr('parameters/splitter/specs.json', json.dumps(spl_specs))
            for sky_t in self.sky_type.iterate(skip_unknown=True):
                with tempfile.TemporaryDirectory() as temp_dirname:
                    sky_type_file_name = Path(temp_dirname) / f'{sky_t.name}.zip'
                    self.get_splitting_model(sky_t).write(sky_type_file_name)
                    arcname = 'parameters/splitter/' + sky_type_file_name.name
                    zipf.write(sky_type_file_name, arcname=arcname)

    def show_diagnostics(self, data, sky_type=None):

        if 'sky_type' not in data:
            raise ValueError('missing `sky_type` in data !!')

        datain = data.query('sza < 85').dropna()

        if sky_type is None:
            fig = self._scatterplots_by_sky_type(datain)
            return fig

        if (sm := self.get_splitting_model(sky_type)) is None:
            logger.warning(f'missing SplittingModel for {sky_type.name} sky')
            return None

        if 'sky_type' in datain:
            fig = sm.show_diagnostics(datain[datain['sky_type'] == sky_type])
        else:
            fig = sm.show_diagnostics(datain)

        title = fig.canvas.manager.get_window_title()
        fig.canvas.manager.set_window_title(f'{title}  Sky type: {sky_type.name}')
        return fig

    def _scatterplots_by_sky_type(self, data):

        assert_required(['sza', 'eth', 'ghi', 'dif', 'sky_type'], data)

        def equalize_limits(ax, min_vmin=None, max_vmax=None, ooline=True, **kwargs):
            bounds = zip(ax.get_xlim(), ax.get_ylim())
            vmin, vmax = [func(*ags) for func, ags in zip((min, max), bounds)]
            vmin = vmin if min_vmin is None else max(min_vmin, vmin)
            vmax = vmax if max_vmax is None else min(max_vmax, vmax)
            if ooline is True:
                default_kwargs = {'ls': '-', 'lw': 1, 'color': 'k', 'marker': ''}
                ax.plot([vmin, vmax], [vmin, vmax], **(default_kwargs | kwargs))
            ax.axis([vmin, vmax, vmin, vmax])
            return vmin, vmax

        def metrics(pred_series, obs_series):
            residue = pred_series - obs_series
            domain = residue.notna()
            mobs = obs_series.loc[domain].mean()
            residue = residue.loc[domain]
            mbe, mae = residue.mean(), residue.abs().mean()
            rmse = residue.pow(2).mean()**0.5
            return f'MBe={mbe/mobs:+.1%} MAe={mae/mobs:.1%} RMSe={rmse/mobs:.1%}'

        pred = self.predict(data)
        df = pd.DataFrame().assign(
            ghi_obs=data['ghi'],
            dif_obs=data['dif'],
            dir_obs=data['ghi'].sub(data['dif']).clip(0., data['ghi']),
            dni_obs=data.eval('(ghi-dif)/cos(0.017453292519943295*sza)').clip(0.),
            dif_pred=pred['dif'],
            dir_pred=pred['dir'],
            dni_pred=pred['dni'],
            Kt=data['ghi'].divide(data['eth']).where(data['eth'] > 0, np.nan).clip(0., 1.2),
            K_obs=data['dif'].divide(data['ghi']).where(data['ghi'] > 0, np.nan).clip(0, 1),
            K_pred=pred['dif'].divide(data['ghi']).where(data['ghi'] > 0, np.nan).clip(0, 1)
        )

        fig, axes = pl.subplots(4, 5, figsize=(16.5, 12.1), constrained_layout=True)
        fig.canvas.manager.set_window_title(f'Diagnostics in {self.__class__.__name__}')

        text_kwargs = {'ha': 'center', 'va': 'bottom', 'fontsize': 8}
        hexbin_kwargs = {'gridsize': 150, 'mincnt': 1, 'cmap': 'jet', 'norm': LogNorm()}

        def iterate_sky_types(data):
            # the purpose of this iterator is to yield the UNKNOWN sky class in the
            # last iteration, and use it to represent all sky classes combined
            n_column = 0
            for sky_t in self.sky_type.iterate(skip_unknown=True):
                yield n_column, sky_t, data['sky_type'] == sky_t
                n_column = n_column + 1
            yield n_column, self.sky_type.UNKNOWN, data['sky_type'] > self.sky_type.UNKNOWN

        for n_column, sky_type, domain in iterate_sky_types(data):

            sky_type_name = sky_type.name
            if sky_type == self.sky_type.UNKNOWN:
                sky_type_name = 'ALL SKY TYPES COMBINED'

            for n_var, variable in enumerate(('dni', 'dir', 'dif')):
                ax = axes[n_var, n_column]
                obs, pred = f'{variable}_obs', f'{variable}_pred'
                ax.hexbin(obs, pred, data=df.dropna(), **(hexbin_kwargs | {'cmap': 'copper_r'}))
                ax.hexbin(obs, pred, data=df.loc[domain].dropna(), **hexbin_kwargs)
                equalize_limits(ax, min_vmin=0)
                ax.set_xlabel(f'Observed {variable.upper()} (W/m$^2$)', fontsize=12)
                ax.set_ylabel(f'Predicted {variable.upper()} (W/m$^2$)', fontsize=12)
                ax.set_title(sky_type_name if n_var == 0 else None, fontsize=12, y=1.07, va='bottom')
                text = metrics(df.loc[domain, obs], df.loc[domain, pred])
                ax.text(0.5, 1.01, text, transform=ax.transAxes, **text_kwargs)

            ax = axes[3, n_column]
            this_df = df.loc[domain].dropna()
            ax.hexbin('Kt', 'K_obs', data=this_df, **(hexbin_kwargs | {'cmap': 'copper_r'}))
            ax.hexbin('Kt', 'K_pred', data=this_df, **hexbin_kwargs)
            equalize_limits(ax, min_vmin=0, ooline=False)
            ax.set_xlabel('Clearness index, Kt', fontsize=12)
            ax.set_ylabel('Predicted diffuse fraction, K', fontsize=12)
            ax.grid()

        return fig


class SplittingModel:

    # NOTE: `ghi_mean` is required here because it must be computed from a
    # continuous monotonic ghi time series. However, such time series is a
    # sucession of sky type classes, while the class SplittingModel is only
    # for a single sky class. Hence, the data being managed by this class
    # are not "continuous". They involve multiple patches of a single sky
    # class (overcast, scatterclouds...)

    REQUIRED_TO_PREDICT = ['sza', 'eth', 'ghi', 'ghi_mean', 'ghics', 'difcs']
    REQUIRED_TO_FIT = REQUIRED_TO_PREDICT + ['dif']

    def __init__(self, base_model=None, sky_type=None):
        if base_model is not None:
            assert isinstance(base_model, BaseModel)
        self._base_model = base_model
        self._sky_type = sky_type
        # self._sky_type = SkyType(sky_type)
        # if self._sky_type == SkyType.UNKNOWN:
        #     warnings.warn('unknown sky type in splitting model', UserWarning)

    @property
    def sky_type(self):
        return self._sky_type

    @property
    def base_model(self):
        return self._base_model

    def set_base_model(self, base_model):
        if not isinstance(base_model, BaseModel):
            base_model_class = base_model.__class__.__name__
            raise ValueError(f'expected a BaseModel. Got {base_model_class}')
        self._base_model = copy.deepcopy(base_model)

    @staticmethod
    def calculate_features(data):

        def optical_airmass(data):
            sza = data['sza']
            cosz = np.cos(np.radians(sza))
            Da = np.maximum(1e-4, 96.741 - sza)
            rec_am = cosz + 0.48353*(sza**0.095846)*(Da**(-1.754))
            return np.where(sza <= 90., 1. / rec_am, np.nan)

        Kt = data.eval("""ghi/eth""").clip(0., 1.15)

        return pd.DataFrame().assign(
            ghi=data.ghi,
            am=optical_airmass(data),
            Kt=Kt,
            DKt=Kt - Kt.shift(1),
            KT=data.eval("""ghi_mean/eth""").clip(0., 1.15),
            Kcs=data.eval("""ghi/ghics""").clip(0.),
            Kds=data.eval("""difcs/ghics""").clip(0., 1.),
            Kde=data.eval("""(ghi-ghics)/ghi""").clip(0.)
        )

    def fit(self, data, **kwargs):
        assert_required(self.REQUIRED_TO_FIT, data)

        datain = data.query('sza < 85').dropna()

        if 'sky_type' in datain and self.sky_type > 1:
            datain = datain.loc[datain['sky_type'] == self.sky_type]

        logger.info(f'Fitting splitting model for {self.sky_type.name} sky ({len(datain)} samples)')
        logger.info(f'  Base model: {self.base_model.__class__.__name__}')
        features = self.calculate_features(datain[self.REQUIRED_TO_FIT])
        K = datain['dif'].divide(datain['ghi']).clip(0., 1.)
        notna = features.notna().all(axis=1) & K.notna()
        self.base_model.fit(features[notna], K[notna], **kwargs)

        # fitting metrics...
        obs = datain.eval(
            """
            dif=dif
            dir=ghi-dif
            dni=(ghi-dif)/cos(0.017453292519943295*sza)
            """
        )
        pred = self.predict(datain)
        logger.debug('  In-sample metrics:')
        for variable in pred.columns:
            residue = pred[variable] - obs[variable]
            notna = residue.notna() & (datain['sza'] < 85.)
            mobs = obs.loc[notna, variable].mean()
            mbe, rmse = residue.mean(), residue.pow(2).mean()**0.5
            r2 = np.corrcoef(pred.loc[notna, variable], obs.loc[notna, variable])[0, 1]**2
            logger.debug(f'    {variable.upper()}: mbe={mbe: .1f}_W/m2 ({mbe/mobs: .1%}) '
                         f'rmse={rmse:4.1f}_W/m2 ({rmse/mobs:5.1%}) R2={r2:5.3f}')

    def predict(self, data, **kwargs):
        assert_required(self.REQUIRED_TO_PREDICT, data)
        features = self.calculate_features(data[self.REQUIRED_TO_PREDICT])
        K_pred = self.base_model.predict(features, **kwargs)
        dif_pred = K_pred.mul(data['ghi']).clip(0., data['ghi'])
        dir_pred = data['ghi'].sub(dif_pred).clip(0., data['ghi'])
        dni_pred = dir_pred.divide(np.cos(np.radians(data['sza']))).clip(0.)
        return pd.DataFrame().assign(dif=dif_pred, dir=dir_pred, dni=dni_pred)

    def write(self, fname):

        if hasattr(self.base_model, 'ai_coeffs'):
            if self.base_model.ai_coeffs is None:
                raise ValueError('cannot be serialized: base model not fitted')

        base_model_class_name = self.base_model.__class__.__name__
        base_model_fname = f'__{base_model_class_name}.json'
        self.base_model.write(base_model_fname)

        with zipfile.ZipFile(fname, 'w') as zipf:
            arcname = f'base_model/{base_model_fname.lstrip("_")}'
            zipf.write(base_model_fname, arcname)
            Path(base_model_fname).unlink()

    @classmethod
    def from_file(cls, fname, sky_type=None):

        with zipfile.ZipFile(fname, 'r') as zipf:

            # read base model
            base_model_dir = zipfile.Path(zipf).joinpath('base_model')
            if not base_model_dir.exists():
                raise ValueError('missing base model')

            # extract base model file(s)
            base_model_members = []
            for member in base_model_dir.iterdir():
                if member.is_file():
                    base_model_members.append(member.name)
                    with open(member.name, 'wb') as f:
                        f.write(member.read_bytes())

            # the base model use only 1 file. In the future, it might change...
            base_model_fname = base_model_members[0]
            base_model_cls_name = Path(base_model_fname).stem  # no file ext...
            base_model_cls = getattr(base_models, base_model_cls_name)
            base_model = base_model_cls.from_file(base_model_fname)
            for member in base_model_members:
                Path(member).unlink()

        # construct the splitting model from the just read files
        splitting_model = cls(base_model=base_model, sky_type=sky_type)

        return splitting_model

    def show_base_model_diagnostics(self, data):
        datain = data.query('sza < 85').dropna()
        if 'sky_type' in datain and self.sky_type > 1:
            datain = datain.loc[datain['sky_type'] == self.sky_type]
        features = self.calculate_features(datain)
        K = datain['dif'].divide(datain['ghi']).clip(0., 1.)
        fig = self.base_model.show_diagnostics(features, K)
        return fig

    def show_diagnostics(self, data):

        assert_required(['eth', 'ghi', 'dif'], data)

        datain = data.query('sza < 85').dropna()
        if 'sky_type' in datain and self.sky_type > 1:
            datain = datain.loc[datain['sky_type'] == self.sky_type]

        def equalize_scatterplot(ax):
            vmin = min(ax.get_xlim()[0], ax.get_ylim()[0])
            vmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([vmin, vmax], [vmin, vmax], 'k-')
            ax.axis([vmin, vmax, vmin, vmax])
            ax.grid()

        def metrics(pred_series, obs_series):
            residue = pred_series - obs_series
            notna = residue.notna()
            mobs = obs_series.loc[notna].mean()
            residue = residue.loc[notna]
            mbe, mae = residue.mean(), residue.abs().mean()
            rmse = residue.pow(2).mean()**0.5
            r2 = np.corrcoef(pred_series.loc[notna], obs_series.loc[notna])[0, 1]**2
            return f'MBe={mbe/mobs:+.1%} MAe={mae/mobs:.1%} RMSe={rmse/mobs:.1%} R2={r2:5.3f}'

        pred = self.predict(datain)
        df = pd.DataFrame().assign(
            dif_obs=datain['dif'],
            dir_obs=datain['ghi'].sub(datain['dif']).clip(0., datain['ghi']),
            dni_obs=datain.eval('(ghi-dif)/cos(0.017453292519943295*sza)').clip(0.),
            dif_pred=pred['dif'],
            dir_pred=pred['dir'],
            dni_pred=pred['dni'],
            Kt=datain['ghi'].divide(datain['eth']).clip(0., 1.2),
            K_obs=datain['dif'].divide(datain['ghi']).clip(0., 1.),
            K_pred=pred['dif'].divide(datain['ghi']).clip(0., 1.)
        )

        fig, axes = pl.subplots(1, 4, figsize=(16.5, 4))

        title = f'Diagnostics in {self.__class__.__name__}[{self.sky_type.name}]'
        fig.canvas.manager.set_window_title(title)

        hexbin_kwargs = {'gridsize': 150, 'mincnt': 1, 'norm': LogNorm()}
        text_kwargs = {'ha': 'left', 'va': 'bottom', 'fontsize': 8}

        for n_var, variable in enumerate(('dni', 'dir', 'dif')):
            ax = axes[n_var]
            obs, pred = f'{variable}_obs', f'{variable}_pred'
            ax.hexbin(obs, pred, data=df, cmap='jet', **hexbin_kwargs)
            ax.set_xlabel(f'Observed {variable.upper()} (W/m$^2$)', fontsize=12)
            ax.set_ylabel(f'Predicted {variable.upper()} (W/m$^2$)', fontsize=12)
            text = metrics(df[pred], df[obs])
            ax.text(0.01, 1.01, text, transform=ax.transAxes, **text_kwargs)
            equalize_scatterplot(ax)

        ax = axes[3]
        ax.hexbin('Kt', 'K_obs', data=df, cmap='copper_r', **hexbin_kwargs)
        ax.hexbin('Kt', 'K_pred', data=df, cmap='jet', **hexbin_kwargs)
        ax.set_xlabel('Clearness index, Kt', fontsize=12)
        ax.set_ylabel('Diffuse fraction, K', fontsize=12)
        ax.axis([0., 1.25, 0., 1.10])
        ax.grid()

        fig.tight_layout(w_pad=1.5)
        return fig
