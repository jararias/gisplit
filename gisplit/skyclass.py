
import abc
import json
import enum
import copy
import textwrap
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pylab as pl
import pandas as pd
from loguru import logger
from sklearn import metrics

import caelus
from caelus.skytype import SkyType as caelus_skytype

import xgboost
from . import egboost
from .tools import classproperty
from .tools.filters import (
    drop_site_from_index,
    add_site_to_index,
    rolling_mean,
    rolling_sum
)


logger.disable(__name__)


#######################################################################
#  The enum sky types must have members > 0, i.e., 0 is not allowed.  #
#  The value 1 is reserved for the UNKNOWN sky type. The sky classes  #
#  take values starting at 2 !!                                       #
#######################################################################

class BaseSkyType(enum.IntEnum):
    '''
    Base class to construct enumerated sky types. For instance:

      SkyType = BaseSkyType('SkyType', {'UNKNOWN': 1, 'OVERCAST': 2})

    Optionally, a short name can be provided too:

      SkyType = BaseSkyType('SkyType', {'UNKNOWN': 1, 'OVERCAST': (2, 'Ov')})

    In this last case, SkyType.UNKNOWN.short_name is 'UNKNOWN' and
    SkyType.OVERCAST.short_name is 'Ov'. Moreover: SkyType.UNKNOWN.name is
    'UNKNOWN', SkyType.UNKNOWN.value is 1, SkyType.OVERCAST.name is 'OVERCAST'
    and SkyType.OVERCAST.value is 2
    '''

    def __new__(cls, value, short_name=None):
        obj = int.__new__(cls)
        try:
            value, short_name = value
        except TypeError:
            pass
        obj._value_ = value
        obj._short_name_ = short_name
        return obj

    @property
    def short_name(self):
        return self._short_name_ or self.name

    @classmethod
    def _missing_(cls, value):
        # pylint: disable=no-member
        return cls.UNKNOWN

    @classmethod
    def iterate(cls, skip_unknown=True):
        """
        Iterate over the class members, but skip the UNKNOWN type
        """
        # pylint: disable=no-member
        for sky_type in cls:
            if sky_type is cls.UNKNOWN and skip_unknown is True:
                continue
            yield sky_type

    @classmethod
    def as_dict(cls):
        return {sky_type.name: (sky_type.value, sky_type.short_name)
                for sky_type in cls}


CAELUS_SKY_TYPE_CLASSES = BaseSkyType(
    'CaelusSkyTypeClasses',
    {
        'UNKNOWN': caelus_skytype.UNKNOWN.value,
        'OVERCAST': caelus_skytype.OVERCAST.value,
        'THICK_CLOUDS': (caelus_skytype.THICK_CLOUDS.value, 'THICK'),
        'SCATTER_CLOUDS': (caelus_skytype.SCATTER_CLOUDS.value, 'SCATTER'),
        'THIN_CLOUDS': (caelus_skytype.THIN_CLOUDS.value, 'THIN'),
        'CLOUDLESS': caelus_skytype.CLOUDLESS.value,
        'CLOUD_ENHANCEMENT': (caelus_skytype.CLOUD_ENHANCEMENT.value, 'CLOUDEN')
    }
)


class BaseSkyClassifier(metaclass=abc.ABCMeta):

    @classproperty
    @classmethod
    @abc.abstractmethod
    def REQUIRED_TO_PREDICT(cls):
        pass

    @classproperty
    @classmethod
    @abc.abstractmethod
    def REQUIRED_TO_FIT(cls):
        pass

    @abc.abstractmethod
    def predict(self, data, **kwargs):
        pass

    @abc.abstractmethod
    def write(self, file_name):
        pass


class CAELUSClassifier(BaseSkyClassifier):
    _REQUIRED_TO_PREDICT = caelus.REQUIRED_TO_CLASSIFY
    _REQUIRED_TO_FIT = caelus.REQUIRED_TO_CLASSIFY

    @classproperty
    @classmethod
    def REQUIRED_TO_PREDICT(cls):
        return cls._REQUIRED_TO_PREDICT

    @classproperty
    @classmethod
    def REQUIRED_TO_FIT(cls):
        return cls._REQUIRED_TO_FIT

    def __init__(self):
        self._time_step = '1min'
        self._sky_type = CAELUS_SKY_TYPE_CLASSES

    def predict(self, data, **kwargs):
        return caelus.classify(data, **kwargs)

    def write(self, file_name):
        assert Path(file_name).suffix == '.zip'
        metadata = {
            'time_step': self._time_step,
            'sky_class': (self._sky_type.__name__, self._sky_type.as_dict())
        }
        metadata_file_name = Path(tempfile.NamedTemporaryFile().name)
        with ZipFile(file_name, mode='w', compression=ZIP_DEFLATED) as zipf:
            with open(metadata_file_name, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            zipf.write(metadata_file_name, arcname='metadata.json')
            zipf.writestr(Path(file_name).with_suffix('.json').name, json.dumps({}))
        metadata_file_name.unlink()

    @classmethod
    def from_file(cls, file_name):
        assert Path(file_name).suffix == '.zip'
        return cls()
