
from loguru import logger

import xgboost
from tqdm import tqdm
from sklearn.model_selection import train_test_split


logger.disable(__name__)


class Dataset:
    def __init__(self, df, features, target):
        self._data = df.copy()
        self._features = list(features)
        self._target = target

        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_y = None

        for feature in self._features:
            if feature not in self._data:
                raise ValueError(f'unknown feature {feature}')

        if self._target not in self._data:
            raise ValueError(f'unknown target {target}')

    def random_split(self, **kwargs):
        # kwargs = test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None
        tts = train_test_split(
            self._data[self._features], self._data[self._target], **kwargs)
        self._train_x, self._test_x, self._train_y, self._test_y = tts

    def explicit_split(self, train_mask, test_mask):
        self._train_x = self._data[self._features][train_mask]
        self._test_x = self._data[self._features][test_mask]
        self._train_y = self._data[self._target][train_mask]
        self._test_y = self._data[self._target][test_mask]

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_y(self):
        return self._train_y

    @property
    def test_x(self):
        return self._test_x

    @property
    def test_y(self):
        return self._test_y


def init_booster_model(kind='regressor', **kwargs):
    assert kind in ('regressor', 'classifier')
    # Regression trees (a.k.a. boosters):
    #  - Options: `gbtree` (non-linear), `gblinear` and `dart`
    kwargs.setdefault('booster', 'gbtree')
    # Optimization:
    #  - for multiclass classification problems,
    #    objective=`multi:softprob` and eval_metric=`mlogloss`
    #  - for regression problems, for instance,
    #    objective=`reg:squarederror` and eval_metric=`rmse`
    kwargs.setdefault('objective', 'reg:squarederror' if kind == 'regressor' else 'multi:softprob')
    kwargs.setdefault('eval_metric', 'rmse' if kind == 'regressor' else 'mlogloss')
    # Hyperparameters:
    #  - The most important hyperparameters are the number of regression
    #    trees (n_estimators), aka as boosters; the size of every
    #    regression tree (max_depth); and the boosting learning rate
    #    (learning_rate), aka as shrinkage or eta [JB18]
    #  - There is a trade-off between the number of estimators and the
    #    size (depth) of the regression trees with respect to the best
    #    solution: smaller values of learning_rate give rise to larger
    #    optimal number of estimators [JB18]
    #  - Using stochasting gradient boosting approaches is often useful.
    #    See Ch 16 in [JB18]
    #  - The optimal hyperparameter values found by [AL17] are
    #    n_estimators=500, max_depth=9 and learning_rate=0.1. They also
    #    "[...] both the linear and non-linear learners do not show a
    #     strong sensitivity to the selection of hyperparameters"
    kwargs.setdefault('n_estimators', 500)
    kwargs.setdefault('max_depth', 10)
    kwargs.setdefault('learning_rate', 0.1)
    # ... secondary hyperparameters:
    #  - parameters that may be used to to stochastic gradient boosting
    #    which can eventually help to keep overfitting under control:
    kwargs.setdefault('colsample_bylevel', None)
    kwargs.setdefault('colsample_bynode', None)
    kwargs.setdefault('colsample_bytree', None)
    kwargs.setdefault('subsample', None)
    # Other arguments:
    kwargs.setdefault('early_stopping_rounds', 10)
    kwargs.setdefault('verbosity', 1)
    kwargs.setdefault('n_jobs', -1)

    return (xgboost.XGBRegressor(**kwargs) if kind == 'regressor'
            else xgboost.XGBClassifier(**kwargs))


def fit_booster_model(model, data, who=None):

    if not isinstance(model, xgboost.XGBModel):
        raise ValueError(
            f'expected a XGBModel, got a {model.__class__.__name__}')

    if not isinstance(data, Dataset):
        raise ValueError(
            f'expected a Dataset, got a {data.__class__.__name__}')

    if ((data.train_x is None) or (data.train_y is None) or
            (data.test_x is None) or (data.test_y is None)):
        logger.warning('splitting data with default options')
        data.random_split()

    class ProgressCounter(xgboost.callback.TrainingCallback):

        def __init__(self, message):
            self.tqdm = tqdm(desc=message, leave=False)
            super().__init__()

        def after_iteration(self, model, epoch, evals_log):
            self.tqdm.update()
            return False

        def after_training(self, model):
            self.tqdm.close()
            return model

    logger.info(who or 'Fitting the booster...')
    features = data.train_x.columns
    logger.info(f' max. {model.n_estimators} estimators')
    logger.info(f' {len(features)} features: {list(features)}')
    logger.info(f' {len(data.train_x)} training cases')
    logger.info(f' {len(data.test_x)} test cases')

    eval_set = None
    verbose = False

    message = ''
    if model.early_stopping_rounds:
        eval_set = [(data.train_x, data.train_y), (data.test_x, data.test_y)]
        message = 'Early stopping iterations'

    counter = ProgressCounter(message=message)
    model.set_params(callbacks=[counter])
    model.fit(data.train_x, data.train_y, eval_set=eval_set, verbose=verbose)

    if hasattr(model, 'best_ntree_limit'):
        logger.info(f'Best_ntree_limit={model.best_ntree_limit}')
