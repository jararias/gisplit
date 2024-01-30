# GISPLIT: High-performance global solar irradiance component-separation model for 1-min data

A Python implementation of the GISPLIT global solar irradiance component-separation model for 1-min data, described in (open access):

>>>
Ruiz-Arias, J.A. and Gueymard, C.A. (2024) GISPLIT: High-performance global solar irradiance component-separation model dynamically constrained by 1-min sky conditions. _Solar Energy_ XXX doi: [10.1016/j.solener.2024.112363](https://doi.org/10.1016/j.solener.2024.112363)
>>>

<p align="center">
    <img src="assets/gisplit_diag.png" alt="GISPLIT diagnostics">
</p>


### Installation

```sh
python3 -m pip install git+https://github.com/jararias/gisplit@main
```

### Tests

To test it and benchmark it against same-class state-of-the-art separation models, install the [splitting_models](https://github.com/jararias/splitting_models) package:

```sh
python3 -m pip install git+https://github.com/jararias/splitting_models@main
```

and run there the tests:

```sh
python3 -c "import splitting_models.tests as sm_tests; sm_tests.basic_test()"
```

from the command-line or:

```python
import pylab as pl
import splitting_models.tests as sm_tests
sm_tests.basic_test()
pl.show()
```

in a python script.

### Brief use notes

Given a pandas Dataframe `data` with all required fields (see below), and assuming that it has a time
index with 10-min resolution, the splitting is as follows:

```python
from gisplit import GISPLIT
gs = GISPLIT(climate=None, engine='reg')
pred = gs.predict(data)
```

Notes on sky_type_or_func!!

where `pred` is a DataFrame with the same (time) index as `data` and three fields: `dif`, `dni` and `dir`,
for diffuse, direct normal and direct horizontal irradiances (in W/m$`^2`$), respectively.

In addition, GISPLIT accepts other two important arguments:

- `target`, which must be set to "sat" when the input GHI is from the satellite model, or "obs" when the
input GHI is observed. It defaults to "sat". This argument is ignored for 1-min GHI because only observations
are possible at this high resolution.

- `engine`, which selects the model that splits the GHI into its components: "xgb" to use a
[extreme gradient boosting regressor](https://xgboost.readthedocs.io/en/stable/), or "reg" to use a conventional
 regression model.

### Important things to have in mind:

- The DataFrame must have a time index with one of the resolutions considered by GISPLIT: 1, 10, 15 or 30 minutes.

- The DataFrame should be full, that is, without time gaps. This is so because, during the prediction, GISPLIT
uses moving windows to calculate variability indices that assume that the time series in the DataFrame are continuous

- When the GHI has to be splitted for multiple sites, all site data can be combined in a single Dataframe, but then
the Dataframe must have a 2-level multi-index: `times_utc` and `site`.

#### Required input fields for the splitting

Regardless the GHI time resolution, GISPLIT performs the GHI splitting following a two step process: first, it classifies
the sky situation, and then, it splits the GHI using models tailored for each sky situation.

There is a key difference between the GISPLIT operation for 1-min GHI series, and for 10-, 15- and 30-min time series. In
the first case, the sky classification is performed using [CAELUS](https://gitlab.solargis.com/backend/sg_caelus), which is
a threshold-based classification algorithm tailored for 1-min data, while for the other three time resolutions the sky
classification is performed using a [extreme gradient boosting](https://xgboost.readthedocs.io/en/stable/).

Also, additionally, the number of sky classes differs for the different time resolutions. For instance, there are 6 sky classes
considered for 1-min GHI series, while only 2 for 30-min GHI series.

>>>
The required fields in the input Dataframe are:

- *sza*: solar zenith angle (degrees from 0 to 180)
- *eth*: extraterrestrial solar irradiance for horizontal surface, in W/m$`^2`$
- *ghi*: global horizontal irradiance, in W/m$`^2`$
- *ghics*: clear-sky global horizontal irradiance, in W/m$`^2`$
- *difcs*: clear-sky diffuse horizontal irradiance, in W/m$`^2`$

for 10-, 15- and 30-min GHI series, and, additionally,

- *longitude*: geographical longitude, [-180, 180]
- *ghicda*: clean-and-dry global horizontal irradiance, in W/m$`^2`$; i.e., for an atmosphere without water vapor and aerosols.

for 1-min GHI series.
>>>