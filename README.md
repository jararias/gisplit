# Global irradiance splitting (GISPLIT)


### Installation

```sh
python3 -m pip install git+https://github.com/jararias/gisplit@main
```

### Tests

```python
import gisplit.tests as gs_tests
gs_tests.test_1min_data()
pl.show()
```

### Brief use notes

Given a pandas Dataframe `data` with all required fields (see below), and assuming that it has a time
index with 10-min resolution, the splitting is as follows:

```python
from gisplit import GISPLIT
gs = GISPLIT(time_step='10min')  # 1min, 10min, 15min or 30min
pred = gs.predict(data)  # data is a Pandas Dataframe with the required fields
```

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