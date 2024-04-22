# ML forecasters vs Professional forecasters

This repo contains modified source code of my thesis. It aims to compare the accuracy of machine learning (ML) forecasters and professional analysts in predicting economic indicators.

Professinoal forecasters refer to analysts and economists in financial companies, academic institutions, or the governemnt.

The datasets are included in this repo. The professinoal forecasts are retrieved from Bloomberg Terminal.

To run the code, you can either use command line script

```
python3 run_forecasts.py \
    svr \
    data/tw_cpi.csv \
    "CPI YoY" \
    "TW CPI YoY (%)
```
or run the Python script in Jupyter Notebook to generate results

```python
from ml_forecasters import MLForecaster

forecaster = MLForecaster("data/tw_cpi.csv", "svr")
forecaster.make_forecasts("CPI YoY", "TW CPI YoY (%)")
```

## Methods
The table below shows main methods of the class.
| Description | Method | 
|-----------------------|---|
| Make forecasts | `.make_forecasts(dataset, model)` |
| Visualize correlation matrix (excluding professional forecasts) | `.corr_matrix()` |

`corr_matrix()` is executable only after making forecasts. Since Seaborn 0.12.x only generate annotations in the top row of heatmaps, please upgrade the package to the latest version.

## Attributes
The table below lists out the attributes of the class.
| Description | Attribute | 
|-----------------------|---|
| The dataset to be used | `dataset` (str) |
| The model to be used |  `model` (str) |
| The full name of the model |  `model_name` (str) |
| The correlation matrix of the dataset (excluding professional forecasts) |  `correlation_matrix` (pandas.DataFrame) |

Inquiries and comments are welcome.
