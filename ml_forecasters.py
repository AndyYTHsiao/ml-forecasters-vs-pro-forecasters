"""
Copyright (c) 2023, Andy Hsiao
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import pandas as pd
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
import warnings
import absl.logging
from sklearn.exceptions import ConvergenceWarning
from typing import Callable

# Import packages for cross validation, grid search, and standardization
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

# Import packages for elastic net, random forest, and support vector regression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Packages for LSTM model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from keras.regularizers import L1L2

# Ignore warnings
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
absl.logging.set_verbosity(absl.logging.ERROR)

SEED = 44


class MLForecaster:
    """
    A class used to build up forecasting models and compare results with professional forecast.

    Attributes:
        dataset (str): The dataset to be used.
        model (str): The model to be used.
        model_name (str): The full name of the model.
        correlation_matrix (pd.DataFrame): The correlation matrix of the dataset.
    """

    def __init__(self, dataset: str, model: str = "en") -> None:
        self.dataset = dataset
        self.model = model
        self.model_name = {
            "en": "Elastic Net",
            "rf": "Random Forest",
            "svr": "Support Vector Regression",
            "lstm": "LSTM",
        }[model]
        self.correlation_matrix = None

        if self.model not in ["en", "rf", "svr", "lstm"]:
            raise ValueError(
                'Invalid model name. Please choose one of: "en", "rf", "svr", "lstm".'
            )

    def draw_corr_matrix(self) -> None:
        """
        Draw a correlation matrix of the given dataframe.
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix has not been calculated yet.")

        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            vmax=0.3,
            center=0,
            cmap=cmap,
            annot=True,
            square=True,
            linewidths=0.5,
            annot_kws={"fontname": "Times New Roman"},
            cbar_kws={"shrink": 0.5},
            fmt=".2f",
            ax=ax,
        )
        plt.show()

    def make_forecasts(self, target: str, ylabel: str) -> None:
        """
        Make forecasts using the given model.

        Args:
            target (str): The label of target variable.
            ylabel (str): The label of y-axis.
        """
        if self.model == "en":
            self._elastic_net(target, ylabel)
        elif self.model == "rf":
            self._random_forest(target, ylabel)
        elif self.model == "svr":
            self._svr(target, ylabel)
        elif self.model == "lstm":
            self._lstm(target, ylabel)

    def _draw_plot(
        self,
        pred: pd.Series,
        y_test: pd.Series,
        pro_forecast: pd.Series,
        ylabel: str,
        model_label: str,
        graph_title: str,
    ) -> None:
        """
        Draw a plot of the results.

        Args:
            pred (pd.Series): The predicted value.
            y_test (pd.Series): The actual value.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
            model_label (str): The label of model.
            graph_title (str): The title of the graph.
        """
        # Set font type and size
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({"font.size": 10})
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(11, 9),
            sharey=True,
            tight_layout=True,
        )

        # Draw forecasts in different forecasting periods
        for ax, num in zip(axes.ravel(), np.arange(12, 60, 12)):
            ax.plot(
                y_test.index[:num],
                y_test[:num],
                color="#003049",
                label="Actual value",
            )
            ax.plot(y_test.index[:num], pred[:num], color="#d62828", label=model_label)
            ax.plot(
                y_test.index[:num],
                pro_forecast[-y_test.shape[0] : (-y_test.shape[0] + num)],
                color="#D1AC00",
                label="Professionals",
            )

            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_text(f"{num}-month-ahead")
            ax.grid(axis="y")

            # Set x-axis format and range
            ax.xaxis.set_major_formatter(md.DateFormatter("%b-%y"))
            ax.set_xlim(y_test.index[0], y_test.index[num])

            # Set the position of the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=3,
                frameon=False,
            )

        fig.supylabel(ylabel)
        plt.yticks()
        plt.suptitle(graph_title)
        plt.show()

    def _evaluation_metrics(
        self,
        pred: pd.Series,
        y_test: pd.Series,
        pro_forecasts: pd.Series,
        num: int,
        model_label: str,
    ) -> None:
        """
        Compare the predicted results with the professional forecast and actual value.

        Args:
            pred (pd.Series): The predicted value.
            y_test (pd.Series): The actual value.
            pro_forecasts (pd.Series): Forecasts made by professionals.
            num (int): The number of months ahead.
            model_label (str): The label of model.
        """
        rmse_ml = root_mean_squared_error(y_test[:num], pred[:num])
        mae_ml = mean_absolute_error(y_test[:num], pred[:num])
        mape_ml = mean_absolute_percentage_error(y_test[:num], pred[:num])

        rmse_pro = mean_squared_error(
            y_test[:num],
            pro_forecasts[-y_test.shape[0] : (-y_test.shape[0] + num)],
            squared=False,
        )
        mae_pro = mean_absolute_error(
            y_test[:num], pro_forecasts[-y_test.shape[0] : (-y_test.shape[0] + num)]
        )
        mape_pro = mean_absolute_percentage_error(
            y_test[:num], pro_forecasts[-y_test.shape[0] : (-y_test.shape[0] + num)]
        )

        res = pd.DataFrame(
            {
                "RMSE": [rmse_ml, rmse_pro],
                "MAE": [mae_ml, mae_pro],
                "MAPE": [mape_ml, mape_pro],
            },
            index=[model_label, "Professionals"],
        ).apply(lambda x: round(x, 4))

        print(res)

    def _grid_search_and_cv(
        self,
        mod: Callable,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        param_grid: dict,
    ) -> Callable:
        """
        Run grid search and cross validation to optimize hyperparameters.

        Args:
            mod (Callable): The model to be used.
            x_train (pd.DataFrame): The training set of independent variables.
            y_train (pd.DataFrame): The training set of dependent variables.
            param_grid (dict): The hyperparameters to be optimized.

        Returns:
            best_mod (Callable): The optimized model.
        """
        cv = TimeSeriesSplit(n_splits=10)
        grid_search = GridSearchCV(
            mod, param_grid, scoring="neg_mean_squared_error", cv=cv
        )
        grid_search.fit(X_train, y_train)
        best_mod = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        print()

        return best_mod

    def _preprocess_data(
        self, target: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        if target == "CPI YoY":
            pro_forecasts_col = "CPI YoY forecasts"
        elif target == "Unemployment rate":
            pro_forecasts_col = "Unemployment rate forecasts"
        else:
            raise ValueError(
                "Indicator must be either 'CPI YoY' or 'Unemployment rate'"
            )

        df = pd.read_csv(self.dataset, parse_dates=["Date"], index_col="Date")

        X = df.loc[:, (df.columns != pro_forecasts_col) & (df.columns != target)]
        y = df.loc[:, target]
        pro_forecasts = df.loc[:, pro_forecasts_col]
        df_excluding_pro_forecasts = df.loc[:, df.columns != pro_forecasts_col]

        self.correlation_matrix = (
            df_excluding_pro_forecasts.corr()
        )  # Save the correlation matrix

        return self._split_data(X, y) + (pro_forecasts,)

    def _split_data(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training set and test set.

        Args:
            X (pd.DataFrame): The independent variables.
            y (pd.Series): The dependent variables.

        Returns:
            X_train (pd.DataFrame): The training set of independent variables.
            X_test (pd.DataFrame): The test set of independent variables.
            y_train (pd.Series): The training set of dependent variables.
            y_test (pd.Series): The test set of dependent variables.
        """
        # Standardize independent variables
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        # Split feature and label into training set and test set (8:2)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED, shuffle=False
        )

        return X_train, X_test, y_train, y_test

    def _elastic_net(self, target: str, ylabel: str) -> None:
        """
        Run elastic net model.

        Args:
            target (str): The target variable.
            ylabel (str): The label of y-axis.
        """
        X_train, X_test, y_train, y_test, pro_forecasts = self._preprocess_data(target)
        param_grid = {
            "alpha": [1e-2, 0.1, 1, 10],
            "l1_ratio": np.arange(0, 1.01, 0.01),
            "tol": [1e-4, 1e-3, 1e-2],
        }
        en_mod = ElasticNet(random_state=SEED, max_iter=1000)
        best_mod = self._grid_search_and_cv(
            en_mod, X_train, y_train, param_grid=param_grid
        )
        best_mod_fit = best_mod.fit(X_train, y_train)
        pred = best_mod_fit.predict(X_test)

        self._formulate_results(pred, y_test, pro_forecasts, ylabel)

    def _random_forest(self, target: str, ylabel: str) -> None:
        """
        Run random forest model.

        Args:
            target (str): The target variable.
            ylabel (str): The label of y-axis.
        """
        X_train, X_test, y_train, y_test, pro_forecasts = self._preprocess_data(target)
        rf_mod = RandomForestRegressor(random_state=SEED)
        param_grid = {
            "n_estimators": [100, 150, 200],
            "max_features": ["auto", "sqrt", "log2", None],
            "max_depth": [1, 3, 5, 7],
        }
        best_mod = self._grid_search_and_cv(
            rf_mod, X_train, y_train, param_grid=param_grid
        )
        best_mod_fit = best_mod.fit(X_train, y_train)
        pred = best_mod_fit.predict(X_test)

        self._formulate_results(pred, y_test, pro_forecasts, ylabel)

    def _svr(self, target: str, ylabel: str) -> None:
        """
        Run SVR model.

        Args:
            target (str): The target variable.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
        """
        X_train, X_test, y_train, y_test, pro_forecasts = self._preprocess_data(target)
        svr_mod = SVR()
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1.0, 5.0],
            "kernel": ["linear", "rbf"],
            "epsilon": [1e-3, 1e-2, 1e-1],
        }
        best_mod = self._grid_search_and_cv(
            svr_mod, X_train, y_train, param_grid=param_grid
        )
        best_mod_fit = best_mod.fit(X_train, y_train)
        pred = best_mod_fit.predict(X_test)

        self._formulate_results(pred, y_test, pro_forecasts, ylabel)

    @tf.function(reduce_retracing=True)
    def _baseline_model(self, dim: int) -> Callable:
        """
        Build up baseline model for LSTM implementation.

        Args:
            dim (int): The number of dimensions.

        Returns:
            A callable model.
        """

        def create_lstm_model(units=32) -> Sequential:
            """
            Create LSTM model.

            Args:
                units (int): The number of units.

            Returns:
                A sequential model.
            """
            model = Sequential()
            model.add(
                LSTM(
                    units=units,
                    input_shape=(1, dim),
                    return_sequences=True,
                    kernel_regularizer=L1L2(l1=0.1, l2=0.1),
                )
            )
            model.add(LSTM(units=units))
            model.add(Dense(1))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer="adam")
            return model

        return create_lstm_model

    def _lstm(self, target: str, ylabel: str) -> None:
        """
        Run LSTM model.

        Args:
            target (str): The target variable.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
        """
        X_train, X_test, y_train, y_test, pro_forecasts = self._preprocess_data(target)

        # Reshape data
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Build LSTM model and run grid search to optimize hyperparameters
        tf.random.set_seed(SEED)
        param_grid = {"batch_size": [32, 64, 128], "epochs": [80, 100, 150, 200]}
        filepath = "model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

        # Improve efficiency by introducing early stopping
        lstm_mod = KerasRegressor(
            model=self._baseline_model(X_train.shape[2]), epochs=100, verbose=0
        )
        best_mod = self._grid_search_and_cv(lstm_mod, X_train, y_train, param_grid)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        check_point = ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
        best_mod_fit = best_mod.fit(
            X_train,
            y_train,
            verbose=0,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, check_point],
        )
        pred = best_mod_fit.predict(X_test)

        self._formulate_results(pred, y_test, pro_forecasts, ylabel)

    def _formulate_results(
        self,
        pred: np.ndarray,
        y_test: np.ndarray,
        pro_forecasts: pd.Series,
        ylabel: str,
    ) -> None:
        """
        Generate results.

        Args:
            pred (np.ndarray): The predicted value.
            y_test (np.ndarray): The actual value.
            pro_forecasts (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis
            model_name (str): The label of model.
        """
        for num in np.arange(12, 60, 12):
            print(f"======= {num}-month-ahead =======")
            self._evaluation_metrics(pred, y_test, pro_forecasts, num, self.model_name)

        self._draw_plot(
            pred,
            y_test,
            pro_forecasts,
            ylabel,
            self.model_name,
            f"{self.model_name} vs Professional forecaster",
        )
