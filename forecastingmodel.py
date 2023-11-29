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

# Typing
from typing import Callable, Iterable

# Import packages for cross validation, grid search, and standardization
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
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


class ForecastingModel:
    """
    A class used to build up forecasting models and compare results with professional forecast.

    Attributes:
        model (str): The model to be used.
        model_name (str): The full name of the model to be used.
    """

    def __init__(self, model: str = "en") -> None:
        """Determine the model to be used."""
        self.model = model

        # Assign the full name of the model to a variable
        if self.model == "en":
            self.model_name = "Elastic net"
        elif self.model == "rf":
            self.model_name = "Random forest"
        elif self.model == "svr":
            self.model_name = "SVR"
        elif self.model == "lstm":
            self.model_name = "LSTM"
        else:
            raise ValueError(
                'Invalid model name. Please choose one of: "en", "rf", "svr", "lstm".'
            )

    def corr_matrix(self, df: pd.DataFrame) -> sns.heatmap:
        """
        Draw a correlation matrix of the given dataframe.

        Args:
            df: The dataframe to be drawn.

        Returns:
            A correlation matrix of the given dataframe.
        """
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100, tight_layout=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            vmax=0.3,
            center=0,
            cmap=cmap,
            annot=True,
            square=True,
            linewidths=0.5,
            annot_kws={"Times New Roman"},
            cbar_kws={"shrink": 0.5},
            fmt=".2f",
        )

    def make_forecasts(
        self, df: pd.DataFrame, label: str, pro_forecast: pd.Series, ylabel: str
    ) -> None:
        """
        Make forecasts using the given model.

        Args:
            df (pd.DataFrame): The dataframe to be used.
            label (str): The label of target variable.
            pro_forecast (pd.Series): The professional forecast.
            ylabel (str): The label of y-axis.
        """
        if self.model == "en":
            self._elastic_net(df, label, pro_forecast, ylabel)
        elif self.model == "rf":
            self._random_forest(df, label, pro_forecast, ylabel)
        elif self.model == "svr":
            self._svr(df, label, pro_forecast, ylabel)
        elif self.model == "lstm":
            self._lstm(df, label, pro_forecast, ylabel)

    def _draw_plot(
        self,
        pred: pd.Series,
        y_test: np.ndarray,
        pro_forecast: pd.Series,
        ylabel: str,
        model_label: str,
        graph_title: str,
    ) -> None:
        """
        Draw a plot of the results.

        Args:
            pred (pd.Series): The predicted value.
            y_test (np.ndarray): The actual value.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
            model_label (str): The label of model.
            graph_title (str): The title of the graph.
        """
        # Set font type and size
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({"font.size": 12})
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 10),
            sharey=True,
            dpi=100,
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

        fig.supxlabel("Date")
        fig.supylabel(ylabel)
        plt.yticks()
        plt.suptitle(graph_title)
        plt.show()

    def _evaluation_metrics(
        self,
        pred: pd.Series,
        y_test,
        pro_forecast: pd.Series,
        num: int,
        model_label: str,
    ) -> None:
        """
        Compare the predicted results with the professional forecast and actual value.

        Args:
            pred (pd.Series): The predicted value.
            y_test (np.ndarray): The actual value.
            pro_forecast (pd.Series): Forecasts made by professionals.
            num (int): The number of months ahead.
            model_label (str): The label of model.
        """
        rmse_ml = mean_squared_error(y_test[:num], pred[:num], squared=False)
        mae_ml = mean_absolute_error(y_test[:num], pred[:num])
        mape_ml = mean_absolute_percentage_error(y_test[:num], pred[:num])

        rmse_pro = mean_squared_error(
            y_test[:num],
            pro_forecast[-y_test.shape[0] : (-y_test.shape[0] + num)],
            squared=False,
        )
        mae_pro = mean_absolute_error(
            y_test[:num], pro_forecast[-y_test.shape[0] : (-y_test.shape[0] + num)]
        )
        mape_pro = mean_absolute_percentage_error(
            y_test[:num], pro_forecast[-y_test.shape[0] : (-y_test.shape[0] + num)]
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

    # Run grid search
    def _grid_search_and_cv(self, mod: Callable, x_train, y_train, param_grid: dict):
        cv = TimeSeriesSplit(n_splits=10)
        grid_search = GridSearchCV(
            mod, param_grid, scoring="neg_mean_squared_error", cv=cv
        )
        grid_search.fit(x_train, y_train)
        best_mod = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)

        return best_mod

    # Standardize and split variables
    def _variable_setup(self, df: pd.DataFrame, target: str) -> Iterable[np.ndarray]:
        # Variable setup
        X = df.drop(target, axis=1)
        y = df[target]

        # Standardize independent variables
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        # Split feature and label into training set and test set (8:2)
        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=44, shuffle=False
        )

        return x_train, x_test, y_train, y_test

    # Run elastic net model
    def _elastic_net(
        self, df: pd.DataFrame, target: str, pro_forecast: pd.Series, ylabel: str
    ) -> None:
        """
        Run elastic net model.

        Args:
            df (pd.DataFrame): The dataframe to be used.
            target (str): The target variable.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
        """
        x_train, x_test, y_train, y_test = self._variable_setup(self, df, target)

        # Build up elastic net regression model and run grid search to optimize hyperparameters
        param_grid = {
            "alpha": [1e-2, 0.1, 1, 10],
            "l1_ratio": np.arange(0, 1.01, 0.01),
            "tol": [1e-4, 1e-3, 1e-2],
        }
        en_mod = ElasticNet(random_state=44, max_iter=5000)
        best_mod = self._grid_search_and_cv(
            self, en_mod, x_train, y_train, param_grid=param_grid
        )
        best_mod_fit = best_mod.fit(x_train, y_train)
        pred = best_mod_fit.predict(x_test)

        self._result(pred, y_test, pro_forecast, ylabel, self.model_name)

    def _random_forest(
        self, df: pd.DataFrame, target: str, pro_forecast: pd.Series, ylabel: str
    ) -> None:
        """
        Run random forest model.

        Args:
            df (pd.DataFrame): The dataframe to be used.
            target (str): The target variable.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
        """
        # Variable setup
        x_train, x_test, y_train, y_test = self._variable_setup(df, target)

        # Build up random forest model and run grid search to optimize hyperparameters
        rf_mod = RandomForestRegressor(random_state=44)
        param_grid = {
            "n_estimators": [100, 150, 200],
            "max_features": ["auto", "sqrt", "log2", None],
            "max_depth": [1, 3, 5, 7],
        }
        best_mod = self._grid_search_and_cv(
            rf_mod, x_train, y_train, param_grid=param_grid
        )
        best_mod_fit = best_mod.fit(x_train, y_train)
        pred = best_mod_fit.predict(x_test)

        self._result(pred, y_test, pro_forecast, ylabel, self.model_name)

    def _svr(
        self, df: pd.DataFrame, target: str, pro_forecast: pd.Series, ylabel: str
    ) -> None:
        """
        Run SVR model.

        Args:
            df (pd.DataFrame): The dataframe to be used.
            target (str): The target variable.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
        """
        x_train, x_test, y_train, y_test = self._variable_setup(df, target)

        # Build up SVR model and run grid search to optimize hyperparameters
        svr_mod = SVR()
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1.0, 5.0],
            "kernel": ["linear", "rbf"],
            "epsilon": [1e-3, 1e-2, 1e-1],
        }
        best_mod = self._grid_search_and_cv(
            svr_mod, x_train, y_train, param_grid=param_grid
        )
        best_mod_fit = best_mod.fit(x_train, y_train)
        pred = best_mod_fit.predict(x_test)

        self._result(pred, y_test, pro_forecast, ylabel, self.model_name)

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

    def _lstm(
        self, df: pd.DataFrame, target: str, pro_forecast: np.ndarray, ylabel: str
    ) -> None:
        """
        Run LSTM model.

        Args:
            df (pd.DataFrame): The dataframe to be used.
            target (str): The target variable.
            pro_forecast (np.ndarray): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
        """

        x_train, x_test, y_train, y_test = self._variable_setup(df, target)

        # Reshape data
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        # Build LSTM model and run grid search to optimize hyperparameters
        tf.random.set_seed(44)
        param_grid = {"batch_size": [32, 64, 128], "epochs": [80, 100, 150, 200]}
        filepath = "model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

        # Improve efficiency by introducing early stopping
        lstm_mod = KerasRegressor(
            model=self._baseline_model(x_train.shape[2]), epochs=100, verbose=0
        )
        best_mod = self._grid_search_and_cv(lstm_mod, x_train, y_train, param_grid)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        check_point = ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
        best_mod_fit = best_mod.fit(
            x_train,
            y_train,
            verbose=0,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, check_point],
        )
        pred = best_mod_fit.predict(x_test)

        self._result(pred, y_test, pro_forecast, ylabel, self.model_name)

    def _result(
        self,
        pred: np.ndarray,
        y_test: np.ndarray,
        pro_forecast: pd.Series,
        ylabel: str,
        model_label: str,
    ) -> None:
        """
        Generate results

        Args:
            pred (np.ndarray): The predicted value.
            y_test (np.ndarray): The actual value.
            pro_forecast (pd.Series): Forecasts made by professionals.
            ylabel (str): The label of y-axis.
            model_label (str): The label of model.
        """
        for num in np.arange(12, 60, 12):
            print(f"======= {num}-month-ahead =======")
            self._evaluation_metrics(pred, y_test, pro_forecast, num, model_label)

        self._draw_plot(
            pred,
            y_test,
            pro_forecast,
            ylabel,
            model_label,
            f"{model_label} vs Professional forecaster",
        )
