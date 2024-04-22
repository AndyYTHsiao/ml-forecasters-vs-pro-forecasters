import argparse
from ml_forecasters import MLForecaster


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML forecasts and professional forecasts for a given model"
    )
    parser.add_argument(
        "model",
        default="en",
        type=str,
        choices=["en", "rf", "svr", "lstm"],
        help="Model to run forecasts for",
    )
    parser.add_argument(
        "dataset", type=str, help="Path of the data to run forecasts for"
    )
    parser.add_argument(
        "target",
        type=str,
        choices=["CPI YoY", "Unemployment rate"],
        help="Indicator to run forecasts for",
    )
    parser.add_argument(
        "ylabel",
        type=str,
        help="Label for the y-axis of the plot",
    )

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    target = args.target
    ylabel = args.ylabel

    if target not in ["CPI YoY", "Unemployment rate"]:
        raise ValueError("Indicator must be either 'CPI YoY' or 'Unemployment rate'")

    forecaster = MLForecaster(dataset, model)
    forecaster.make_forecasts(target, ylabel)


if __name__ == "__main__":
    main()
