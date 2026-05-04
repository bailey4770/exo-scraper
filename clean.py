import pandas as pd
import numpy as np

DATA_FILE_PATH = "./data/exoplanet.eu_catalog_29-04-26_12_09_18.csv"
RESULT_FILE_PATH = "./data/golden_dataset.csv"

TEST_STARS: list[str] = [
    "Kepler-69",
    "Kepler-7",
    "Kepler-6",
    "Kepler-17",
    "Kepler-5",
    "Kepler-8",
    "Kepler-12",
]
RETURN_COLS: list[str] = [
    "star_name",
    "name",
    "orbital_period_min",
    "orbital_period",
    "orbital_period_max",
    "min_depth",
    "depth",
    "max_depth",
    "transit_duration_min",
    "transit_duration",
    "transit_duration_max",
]

# Need to ensure units are equal when calculating ratios
# Jupiter's radius as a fraction of the Sun
Rj_SOLAR_RADIUS = 0.10045
# Sun's radius as a fraction of 1 AU
Rs_AU = 0.00465


def _get_orbital_period_cols(df: pd.DataFrame) -> pd.DataFrame:
    orbital_period_min = df["orbital_period"] - df["orbital_period_error_min"]
    orbital_period_max = df["orbital_period"] + df["orbital_period_error_max"]

    return df.assign(
        orbital_period_min=orbital_period_min, orbital_period_max=orbital_period_max
    )


def _get_depth_cols(df: pd.DataFrame) -> pd.DataFrame:
    radius_min = df["radius"] - df["radius_error_min"]
    radius_max = df["radius"] + df["radius_error_max"]
    star_radius_min = df["star_radius"] - df["star_radius_error_min"]
    star_radius_max = df["star_radius"] + df["star_radius_error_max"]

    min_depth = (radius_min * Rj_SOLAR_RADIUS / star_radius_max) ** 2
    depth = (df["radius"] * Rj_SOLAR_RADIUS / df["star_radius"]) ** 2
    max_depth = (radius_max * Rj_SOLAR_RADIUS / star_radius_min) ** 2
    return df.assign(min_depth=min_depth, depth=depth, max_depth=max_depth)


def _get_transit_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expected transit duration from stored, exoplanet facts.

    Equation for transit duration (Source: Claude -> Winn (2010)):
    T = (orbital_period / π) * arcsin(
        (star_radius / semi_major_axis) * sqrt((1 + (radius / star_radius)² - impact_parameter²)
    )

    Output is (source: own desmos analysis):
    - increasing in orbital period
    - increasing in star_radius
    - decreasing in semi_major_axis
    - increasing in radius
    - decreasing in impact_parameter
    """

    def _get_arcsin_input(
        star_radius: pd.Series,
        semi_major_axis: pd.Series,
        radius: pd.Series,
        impact_parameter: pd.Series,
    ) -> pd.Series:
        arcsin_input = (
            (star_radius * Rs_AU)
            / semi_major_axis
            * np.sqrt(
                ((1 + ((radius * Rj_SOLAR_RADIUS) / star_radius)) ** 2)
                - (impact_parameter**2)
            )
        )

        arcsin_input = arcsin_input.where(arcsin_input.between(0, 1))
        arcsin_input = arcsin_input.dropna()
        assert arcsin_input.between(0, 1).all()
        return arcsin_input

    arcsin_input_min = _get_arcsin_input(
        star_radius=(df["star_radius"] - df["star_radius_error_min"]),
        semi_major_axis=(df["semi_major_axis"] + df["semi_major_axis_error_max"]),
        radius=(df["radius"] - df["radius_error_min"]),
        impact_parameter=(df["impact_parameter"] + df["impact_parameter_error_max"]),
    )
    transit_duration_min = (
        (df["orbital_period"] - df["orbital_period_error_min"]) / np.pi
    ) * np.arcsin(arcsin_input_min)

    arcsin_input = _get_arcsin_input(
        star_radius=df["star_radius"],
        semi_major_axis=df["semi_major_axis"],
        radius=df["radius"],
        impact_parameter=df["impact_parameter"],
    )
    transit_duration = (df["orbital_period"] / np.pi) * np.arcsin(arcsin_input)

    arcsin_input_max = _get_arcsin_input(
        star_radius=(df["star_radius"] + df["star_radius_error_max"]),
        semi_major_axis=(df["semi_major_axis"] - df["semi_major_axis_error_min"]),
        radius=(df["radius"] + df["radius_error_max"]),
        impact_parameter=(df["impact_parameter"] - df["impact_parameter_error_min"]),
    )
    transit_duration_max = (
        (df["orbital_period"] + df["orbital_period_error_max"]) / np.pi
    ) * np.arcsin(arcsin_input_max)

    return df.assign(
        transit_duration_min=transit_duration_min,
        transit_duration=transit_duration,
        transit_duration_max=transit_duration_max,
    )


def main():
    NA_SUBSET = [
        "orbital_period",
        "star_radius",
        "semi_major_axis",
        "radius",
        "impact_parameter",
    ]

    df: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)
    # code also works for all planets in dataset, removing those where returned cols have na values
    test_planets = df.loc[df["star_name"].isin(TEST_STARS)]
    dropped_na = test_planets.dropna(subset=NA_SUBSET)

    with_orbital_error_cols = _get_orbital_period_cols(dropped_na)
    with_depth_error_cols = _get_depth_cols(with_orbital_error_cols)
    with_transit_error_cols = _get_transit_duration(with_depth_error_cols)

    res = with_transit_error_cols.dropna(subset=NA_SUBSET)
    print(res[RETURN_COLS])
    res.to_csv(RESULT_FILE_PATH)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    main()
