import pandas as pd
import numpy as np
import os

DATA_FILE_PATH = "./data/raw/exoplanet.eu_catalog_29-04-26_12_09_18.csv"
RESULT_FILE_PATH = "./data/processed/reference_transit_params.csv"

TEST_STARS: list[str] = [
    "Kepler-69",
    "Kepler-7",
    "Kepler-6",
    "Kepler-17",
    "Kepler-5",
    "Kepler-8",
    "Kepler-12",
]
REQUIRED_COLS = [
    "name",
    "star_name",
    "orbital_period_error_min",
    "orbital_period",
    "orbital_period_error_max",
    "star_radius_error_min",
    "star_radius",
    "star_radius_error_max",
    "semi_major_axis_error_min",
    "semi_major_axis",
    "semi_major_axis_error_max",
    "radius_error_min",
    "radius",
    "radius_error_max",
    "impact_parameter_error_min",
    "impact_parameter",
    "impact_parameter_error_max",
]
RETURN_COLS: list[str] = [
    "star_name",
    "name",
    "orbital_period_min",
    "orbital_period",
    "orbital_period_max",
    "depth_min",
    "depth",
    "depth_max",
    "transit_duration_min",
    "transit_duration",
    "transit_duration_max",
]


# Need to ensure units are equal when calculating ratios
R_JUPITER = 0.10045  # Jupiter's radius as a fraction of the Sun's radius
R_SUN = 0.00465  # Sun's radius as a fraction of 1 AU


def extract(
    file_path: str, required_cols: list[str], test_stars: list[str] | None = None
) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(file_path)
    dropped: pd.DataFrame = df[required_cols]

    # code also works for all planets in dataset
    if test_stars is None:
        return dropped
    return dropped.loc[df["star_name"].isin(test_stars)]


def transform(df: pd.DataFrame, return_cols: list[str]) -> pd.DataFrame:
    def _add_orbital_period_cols(df: pd.DataFrame) -> pd.DataFrame:
        orbital_period_min = df["orbital_period"] - df["orbital_period_error_min"]
        orbital_period_max = df["orbital_period"] + df["orbital_period_error_max"]

        return df.assign(
            orbital_period_min=orbital_period_min, orbital_period_max=orbital_period_max
        )

    def _add_depth_cols(df: pd.DataFrame) -> pd.DataFrame:
        radius_min = df["radius"] - df["radius_error_min"]
        radius_max = df["radius"] + df["radius_error_max"]
        star_radius_min = df["star_radius"] - df["star_radius_error_min"]
        star_radius_max = df["star_radius"] + df["star_radius_error_max"]

        depth_min = (radius_min * R_JUPITER / star_radius_max) ** 2
        depth = (df["radius"] * R_JUPITER / df["star_radius"]) ** 2
        depth_max = (radius_max * R_JUPITER / star_radius_min) ** 2
        return df.assign(depth_min=depth_min, depth=depth, depth_max=depth_max)

    def _add_transit_duration(df: pd.DataFrame) -> pd.DataFrame:
        """Compute expected transit duration from stored, exoplanet facts.

        Equation for transit duration (Winn, Joshua N. “Transits and Occultations.” arXiv:1001.2010, arXiv, 24 Sept. 2014. arXiv.org, https://doi.org/10.48550/arXiv.1001.2010):
        T = (orbital_period / π) * arcsin(
            (star_radius / semi_major_axis) * sqrt((1 + (radius / star_radius))² - impact_parameter²)
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
                (star_radius * R_SUN)
                / semi_major_axis
                * np.sqrt(
                    ((1 + ((radius * R_JUPITER) / star_radius)) ** 2)
                    - (impact_parameter**2)
                )
            )

            valid_mask = arcsin_input.isna() | arcsin_input.between(0, 1)
            return arcsin_input.where(valid_mask)

        arcsin_input_min = _get_arcsin_input(
            star_radius=(df["star_radius"] - df["star_radius_error_min"]),
            semi_major_axis=(df["semi_major_axis"] + df["semi_major_axis_error_max"]),
            radius=(df["radius"] - df["radius_error_min"]),
            impact_parameter=(
                df["impact_parameter"] + df["impact_parameter_error_max"]
            ),
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
            impact_parameter=(
                df["impact_parameter"] - df["impact_parameter_error_min"]
            ),
        )
        transit_duration_max = (
            (df["orbital_period"] + df["orbital_period_error_max"]) / np.pi
        ) * np.arcsin(arcsin_input_max)

        return df.assign(
            transit_duration_min=transit_duration_min,
            transit_duration=transit_duration,
            transit_duration_max=transit_duration_max,
        )

    with_orbital_error_cols = _add_orbital_period_cols(df)
    with_depth_error_cols = _add_depth_cols(with_orbital_error_cols)
    with_transit_error_cols = _add_transit_duration(with_depth_error_cols)

    return with_transit_error_cols[return_cols]


def load(transformed: pd.DataFrame, file_path: str, display: bool = False):
    if display:
        print(transformed)

    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    transformed.to_csv(file_path, index=False)
    assert os.path.exists(file_path)


def main():
    df: pd.DataFrame = extract(DATA_FILE_PATH, REQUIRED_COLS, TEST_STARS)
    transformed: pd.DataFrame = transform(df, return_cols=RETURN_COLS)
    load(transformed, RESULT_FILE_PATH, display=True)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    main()
