import pandas as pd
from math import pi
from numpy import arcsin

DATA_FILE_PATH = "./data/exoplanet.eu_catalog_29-04-26_12_09_18.csv"

# Need to ensure units are equal when dividing

# Jupiter's radius as a fraction of the Sun
Rj_SOLAR_RADIUS = 0.10045
# Sun's radius as a fraction of 1 AU
Rs_AU = 0.00465


def _get_orbital_period_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        orbital_period_min=(df["orbital_period"] - df["orbital_period_error_min"]),
        orbital_period_max=(df["orbital_period"] + df["orbital_period_error_max"]),
    )


def _get_depth_cols(df: pd.DataFrame):
    res: pd.DataFrame = df.assign(
        radius_min=(df["radius"] - df["radius_error_min"]),
        radius_max=(df["radius"] + df["radius_error_max"]),
        star_radius_min=(df["star_radius"] - df["star_radius_error_min"]),
        star_radius_max=(df["star_radius"] + df["star_radius_error_max"]),
    )
    res["min_depth"] = (
        res["radius_min"] * Rj_SOLAR_RADIUS / res["star_radius_max"]
    ) ** 2
    res["max_depth"] = (
        res["radius_max"] * Rj_SOLAR_RADIUS / res["star_radius_min"]
    ) ** 2
    return res


def get_transit_duration(df: pd.DataFrame):
    transit_duration_min = (df["orbital_period_min"] / pi) * arcsin(
        (df["star_radius`"])
    )


def main():
    df: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)
    # print(df)
    # print("cols: ", list(df))

    return_cols: list[str] = [
        "star_name",
        "name",
        "orbital_period_min",
        "orbital_period_max",
        "radius",
        "star_radius",
        "min_depth",
        "max_depth",
    ]

    final = _get_orbital_period_cols(df)
    final = _get_depth_cols(final)
    print(
        final.loc[
            final["star_name"] == "Kepler-69",
            return_cols,
        ]
    )


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    main()
