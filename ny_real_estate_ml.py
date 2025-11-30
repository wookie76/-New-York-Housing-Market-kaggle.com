#!/usr/bin/env python3
"""
NYC Housing Price Model – v7.4 ULTIMATE FINAL
+ Clip syntax fixed for all NumPy versions
+ Relaxed price validation (ge=0)
+ Honest MAE = $420k range
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import catboost as cb
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["CATBOOST_DISABLE_INTERACTIVE_FEATURES"] = "1"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
DEFAULT_SQFT = 800
MIN_SQFT = 200
MIN_YEAR_BUILT = 1800
DEFAULT_BUILDING_AGE = 50
NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.29, -73.68
TS_LAT, TS_LON = 40.7580, -73.9855
EARTH_RADIUS_KM = 111.0

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "NY-House-Dataset.csv"
PLUTO_PATH = ROOT / "pluto_2024v1.parquet"
ARTEFACTS_DIR = ROOT / "artefacts"
ARTEFACTS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# --------------------------------------------------------------------------- #
# Load PLUTO
# --------------------------------------------------------------------------- #
logger.info("Loading PLUTO...")
usecols = ["BBL", "latitude", "longitude", "yearbuilt"]
if PLUTO_PATH.suffix == ".parquet":
    df_pluto = pd.read_parquet(PLUTO_PATH, columns=usecols)
else:
    df_pluto = pd.read_csv(PLUTO_PATH, usecols=usecols, low_memory=False)

df_pluto = df_pluto.dropna(subset=["latitude", "longitude", "yearbuilt"])
df_pluto = df_pluto[df_pluto["yearbuilt"] > 1600]
PLUTO_COORDS = df_pluto[["latitude", "longitude"]].values
PLUTO_TREE = cKDTree(PLUTO_COORDS)
bbl_to_year = pd.Series(
    df_pluto["yearbuilt"].values, index=df_pluto["BBL"].astype(str)
).to_dict()
logger.success("PLUTO loaded – {} lots", len(df_pluto))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class RawRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    formatted_address: str = Field(alias="FORMATTED_ADDRESS")
    latitude: float = Field(alias="LATITUDE")
    longitude: float = Field(alias="LONGITUDE")
    price: float = Field(alias="PRICE")
    beds: float = Field(alias="BEDS")
    bath: float = Field(alias="BATH")
    propertysqft: float = Field(alias="PROPERTYSQFT")
    type_raw: str = Field(alias="TYPE")
    locality: str = Field(alias="LOCALITY")

    @field_validator("latitude")
    @classmethod
    def lat_ok(cls, v: float) -> float:
        if not (NYC_LAT_MIN <= v <= NYC_LAT_MAX):
            raise ValueError("Bad lat")
        return v

    @field_validator("longitude")
    @classmethod
    def lon_ok(cls, v: float) -> float:
        if not (NYC_LON_MIN <= v <= NYC_LON_MAX):
            raise ValueError("Bad lon")
        return v

    @field_validator("price")
    @classmethod
    def price_ok(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Negative price")
        return v  # Relaxed: allow $0


# --------------------------------------------------------------------------- #
# Extract BBL
# --------------------------------------------------------------------------- #
def extract_bbl(address: str) -> str | None:
    patterns = [
        r"BBL[:\s]+(\d{10})",
        r"Block[:\s]*(\d{1,5})[^0-9]*Lot[:\s]*(\d{1,4})",
        r"\((\d{10})\)",
    ]
    for pat in patterns:
        m = re.search(pat, address, re.I)
        if m:
            if len(m.groups()) == 1:
                return m.group(1)
            else:
                borough_map = {
                    "manhattan": "1",
                    "brooklyn": "3",
                    "queens": "4",
                    "bronx": "2",
                    "staten": "5",
                }
                borough_key = address.lower().split()[0]
                borough = borough_map.get(borough_key, "1")
                block = m.group(1).zfill(5)
                lot = m.group(2).zfill(4)
                return borough + block + lot
    return None


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    df["propertysqft"] = (
        pd.to_numeric(df["propertysqft"], errors="coerce")
        .fillna(DEFAULT_SQFT)
        .clip(lower=MIN_SQFT)
    )
    df["bath"] = df["bath"].clip(upper=df["bath"].quantile(0.99))
    df["beds"] = df["beds"].clip(lower=0)
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["log_sqft"] = np.log1p(df["propertysqft"])
    df["log_price"] = np.log1p(df["price"])
    return df


def add_distances(df: pd.DataFrame) -> pd.DataFrame:
    df["dist_to_core"] = (
        (df["latitude"] - TS_LAT) ** 2 + (df["longitude"] - TS_LON) ** 2
    ) ** 0.5 * EARTH_RADIUS_KM
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["log_sqft_x_core"] = df["log_sqft"] * df["dist_to_core"] / 10_000
    df["beds_per_sqft"] = df["beds"] / df["propertysqft"].clip(lower=100)
    df["bath_per_sqft"] = df["bath"] / df["propertysqft"].clip(lower=100)
    df["size_rank"] = df["propertysqft"].rank(pct=True)
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["type_clean"] = df["type_raw"].astype(str).str.split().str[0].str.lower()
    df["borough"] = df["locality"].apply(
        lambda x: next(
            (
                b
                for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "StatenIsland"]
                if b.lower() in str(x).lower()
            ),
            "Other",
        )
    )
    return df


def add_building_age(df: pd.DataFrame) -> pd.DataFrame:
    df["bbl"] = df["formatted_address"].apply(extract_bbl)
    df["building_age"] = df["bbl"].astype(str).map(bbl_to_year)

    missing_mask = df["building_age"].isna()
    if missing_mask.sum() > 0:
        coords = df.loc[missing_mask, ["latitude", "longitude"]].values
        distances, indices = PLUTO_TREE.query(coords, k=1)
        fallback_years = df_pluto.iloc[indices]["yearbuilt"].values
        df.loc[missing_mask, "building_age"] = fallback_years

    df["building_age"] = 2025 - np.clip(df["building_age"], MIN_YEAR_BUILT, None)
    df["building_age"] = df["building_age"].fillna(DEFAULT_BUILDING_AGE)
    df.drop(columns=["bbl"], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_numerics(df)
    df = add_log_transforms(df)
    df = add_distances(df)
    df = add_interactions(df)
    df = clean_categoricals(df)
    df = add_building_age(df)
    return df


# --------------------------------------------------------------------------- #
# Target encoding
# --------------------------------------------------------------------------- #
def apply_target_encoding(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    for col in ["borough", "type_clean"]:
        means = train.groupby(col)["log_price"].mean()
        global_mean = train["log_price"].mean()
        train[f"{col}_te"] = train[col].map(means).fillna(global_mean)
        val[f"{col}_te"] = val[col].map(means).fillna(global_mean)
        test[f"{col}_te"] = test[col].map(means).fillna(global_mean)
    return train, val, test


# --------------------------------------------------------------------------- #
# Load + prepare
# --------------------------------------------------------------------------- #
def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading housing data...")
    raw = pd.read_csv(DATA_PATH)
    valid = []
    dropped_count = 0
    for _, row in tqdm(raw.iterrows(), total=len(raw), desc="Validating"):
        try:
            valid.append(RawRow.model_validate(row.to_dict()).model_dump())
        except Exception as e:
            dropped_count += 1
            logger.debug(f"Dropped row: {e}")
    df = pd.DataFrame(valid)
    logger.info(f"Kept {len(df)} / {len(raw)} rows (dropped {dropped_count})")

    p1, p99 = np.percentile(df["price"], [1, 99])
    df["price"] = np.clip(df["price"], p1, p99)

    df = engineer_features(df)

    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(df["price"], 10, duplicates="drop"),
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(temp["price"], 10, duplicates="drop"),
    )

    train, val, test = apply_target_encoding(train, val, test)
    logger.info(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# --------------------------------------------------------------------------- #
# Features & training
# --------------------------------------------------------------------------- #
FEATURES = [
    "latitude",
    "longitude",
    "beds",
    "bath",
    "propertysqft",
    "log_sqft",
    "dist_to_core",
    "log_sqft_x_core",
    "beds_per_sqft",
    "bath_per_sqft",
    "size_rank",
    "borough_te",
    "type_clean_te",
    "building_age",
]


def train_final_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = train[FEATURES], train["log_price"]
    X_val, y_val = val[FEATURES], val["log_price"]
    X_test, y_test = test[FEATURES], test["log_price"]

    model = cb.CatBoostRegressor(
        iterations=6000,
        learning_rate=0.025,
        depth=10,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=500,
        loss_function="RMSE",
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    pred_log = model.predict(X_test)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(pred_log))
    r2 = r2_score(y_test, pred_log)

    logger.info("v7.4 FINAL TEST → R² = {:.4f} | MAE = ${:,.0f}", r2, mae)

    sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(ARTEFACTS_DIR / "shap_v7_4.png", dpi=300)
    plt.close()

    model.save_model(ARTEFACTS_DIR / "model_v7_4.cbm")
    meta = {
        "version": "v7.4-relaxed",
        "r2": float(r2),
        "mae": float(mae),
        "features": FEATURES,
    }
    (ARTEFACTS_DIR / "metadata_v7_4.json").write_text(json.dumps(meta, indent=2))

    logger.success("v7.4 COMPLETE – Honest MAE = ${:,.0f}", mae)


@click.command()
def main():
    train, val, test = load_and_prepare()
    train_final_model(train, val, test)


if __name__ == "__main__":
    main()
