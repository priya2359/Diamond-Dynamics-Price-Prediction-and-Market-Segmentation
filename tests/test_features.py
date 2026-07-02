# filename: tests/test_features.py
# purpose:  Unit tests for src/features/engineer.py and src/features/encode.py
# version:  1.0

import pandas as pd
import pytest

import config
from src.features.engineer import (
    add_volume,
    add_price_per_carat,
    add_dimension_ratio,
    add_carat_category,
    add_price_inr,
    engineer_features,
)
from src.features.encode import encode_ordinal, build_ordinal_encoder


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "carat": [0.5, 1.0, 2.0],
        "x": [5.0, 6.0, 7.0],
        "y": [5.0, 6.0, 7.0],
        "z": [3.0, 3.5, 4.0],
        "price": [1000, 3000, 8000],
        "depth": [61.0, 62.0, 63.0],
        "table": [55.0, 57.0, 59.0],
        "cut": ["Ideal", "Good", "Fair"],
        "color": ["E", "G", "J"],
        "clarity": ["VS2", "SI1", "I1"],
    })


class TestAddVolume:
    def test_formula(self):
        df = _base_df()
        result = add_volume(df)
        assert result["volume"].iloc[0] == pytest.approx(5.0 * 5.0 * 3.0)

    def test_column_added(self):
        df = _base_df()
        result = add_volume(df)
        assert "volume" in result.columns


class TestAddPricePerCarat:
    def test_formula(self):
        df = _base_df()
        result = add_price_per_carat(df)
        assert result["price_per_carat"].iloc[0] == pytest.approx(1000 / 0.5)


class TestAddDimensionRatio:
    def test_formula(self):
        df = _base_df()
        result = add_dimension_ratio(df)
        expected = (5.0 + 5.0) / (2 * 3.0)
        assert result["dimension_ratio"].iloc[0] == pytest.approx(expected)


class TestAddCaratCategory:
    def test_light(self):
        df = pd.DataFrame({"carat": [0.3]})
        result = add_carat_category(df)
        assert result["carat_category"].iloc[0] == "Light"

    def test_medium(self):
        df = pd.DataFrame({"carat": [0.7]})
        result = add_carat_category(df)
        assert result["carat_category"].iloc[0] == "Medium"

    def test_heavy(self):
        df = pd.DataFrame({"carat": [2.0]})
        result = add_carat_category(df)
        assert result["carat_category"].iloc[0] == "Heavy"

    def test_boundary_050_is_medium(self):
        df = pd.DataFrame({"carat": [0.5]})
        result = add_carat_category(df)
        assert result["carat_category"].iloc[0] == "Medium"

    def test_boundary_150_is_heavy(self):
        df = pd.DataFrame({"carat": [1.5]})
        result = add_carat_category(df)
        assert result["carat_category"].iloc[0] == "Heavy"


class TestAddPriceInr:
    def test_uses_config_rate(self):
        df = pd.DataFrame({"price": [100.0]})
        result = add_price_inr(df)
        assert result["price_inr"].iloc[0] == pytest.approx(100.0 * config.USD_TO_INR)


class TestEngineerFeatures:
    def test_all_columns_present(self):
        df = _base_df()
        result = engineer_features(df)
        for col in ["volume", "price_per_carat", "dimension_ratio", "carat_category", "price_inr"]:
            assert col in result.columns


class TestEncodeOrdinal:
    def test_correct_codes(self):
        df = pd.DataFrame({
            "cut": ["Fair", "Ideal"],
            "color": ["J", "D"],
            "clarity": ["I1", "IF"],
            "carat_category": ["Light", "Heavy"],
        })
        result, mapping = encode_ordinal(df)
        assert result["cut"].iloc[0] == 0
        assert result["cut"].iloc[1] == 4
        assert result["color"].iloc[0] == 0
        assert result["color"].iloc[1] == 6
        assert result["clarity"].iloc[0] == 0
        assert result["clarity"].iloc[1] == 7

    def test_ordering_preserved(self):
        df = pd.DataFrame({"cut": config.CUT_ORDER})
        result, _ = encode_ordinal(df)
        assert list(result["cut"]) == list(range(5))


class TestBuildOrdinalEncoder:
    def test_categories_match_config(self):
        encoder = build_ordinal_encoder()
        expected_cats = [
            config.CARAT_CATEGORY_LABELS,
            config.CLARITY_ORDER,
            config.COLOR_ORDER,
            config.CUT_ORDER,
        ]
        for enc_cats, exp_cats in zip(encoder.categories, expected_cats):
            assert list(enc_cats) == list(exp_cats)
