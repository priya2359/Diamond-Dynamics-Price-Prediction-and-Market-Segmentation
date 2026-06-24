# filename: tests/test_data_quality.py
# purpose:  Data validation tests on actual processed artifacts
# version:  1.0

import pandas as pd
import pytest


class TestProcessedCsv:
    @pytest.fixture(scope="class")
    def df(self, processed_data_dir):
        return pd.read_csv(processed_data_dir / "diamonds_processed.csv")

    def test_expected_row_count(self, df):
        assert len(df) == 53794

    def test_no_nans(self, df):
        assert df.isnull().sum().sum() == 0

    def test_numeric_columns_are_numeric(self, df):
        for col in ["carat", "depth", "table", "price", "volume"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_no_negative_prices(self, df):
        assert (df["price"] >= 0).all()

    def test_has_expected_columns(self, df):
        expected = {"carat", "depth", "table", "price", "volume", "cut", "color", "clarity"}
        assert expected.issubset(set(df.columns))
