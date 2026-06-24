# filename: tests/test_data_clean.py
# purpose:  Unit tests for src/data/clean.py -- cleaning pipeline steps
# version:  1.0

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    drop_duplicates,
    fix_decimal_errors,
    zero_to_nan,
    impute_dimensions,
    clean_dataframe,
)


def _make_df(**overrides) -> pd.DataFrame:
    """Build a small synthetic diamonds-like DataFrame."""
    base = {
        "carat": [0.5, 0.6, 0.7, 0.8, 0.9],
        "cut": ["Ideal"] * 5,
        "color": ["E"] * 5,
        "clarity": ["VS2"] * 5,
        "depth": [61.0] * 5,
        "table": [55.0] * 5,
        "price": [1000, 1200, 1400, 1600, 1800],
        "x": [5.0, 5.2, 5.4, 5.6, 5.8],
        "y": [5.0, 5.2, 5.4, 5.6, 5.8],
        "z": [3.1, 3.2, 3.3, 3.4, 3.5],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestDropDuplicates:
    def test_removes_exact_dupes(self):
        df = _make_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        assert len(df) == 6
        result = drop_duplicates(df)
        assert len(result) == 5

    def test_no_dupes_unchanged(self):
        df = _make_df()
        result = drop_duplicates(df)
        assert len(result) == 5


class TestFixDecimalErrors:
    def test_corrects_outlier_dimension(self):
        df = _make_df(y=[5.0, 5.2, 54.0, 5.6, 5.8])
        result = fix_decimal_errors(df)
        assert result.loc[2, "y"] == pytest.approx(5.4)

    def test_no_change_on_normal_data(self):
        df = _make_df()
        result = fix_decimal_errors(df)
        pd.testing.assert_frame_equal(result, df)


class TestZeroToNan:
    def test_converts_zeros(self):
        df = _make_df(z=[3.1, 0, 3.3, 3.4, 0])
        result = zero_to_nan(df)
        assert result["z"].isna().sum() == 2

    def test_preserves_nonzero(self):
        df = _make_df()
        result = zero_to_nan(df)
        assert result["z"].isna().sum() == 0


class TestImputeDimensions:
    def test_fills_nans(self):
        df = _make_df()
        df.loc[1, "z"] = np.nan
        result, _ = impute_dimensions(df)
        assert result["z"].isna().sum() == 0

    def test_returns_params_dict(self):
        df = _make_df()
        df.loc[1, "z"] = np.nan
        _, params = impute_dimensions(df)
        assert "z" in params
        assert "coef" in params["z"]
        assert "intercept" in params["z"]


class TestCleanDataframe:
    def test_end_to_end(self):
        df = _make_df(z=[3.1, 0, 3.3, 3.4, 3.5])
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result, params = clean_dataframe(df)
        assert result.isnull().sum().sum() == 0
        assert len(result) == 5
