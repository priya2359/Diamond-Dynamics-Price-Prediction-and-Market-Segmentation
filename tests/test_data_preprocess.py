# filename: tests/test_data_preprocess.py
# purpose:  Unit tests for src/data/preprocess.py -- outlier handling + transforms
# version:  1.0

import numpy as np
import pandas as pd
import pytest

import config
from src.data.preprocess import (
    _iqr_fences,
    cap_iqr_outliers,
    cap_zscore_outliers,
    apply_transforms,
)


def _make_series(values: list) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestIqrFences:
    def test_symmetric_data(self):
        s = _make_series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lo, hi = _iqr_fences(s)
        assert lo < s.min()
        assert hi > s.max()

    def test_fence_width_scales_with_iqr(self):
        narrow = _make_series([4, 5, 5, 5, 6])
        wide = _make_series([1, 3, 5, 7, 9])
        lo_n, hi_n = _iqr_fences(narrow)
        lo_w, hi_w = _iqr_fences(wide)
        assert (hi_w - lo_w) > (hi_n - lo_n)


class TestCapIqrOutliers:
    def test_clips_values(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        result, fences = cap_iqr_outliers(df, ["val"])
        assert result["val"].max() <= fences["val"]["upper"]

    def test_returns_fences_dict(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        _, fences = cap_iqr_outliers(df, ["val"])
        assert "val" in fences
        assert "lower" in fences["val"]
        assert "upper" in fences["val"]


class TestCapZscoreOutliers:
    def test_respects_physical_bounds(self):
        depth_vals = list(range(55, 72)) + [45.0, 80.0]
        df = pd.DataFrame({"depth": depth_vals})
        result, fences = cap_zscore_outliers(df, ["depth"])
        assert result["depth"].min() >= config.DEPTH_PHYSICAL_BOUNDS[0]
        assert result["depth"].max() <= config.DEPTH_PHYSICAL_BOUNDS[1]


class TestApplyTransforms:
    def test_log1p_applied(self):
        df = pd.DataFrame({"price": [100.0, 200.0], "dimension_ratio": [1.5, 2.0]})
        result = apply_transforms(df)
        assert result["price"].iloc[0] == pytest.approx(np.log1p(100.0))

    def test_sqrt_applied(self):
        df = pd.DataFrame({
            "carat": [0.49, 1.0],
            "volume": [100.0, 200.0],
            "table": [57.0, 60.0],
            "price_per_carat": [2000.0, 3000.0],
        })
        result = apply_transforms(df)
        assert result["carat"].iloc[0] == pytest.approx(np.sqrt(0.49))
        assert result["volume"].iloc[1] == pytest.approx(np.sqrt(200.0))

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"other": [1.0, 2.0]})
        result = apply_transforms(df)
        assert result["other"].iloc[0] == 1.0
