# filename: tests/test_inference.py
# purpose:  Unit tests for src/inference/prepare_input.py -- feature assembly + prediction
# version:  1.0

import numpy as np
import pytest

from src.inference.prepare_input import (
    _carat_category,
    build_regression_input,
    build_clustering_input,
    predict_price,
    predict_segment,
    REGRESSION_FEATURE_ORDER,
    CLUSTERING_FEATURE_ORDER,
)


class TestCaratCategory:
    def test_light(self):
        assert _carat_category(0.3) == "Light"

    def test_medium(self):
        assert _carat_category(0.7) == "Medium"

    def test_heavy(self):
        assert _carat_category(2.0) == "Heavy"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="outside category bins"):
            _carat_category(-1.0)


class TestBuildRegressionInput:
    def test_shape(self, valid_diamond):
        df = build_regression_input(**valid_diamond)
        assert df.shape == (1, 8)

    def test_column_order(self, valid_diamond):
        df = build_regression_input(**valid_diamond)
        assert list(df.columns) == REGRESSION_FEATURE_ORDER

    def test_sqrt_carat_applied(self, valid_diamond):
        df = build_regression_input(**valid_diamond)
        assert df["carat"].iloc[0] == pytest.approx(np.sqrt(0.7))

    def test_sqrt_volume_applied(self, valid_diamond):
        df = build_regression_input(**valid_diamond)
        volume = 5.7 * 5.71 * 3.53
        assert df["volume"].iloc[0] == pytest.approx(np.sqrt(volume))

    def test_zero_xy_sum_raises(self):
        with pytest.raises(ValueError, match="x \\+ y = 0"):
            build_regression_input(carat=0.5, x=0, y=0, z=3.0, cut="Ideal", color="E", clarity="VS2")


class TestBuildClusteringInput:
    def test_includes_price_per_carat(self, valid_diamond):
        df = build_clustering_input(**valid_diamond, price_usd=2000.0)
        assert "price_per_carat" in df.columns

    def test_column_order(self, valid_diamond):
        df = build_clustering_input(**valid_diamond, price_usd=2000.0)
        assert list(df.columns) == CLUSTERING_FEATURE_ORDER

    def test_zero_carat_raises(self):
        with pytest.raises(ValueError, match="carat must be > 0"):
            build_clustering_input(
                carat=0, x=5.0, y=5.0, z=3.0,
                cut="Ideal", color="E", clarity="VS2", price_usd=1000.0,
            )


class TestPredictPrice:
    def test_returns_positive_finite(self, regression_pipeline, valid_diamond):
        result = predict_price(regression_pipeline, **valid_diamond)
        assert result["price_usd"] > 0
        assert np.isfinite(result["price_usd"])

    def test_inr_conversion(self, regression_pipeline, valid_diamond):
        import config as cfg
        result = predict_price(regression_pipeline, **valid_diamond)
        assert result["price_inr"] == pytest.approx(result["price_usd"] * cfg.USD_TO_INR)

    def test_return_features_flag(self, regression_pipeline, valid_diamond):
        result = predict_price(regression_pipeline, **valid_diamond, return_features=True)
        assert "_features_df" in result
        assert result["_features_df"].shape == (1, 8)


class TestPredictSegment:
    def test_valid_cluster(self, clustering_pipeline, cluster_profiles, regression_pipeline, valid_diamond):
        price = predict_price(regression_pipeline, **valid_diamond)
        result = predict_segment(
            clustering_pipeline, cluster_profiles,
            price_usd=price["price_usd"], **valid_diamond,
        )
        assert result["cluster_id"] in (0, 1)
        assert isinstance(result["cluster_name"], str)
        assert isinstance(result["is_ood"], bool)
