# filename: tests/test_model_golden.py
# purpose:  Golden input model regression tests -- verify saved models produce expected outputs
# version:  1.0

import pytest

from src.inference.prepare_input import predict_price, predict_segment


class TestRegressionGoldenInput:
    def test_price_in_expected_range(self, regression_pipeline, valid_diamond):
        result = predict_price(regression_pipeline, **valid_diamond)
        assert 1000 < result["price_usd"] < 6000

    def test_price_is_deterministic(self, regression_pipeline, valid_diamond):
        r1 = predict_price(regression_pipeline, **valid_diamond)
        r2 = predict_price(regression_pipeline, **valid_diamond)
        assert r1["price_usd"] == pytest.approx(r2["price_usd"])


class TestClusteringGoldenInput:
    def test_cluster_assignment(self, clustering_pipeline, cluster_profiles, regression_pipeline, valid_diamond):
        price = predict_price(regression_pipeline, **valid_diamond)
        result = predict_segment(
            clustering_pipeline, cluster_profiles,
            price_usd=price["price_usd"], **valid_diamond,
        )
        assert result["cluster_id"] in (0, 1)
        assert result["centroid_distance"] >= 0
        assert result["centroid_distance_p95"] > 0
