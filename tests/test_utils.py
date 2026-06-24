# filename: tests/test_utils.py
# purpose:  Unit tests for src/utils/helpers.py -- NumpyEncoder, save_json
# version:  1.0

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.helpers import NumpyEncoder, save_json


class TestNumpyEncoder:
    def test_encodes_numpy_int(self):
        result = json.dumps({"val": np.int64(42)}, cls=NumpyEncoder)
        assert json.loads(result) == {"val": 42}

    def test_encodes_numpy_float(self):
        result = json.dumps({"val": np.float64(3.14)}, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["val"] == pytest.approx(3.14)

    def test_encodes_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = json.dumps({"val": arr}, cls=NumpyEncoder)
        assert json.loads(result) == {"val": [1, 2, 3]}

    def test_raises_on_unsupported_type(self):
        with pytest.raises(TypeError):
            json.dumps({"val": object()}, cls=NumpyEncoder)


class TestSaveJson:
    def test_creates_file(self, tmp_path: Path):
        path = tmp_path / "out.json"
        save_json({"key": "value"}, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "out.json"
        save_json({"key": "value"}, path)
        assert path.exists()

    def test_writes_valid_json(self, tmp_path: Path):
        path = tmp_path / "out.json"
        data = {"a": 1, "b": [2, 3]}
        save_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_handles_numpy_types(self, tmp_path: Path):
        path = tmp_path / "np.json"
        save_json({"val": np.float64(1.5)}, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["val"] == 1.5
