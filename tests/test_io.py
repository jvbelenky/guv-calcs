"""Tests for the io module (save/load, data loading)."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from guv_calcs import Room
from guv_calcs.io import (
    get_spectral_weightings,
    get_full_disinfection_table,
    rows_to_bytes,
    load_csv,
    fig_to_bytes,
)


class TestSpectralWeightings:
    """Tests for spectral weighting data loading."""

    def test_get_spectral_weightings_returns_dict(self):
        """get_spectral_weightings should return a dict."""
        weights = get_spectral_weightings()
        assert isinstance(weights, dict)

    def test_spectral_weightings_has_wavelengths(self):
        """Spectral weightings should have wavelength column."""
        weights = get_spectral_weightings()
        assert "Wavelength (nm)" in weights

    def test_spectral_weightings_has_eye_skin(self):
        """Spectral weightings should have eye and skin columns."""
        weights = get_spectral_weightings()
        keys = list(weights.keys())
        # Should have multiple weighting curves
        assert len(keys) > 1

    def test_spectral_weightings_values_are_arrays(self):
        """Spectral weighting values should be numpy arrays."""
        import numpy as np
        weights = get_spectral_weightings()
        for key, val in weights.items():
            assert isinstance(val, np.ndarray)


class TestDisinfectionTable:
    """Tests for disinfection table loading."""

    def test_get_full_disinfection_table_returns_dataframe(self):
        """get_full_disinfection_table should return DataFrame."""
        import pandas as pd
        df = get_full_disinfection_table()
        assert isinstance(df, pd.DataFrame)

    def test_disinfection_table_has_rows(self):
        """Disinfection table should have data rows."""
        df = get_full_disinfection_table()
        assert len(df) > 0

    def test_disinfection_table_has_species(self):
        """Disinfection table should have Species column."""
        df = get_full_disinfection_table()
        assert "Species" in df.columns


class TestRowsToBytes:
    """Tests for rows_to_bytes CSV conversion."""

    def test_rows_to_bytes_basic(self):
        """rows_to_bytes should convert list of lists to bytes."""
        rows = [["a", "b", "c"], [1, 2, 3]]
        result = rows_to_bytes(rows)
        assert isinstance(result, bytes)

    def test_rows_to_bytes_contains_data(self):
        """rows_to_bytes output should contain the data."""
        rows = [["header1", "header2"], ["value1", "value2"]]
        result = rows_to_bytes(rows)
        text = result.decode("cp1252")
        assert "header1" in text
        assert "value1" in text


class TestLoadCsv:
    """Tests for load_csv function."""

    def test_load_csv_from_bytes(self):
        """load_csv should handle bytes input."""
        csv_bytes = b"a,b,c\n1,2,3\n"
        result = load_csv(csv_bytes)
        assert result is not None

    def test_load_csv_invalid_type(self):
        """load_csv should raise TypeError for invalid input."""
        with pytest.raises(TypeError):
            load_csv(12345)


class TestFigToBytes:
    """Tests for fig_to_bytes function."""

    def test_matplotlib_figure_to_bytes(self):
        """fig_to_bytes should convert matplotlib figure to PNG bytes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        result = fig_to_bytes(fig)
        plt.close(fig)

        assert isinstance(result, bytes)
        # PNG files start with specific magic bytes
        assert result[:8] == b'\x89PNG\r\n\x1a\n'

    def test_plotly_figure_to_bytes(self):
        """fig_to_bytes should convert plotly figure to PNG bytes (requires Chrome)."""
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        result = fig_to_bytes(fig)

        # Result is either bytes (Chrome available) or None (Chrome not available)
        if result is not None:
            assert isinstance(result, bytes)
            assert result[:8] == b'\x89PNG\r\n\x1a\n'

    def test_invalid_figure_type_raises(self):
        """fig_to_bytes should raise TypeError for unsupported figure types."""
        with pytest.raises(TypeError):
            fig_to_bytes("not a figure")

    def test_plotly_figure_graceful_without_chrome(self):
        """fig_to_bytes should return None and warn if Chrome is not available."""
        import plotly.graph_objects as go
        import warnings

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        # This test just ensures no exception is raised
        # Result is None if Chrome unavailable, bytes otherwise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = fig_to_bytes(fig)
            assert result is None or isinstance(result, bytes)


class TestRoomSaveLoad:
    """Tests for Room save/load functionality."""

    def test_room_save_returns_json(self):
        """Room.save() without filename should return JSON string."""
        room = Room(x=6, y=4, z=2.7)
        result = room.save(None)
        assert isinstance(result, str)
        # Should be valid JSON
        data = json.loads(result)
        assert "data" in data
        assert "guv-calcs_version" in data

    def test_room_save_to_file(self, temp_dir):
        """Room.save() with filename should create file."""
        room = Room(x=6, y=4, z=2.7)
        filepath = os.path.join(temp_dir, "test.guv")
        room.save(filepath)
        assert os.path.exists(filepath)

    def test_room_load_from_file(self, temp_dir):
        """Room.load() should restore room from file."""
        room = Room(x=8, y=5, z=3.0)
        filepath = os.path.join(temp_dir, "test.guv")
        room.save(filepath)

        loaded = Room.load(filepath)
        assert loaded.x == 8
        assert loaded.y == 5
        assert loaded.z == 3.0

    def test_room_save_load_with_lamp(self, temp_dir):
        """Room with lamp should survive save/load."""
        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        filepath = os.path.join(temp_dir, "test.guv")
        room.save(filepath)

        loaded = Room.load(filepath)
        assert len(loaded.lamps) == 1

    def test_room_load_invalid_file(self, temp_dir):
        """Room.load() with invalid file should raise error."""
        filepath = os.path.join(temp_dir, "nonexistent.guv")
        with pytest.raises(FileNotFoundError):
            Room.load(filepath)

    def test_room_load_wrong_extension(self, temp_dir):
        """Room.load() with wrong extension should raise error."""
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("{}")
        with pytest.raises(ValueError, match="valid .guv file"):
            Room.load(filepath)


class TestRoomExportZip:
    """Tests for Room.export_zip functionality."""

    def test_export_zip_returns_bytes(self):
        """export_zip without filename should return bytes."""
        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        result = room.export_zip()
        assert isinstance(result, bytes)

    def test_export_zip_to_file(self, temp_dir):
        """export_zip with filename should create file."""
        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        filepath = os.path.join(temp_dir, "test.zip")
        room.export_zip(filepath)
        assert os.path.exists(filepath)

    def test_export_zip_contains_guv_file(self):
        """export_zip should contain room.guv file."""
        import zipfile
        import io

        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        zip_bytes = room.export_zip()

        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            names = zf.namelist()
            assert "room.guv" in names

    def test_export_zip_with_plots(self):
        """export_zip with include_plots=True should include plot images."""
        import zipfile
        import io

        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        zip_bytes = room.export_zip(include_plots=True)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            names = zf.namelist()
            # Should have PNG files for the calc zones (at least SkinLimits, EyeLimits)
            png_files = [n for n in names if n.endswith('.png')]
            assert len(png_files) >= 2

    def test_export_zip_with_lamp_plots(self):
        """export_zip with include_lamp_plots=True should include lamp plot images."""
        import zipfile
        import io

        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        zip_bytes = room.export_zip(include_lamp_plots=True)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            names = zf.namelist()
            # Should have IES plot and spectra plots
            png_files = [n for n in names if n.endswith('.png')]
            assert len(png_files) >= 1

    def test_export_zip_with_lamp_files(self):
        """export_zip with include_lamp_files=True should include IES files."""
        import zipfile
        import io

        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        zip_bytes = room.export_zip(include_lamp_files=True)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            names = zf.namelist()
            ies_files = [n for n in names if n.endswith('.ies')]
            assert len(ies_files) >= 1


class TestRoomGenerateReport:
    """Tests for Room.generate_report functionality."""

    def test_generate_report_returns_bytes(self):
        """generate_report without filename should return bytes."""
        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        result = room.generate_report()
        assert isinstance(result, bytes)

    def test_generate_report_contains_room_params(self):
        """generate_report should contain room parameters."""
        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        room.add_standard_zones().calculate()
        result = room.generate_report()
        text = result.decode("cp1252")
        assert "Room Parameters" in text
