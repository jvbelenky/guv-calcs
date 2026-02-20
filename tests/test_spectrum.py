"""Tests for spectrum file parsing (CSV and Excel support)."""

import io
import pytest
import pandas as pd
from guv_calcs.lamp.spectrum import Spectrum
from guv_calcs.io._data_loaders import load_spectrum_file


class TestSpectrumFromFile:
    """Tests for load_spectrum_file() and Spectrum.from_file()."""

    def test_simple_csv_bytes(self):
        """CSV bytes with a header row and numeric data."""
        csv_content = b"wavelength,intensity\n200,0.1\n250,0.5\n300,1.0\n350,0.3\n"
        result = load_spectrum_file(csv_content)
        assert len(result) == 4
        assert result[0] == (200.0, 0.1)
        assert result[2] == (300.0, 1.0)

    def test_csv_with_multiple_header_rows(self):
        """CSV with several non-numeric header/metadata rows."""
        csv_content = (
            b"Instrument: SpectroMax 3000\n"
            b"Date: 2024-01-15\n"
            b"Operator: John Doe\n"
            b"wavelength,intensity\n"
            b"200,0.1\n"
            b"250,0.5\n"
            b"300,1.0\n"
        )
        result = load_spectrum_file(csv_content)
        assert len(result) == 3
        assert result[0] == (200.0, 0.1)

    def test_multi_column_csv(self):
        """CSV with extra columns - only first two should be used."""
        csv_content = (
            b"wavelength,intensity,extra1,extra2\n"
            b"200,0.1,999,888\n"
            b"250,0.5,999,888\n"
            b"300,1.0,999,888\n"
        )
        result = load_spectrum_file(csv_content)
        assert len(result) == 3
        assert result[1] == (250.0, 0.5)

    def test_pure_numeric_csv(self):
        """CSV with no header at all - just numeric data."""
        csv_content = b"200,0.1\n250,0.5\n300,1.0\n"
        result = load_spectrum_file(csv_content)
        assert len(result) == 3
        assert result[0] == (200.0, 0.1)

    def test_excel_xlsx_roundtrip(self):
        """Create an xlsx in memory, then parse it back."""
        df = pd.DataFrame({0: [200, 250, 300], 1: [0.1, 0.5, 1.0]})
        buf = io.BytesIO()
        df.to_excel(buf, index=False, header=False, engine="openpyxl")
        xlsx_bytes = buf.getvalue()

        result = load_spectrum_file(xlsx_bytes)
        assert len(result) == 3
        assert result[0] == (200.0, 0.1)
        assert result[2] == (300.0, 1.0)

    def test_excel_with_header_rows(self):
        """Excel file with metadata headers before numeric data."""
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Instrument", "SpectroMax 3000"])
        ws.append(["Date", "2024-01-15"])
        ws.append(["Operator", "John Doe"])
        ws.append(["wavelength", "intensity"])
        ws.append([200, 0.1])
        ws.append([250, 0.5])
        ws.append([300, 1.0])

        buf = io.BytesIO()
        wb.save(buf)
        xlsx_bytes = buf.getvalue()

        result = load_spectrum_file(xlsx_bytes)
        assert len(result) == 3
        assert result[0] == (200.0, 0.1)
        assert result[2] == (300.0, 1.0)

    def test_no_numeric_data_raises(self):
        """File with no numeric data should raise ValueError."""
        csv_content = b"header1,header2\nfoo,bar\nbaz,qux\n"
        with pytest.raises(ValueError, match="No numeric data"):
            load_spectrum_file(csv_content)

    def test_unsupported_extension_raises(self):
        """File path with unsupported extension should raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported file extension"):
            load_spectrum_file("/fake/path/file.json")

    def test_spectrum_from_file_csv_bytes(self):
        """Spectrum.from_file() works with CSV bytes end-to-end."""
        csv_content = b"wavelength,intensity\n200,0.1\n250,0.5\n300,1.0\n"
        spectrum = Spectrum.from_file(csv_content)
        assert len(spectrum.wavelengths) == 3
        assert spectrum.wavelengths == (200.0, 250.0, 300.0)
        assert spectrum.intensities == (0.1, 0.5, 1.0)

    def test_spectrum_from_file_excel_bytes(self):
        """Spectrum.from_file() works with Excel bytes end-to-end."""
        df = pd.DataFrame({0: [200, 250, 300], 1: [0.1, 0.5, 1.0]})
        buf = io.BytesIO()
        df.to_excel(buf, index=False, header=False, engine="openpyxl")
        xlsx_bytes = buf.getvalue()

        spectrum = Spectrum.from_file(xlsx_bytes)
        assert len(spectrum.wavelengths) == 3
        assert spectrum.peak_wavelength == 300.0
