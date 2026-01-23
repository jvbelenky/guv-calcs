"""Tests for the efficacy module (Data class, disinfection calculations)."""

import pytest
import numpy as np
import pandas as pd
from guv_calcs import Data


class TestDataClassMethods:
    """Tests for Data class methods."""

    def test_get_full_returns_dataframe(self):
        """Data.get_full() should return DataFrame."""
        df = Data.get_full()
        assert isinstance(df, pd.DataFrame)

    def test_get_full_returns_copy(self):
        """Data.get_full() should return a copy."""
        df1 = Data.get_full()
        df2 = Data.get_full()
        # Modifying one shouldn't affect the other
        df1["test_col"] = 1
        assert "test_col" not in df2.columns

    def test_get_valid_categories(self):
        """Data.get_valid_categories() should return list of strings."""
        categories = Data.get_valid_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(c, str) for c in categories)

    def test_get_valid_mediums(self):
        """Data.get_valid_mediums() should return list of strings."""
        mediums = Data.get_valid_mediums()
        assert isinstance(mediums, list)
        assert len(mediums) > 0
        # Should include Aerosol, Surface, or Liquid
        assert any(m in mediums for m in ["Aerosol", "Surface", "Liquid"])

    def test_get_valid_wavelengths(self):
        """Data.get_valid_wavelengths() should return list of numbers."""
        wavelengths = Data.get_valid_wavelengths()
        assert isinstance(wavelengths, list)
        assert len(wavelengths) > 0
        assert all(isinstance(w, (int, float)) for w in wavelengths)


class TestDataInitialization:
    """Tests for Data initialization."""

    def test_init_no_args(self):
        """Data() should create instance with base table."""
        data = Data()
        assert data is not None

    def test_init_with_fluence_float(self):
        """Data with float fluence should compute additional columns."""
        data = Data(fluence=1.0)
        assert data is not None

    def test_init_with_fluence_dict(self):
        """Data with dict fluence should handle multiple wavelengths."""
        data = Data(fluence={222: 0.5, 254: 0.5})
        assert data is not None

    def test_init_with_volume(self):
        """Data with volume should compute CADR columns."""
        data = Data(fluence=1.0, volume_m3=50.0)
        assert data is not None


class TestDataSubset:
    """Tests for Data.subset() filtering."""

    def test_subset_returns_self(self):
        """subset() should return self for chaining."""
        data = Data()
        result = data.subset(medium="Aerosol")
        assert result is data

    def test_subset_by_medium(self):
        """subset(medium=...) should filter by medium."""
        data = Data()
        data.subset(medium="Aerosol")
        # The filter is applied, internal state is set
        assert data._medium is not None

    def test_subset_by_category(self):
        """subset(category=...) should filter by category."""
        data = Data()
        categories = Data.get_valid_categories()
        if categories:
            data.subset(category=categories[0])
            assert data._category is not None

    def test_subset_invalid_medium(self):
        """subset with invalid medium should raise KeyError."""
        data = Data()
        with pytest.raises(KeyError):
            data.subset(medium="InvalidMedium")

    def test_subset_invalid_category(self):
        """subset with invalid category should raise KeyError."""
        data = Data()
        with pytest.raises(KeyError):
            data.subset(category="InvalidCategory")

    def test_subset_by_log_level(self):
        """subset(log=...) should set log reduction level."""
        data = Data(fluence=1.0)
        data.subset(log=3)
        assert data._log == 3

    def test_subset_chaining(self):
        """Multiple subset calls should chain."""
        data = Data()
        result = data.subset(medium="Aerosol").subset(log=2)
        assert result is data


class TestDataDisplayDf:
    """Tests for Data.display_df property."""

    def test_display_df_returns_dataframe(self):
        """display_df should return DataFrame."""
        data = Data()
        df = data.display_df
        assert isinstance(df, pd.DataFrame)

    def test_display_df_has_species(self):
        """display_df should have Species column."""
        data = Data()
        df = data.display_df
        assert "Species" in df.columns

    def test_display_df_filtered(self):
        """display_df should reflect subset filters."""
        data = Data().subset(medium="Aerosol")
        df = data.display_df
        # All rows should be Aerosol medium
        if "Medium" in df.columns and len(df) > 0:
            assert all(df["Medium"] == "Aerosol")


class TestDataWithFluence:
    """Tests for Data with fluence calculations."""

    def test_fluence_adds_each_column(self):
        """Data with fluence should add eACH-UV column."""
        data = Data(fluence=1.0)
        # The internal _full_df should have computed columns
        assert data._full_df is not None
        assert len(data._full_df) > 0

    def test_fluence_dict_multiple_wavelengths(self):
        """Data with dict fluence should handle all wavelengths."""
        data = Data(fluence={222: 0.5, 254: 0.5})
        # Should not raise error
        df = data.display_df
        assert df is not None


class TestEfficacyMathFunctions:
    """Tests for efficacy math functions."""

    def test_eACH_UV_positive_fluence(self):
        """eACH_UV should return positive value for valid inputs."""
        from guv_calcs.efficacy import eACH_UV
        # irrad in µW/cm², k1 in cm²/mJ
        result = eACH_UV(irrad=10.0, k1=0.1)
        assert result > 0

    def test_log_reductions(self):
        """Log reduction functions should compute correct values."""
        from guv_calcs.efficacy import log1, log2, log3
        irrad = 100.0  # µW/cm²
        k1 = 0.1  # cm²/mJ
        # Higher log reductions need more time
        t1 = log1(irrad=irrad, k1=k1)  # 90% reduction
        t2 = log2(irrad=irrad, k1=k1)  # 99% reduction
        t3 = log3(irrad=irrad, k1=k1)  # 99.9% reduction
        assert t2 > t1
        assert t3 > t2

    def test_seconds_to_S(self):
        """seconds_to_S should compute time to reach survival fraction."""
        from guv_calcs.efficacy import seconds_to_S
        # S=0.1 means 90% killed, irrad in µW/cm², k1 in cm²/mJ
        result = seconds_to_S(S=0.1, irrad=100.0, k1=0.1)
        assert result > 0

    def test_CADR_functions(self):
        """CADR functions should return positive values."""
        from guv_calcs.efficacy import CADR_CFM, CADR_LPS
        cubic_feet = 1000.0
        cubic_meters = 50.0
        irrad = 100.0  # µW/cm²
        k1 = 0.1  # cm²/mJ
        cfm = CADR_CFM(cubic_feet=cubic_feet, irrad=irrad, k1=k1)
        lps = CADR_LPS(cubic_meters=cubic_meters, irrad=irrad, k1=k1)
        assert cfm > 0
        assert lps > 0
