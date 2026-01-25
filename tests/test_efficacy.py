"""Tests for the efficacy module (Data class, disinfection calculations)."""

import pytest
import numpy as np
import pandas as pd
from guv_calcs import Data
from guv_calcs.efficacy._kinetics import (
    species_matches,
    parse_resistant,
    extract_kinetic_params,
    filter_wavelengths,
    compute_row,
)
from guv_calcs.efficacy._filtering import (
    filter_by_column,
    validate_filter,
    apply_row_filters,
    get_effective_wavelengths,
)
from guv_calcs.efficacy._state import DataState


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


class TestAverageValueParametric:
    """Tests for average_value with parametric inputs."""

    def test_average_value_single_function(self):
        """average_value with single function returns float."""
        data = Data(fluence=0.5)
        result = data.average_value("log2", species="coli")
        assert isinstance(result, float)

    def test_average_value_function_list(self):
        """average_value with function list returns dict keyed by function."""
        data = Data(fluence=0.5)
        result = data.average_value(["log2", "log3"], species="coli")
        assert isinstance(result, dict)
        assert "log2" in result
        assert "log3" in result
        assert isinstance(result["log2"], float)
        assert isinstance(result["log3"], float)
        # log3 should require more time than log2
        assert result["log3"] > result["log2"]

    def test_average_value_function_list_with_species_list(self):
        """average_value with both function and species lists returns nested dict."""
        data = Data(fluence=0.5)
        result = data.average_value(["log2", "each"], species=["coli", "staph"])
        assert isinstance(result, dict)
        assert "log2" in result
        assert "each" in result
        # Each function key should have a dict of species
        assert isinstance(result["log2"], dict)
        assert "coli" in result["log2"]
        assert "staph" in result["log2"]

    def test_average_value_function_list_order(self):
        """Function list should be outermost dimension in nested dict."""
        data = Data(fluence=0.5)
        result = data.average_value(["log2", "log3"], species=["coli"])
        # Function is outermost, so result["log2"]["coli"] should exist
        assert "log2" in result
        assert "coli" in result["log2"]


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


class TestKineticsFunctions:
    """Tests for _kinetics module functions."""

    def test_species_matches_exact(self):
        """species_matches should match exact species names."""
        assert species_matches("Escherichia coli", "Escherichia coli")

    def test_species_matches_partial(self):
        """species_matches should match partial names (all query words in target)."""
        assert species_matches("e. coli", "Escherichia coli")
        assert species_matches("coli", "Escherichia coli")

    def test_species_matches_case_insensitive(self):
        """species_matches should be case-insensitive."""
        assert species_matches("E. COLI", "Escherichia coli")
        assert species_matches("STAPH", "Staphylococcus aureus")

    def test_species_matches_multiple_words(self):
        """species_matches should match all query words."""
        assert species_matches("staph aureus", "Staphylococcus aureus")
        assert not species_matches("staph coli", "Staphylococcus aureus")

    def test_species_matches_no_match(self):
        """species_matches should return False for non-matching names."""
        assert not species_matches("salmonella", "Escherichia coli")

    def test_parse_resistant_percentage(self):
        """parse_resistant should parse percentage strings."""
        assert parse_resistant("0.33%") == pytest.approx(0.0033)
        assert parse_resistant("1%") == pytest.approx(0.01)
        assert parse_resistant("100%") == pytest.approx(1.0)

    def test_parse_resistant_nan(self):
        """parse_resistant should return 0.0 for NaN values."""
        assert parse_resistant(pd.NA) == 0.0
        assert parse_resistant(np.nan) == 0.0

    def test_parse_resistant_numeric(self):
        """parse_resistant should pass through numeric values."""
        assert parse_resistant(0.01) == 0.01
        assert parse_resistant(0.5) == 0.5

    def test_extract_kinetic_params(self):
        """extract_kinetic_params should extract k1, k2, f from row."""
        row = pd.Series({
            "k1 [cm2/mJ]": 0.1,
            "k2 [cm2/mJ]": 0.01,
            "% resistant": "1%"
        })
        params = extract_kinetic_params(row)
        assert params["k1"] == 0.1
        assert params["k2"] == 0.01
        assert params["f"] == pytest.approx(0.01)

    def test_extract_kinetic_params_nan(self):
        """extract_kinetic_params should handle NaN values."""
        row = pd.Series({
            "k1 [cm2/mJ]": np.nan,
            "k2 [cm2/mJ]": np.nan,
            "% resistant": np.nan
        })
        params = extract_kinetic_params(row)
        assert params["k1"] == 0.0
        assert params["k2"] == 0.0
        assert params["f"] == 0.0

    def test_compute_row_scalar_fluence(self):
        """compute_row should compute with scalar fluence."""
        from guv_calcs.efficacy import eACH_UV
        row = pd.Series({
            "wavelength [nm]": 222,
            "k1 [cm2/mJ]": 0.1,
            "k2 [cm2/mJ]": 0.0,
            "% resistant": None
        })
        result = compute_row(row, 10.0, eACH_UV)
        assert result is not None
        assert result > 0

    def test_compute_row_dict_fluence(self):
        """compute_row should use wavelength-specific fluence from dict."""
        from guv_calcs.efficacy import eACH_UV
        row = pd.Series({
            "wavelength [nm]": 254,
            "k1 [cm2/mJ]": 0.05,
            "k2 [cm2/mJ]": 0.0,
            "% resistant": None
        })
        fluence_dict = {222: 5.0, 254: 10.0}
        result = compute_row(row, fluence_dict, eACH_UV)
        assert result is not None
        assert result > 0

    def test_compute_row_missing_wavelength(self):
        """compute_row should return None if wavelength not in dict."""
        from guv_calcs.efficacy import eACH_UV
        row = pd.Series({
            "wavelength [nm]": 300,
            "k1 [cm2/mJ]": 0.05,
            "k2 [cm2/mJ]": 0.0,
            "% resistant": None
        })
        fluence_dict = {222: 5.0, 254: 10.0}
        result = compute_row(row, fluence_dict, eACH_UV)
        assert result is None


class TestDataState:
    """Tests for DataState dataclass."""

    def test_datastate_defaults(self):
        """DataState should have sensible defaults."""
        state = DataState()
        assert state.fluence is None
        assert state.volume_m3 is None
        assert state.medium is None
        assert state.category is None
        assert state.wavelength is None
        assert state.log == 2
        assert state.use_metric_units is True
        assert state.fluence_wavelengths is None
        assert state.time_cols == {}

    def test_datastate_with_values(self):
        """DataState should store provided values."""
        state = DataState(
            fluence=0.5,
            volume_m3=50.0,
            medium="Aerosol",
            category="Virus",
            log=3
        )
        assert state.fluence == 0.5
        assert state.volume_m3 == 50.0
        assert state.medium == "Aerosol"
        assert state.category == "Virus"
        assert state.log == 3

    def test_datastate_fluence_dict_initializes_wavelengths(self):
        """DataState should auto-set fluence_wavelengths from fluence dict."""
        state = DataState(fluence={222: 0.5, 254: 0.3})
        assert state.fluence_wavelengths == [222, 254]

    def test_datastate_fluence_scalar_no_wavelengths(self):
        """DataState should not set fluence_wavelengths for scalar fluence."""
        state = DataState(fluence=0.5)
        assert state.fluence_wavelengths is None


class TestFilteringFunctions:
    """Tests for _filtering module functions."""

    def test_filter_by_column_scalar(self):
        """filter_by_column should filter by scalar value."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        result = filter_by_column(df, "A", 2)
        assert len(result) == 1
        assert result["B"].iloc[0] == "y"

    def test_filter_by_column_list(self):
        """filter_by_column should filter by list of values."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        result = filter_by_column(df, "A", [1, 3])
        assert len(result) == 2
        assert list(result["B"]) == ["x", "z"]

    def test_filter_by_column_tuple_range(self):
        """filter_by_column should filter by tuple range (min, max)."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = filter_by_column(df, "A", (2, 4))
        assert list(result["A"]) == [2, 3, 4]

    def test_filter_by_column_none(self):
        """filter_by_column should return df unchanged if value is None."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = filter_by_column(df, "A", None)
        assert len(result) == 3

    def test_validate_filter_valid_scalar(self):
        """validate_filter should pass valid scalar."""
        result = validate_filter("Aerosol", ["Aerosol", "Surface"], "medium")
        assert result == "Aerosol"

    def test_validate_filter_invalid_scalar(self):
        """validate_filter should raise KeyError for invalid scalar."""
        with pytest.raises(KeyError):
            validate_filter("Invalid", ["Aerosol", "Surface"], "medium")

    def test_validate_filter_list_normalizes(self):
        """validate_filter should normalize single-item list to scalar."""
        result = validate_filter(["Aerosol"], ["Aerosol", "Surface"], "medium")
        assert result == "Aerosol"

    def test_validate_filter_list_invalid(self):
        """validate_filter should raise KeyError for invalid list items."""
        with pytest.raises(KeyError):
            validate_filter(["Aerosol", "Bad"], ["Aerosol", "Surface"], "medium")

    def test_get_effective_wavelengths_none(self):
        """get_effective_wavelengths should return None if no filters."""
        result = get_effective_wavelengths(None, None)
        assert result is None

    def test_get_effective_wavelengths_user_only(self):
        """get_effective_wavelengths should return user wavelength."""
        result = get_effective_wavelengths(222, None)
        assert result == [222]

    def test_get_effective_wavelengths_fluence_only(self):
        """get_effective_wavelengths should return fluence wavelengths."""
        result = get_effective_wavelengths(None, [222, 254])
        assert result == [222, 254]

    def test_get_effective_wavelengths_merged(self):
        """get_effective_wavelengths should merge user and fluence wavelengths."""
        result = get_effective_wavelengths(280, [222, 254])
        assert result == [222, 254, 280]

    def test_get_effective_wavelengths_tuple_passthrough(self):
        """get_effective_wavelengths should pass through tuple range."""
        result = get_effective_wavelengths((200, 300), [222])
        assert result == (200, 300)
