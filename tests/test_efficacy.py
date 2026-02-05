"""Tests for the efficacy module (InactivationData class, disinfection calculations)."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests

import pytest
import numpy as np
import pandas as pd
from guv_calcs import InactivationData
from guv_calcs.efficacy._kinetics import (
    species_matches,
    parse_resistant,
    extract_kinetic_params,
    filter_wavelengths,
    compute_row,
)
from guv_calcs.efficacy._filtering import (
    filter_by_column,
    words_match,
    filter_by_words,
    apply_row_filters,
    get_effective_wavelengths,
    ALIASES,
)
from guv_calcs.efficacy._state import DataState


class TestInactivationDataClassMethods:
    """Tests for InactivationData class methods."""

    def test_get_full_returns_dataframe(self):
        """InactivationData.get_full() should return DataFrame."""
        df = InactivationData.get_full()
        assert isinstance(df, pd.DataFrame)

    def test_get_full_returns_copy(self):
        """InactivationData.get_full() should return a copy."""
        df1 = InactivationData.get_full()
        df2 = InactivationData.get_full()
        # Modifying one shouldn't affect the other
        df1["test_col"] = 1
        assert "test_col" not in df2.columns

    def test_get_valid_categories(self):
        """InactivationData.get_valid_categories() should return list of strings."""
        categories = InactivationData.get_valid_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(c, str) for c in categories)

    def test_get_valid_mediums(self):
        """InactivationData.get_valid_mediums() should return list of strings."""
        mediums = InactivationData.get_valid_mediums()
        assert isinstance(mediums, list)
        assert len(mediums) > 0
        # Should include Aerosol, Surface, or Liquid
        assert any(m in mediums for m in ["Aerosol", "Surface", "Liquid"])

    def test_get_valid_wavelengths(self):
        """InactivationData.get_valid_wavelengths() should return list of numbers."""
        wavelengths = InactivationData.get_valid_wavelengths()
        assert isinstance(wavelengths, list)
        assert len(wavelengths) > 0
        assert all(isinstance(w, (int, float)) for w in wavelengths)


class TestInactivationDataInitialization:
    """Tests for InactivationData initialization."""

    def test_init_no_args(self):
        """InactivationData() should create instance with base table."""
        data = InactivationData()
        assert data is not None

    def test_init_with_fluence_float(self):
        """InactivationData with float fluence should compute additional columns."""
        data = InactivationData(fluence=1.0)
        assert data is not None

    def test_init_with_fluence_dict(self):
        """InactivationData with dict fluence should handle multiple wavelengths."""
        data = InactivationData(fluence={222: 0.5, 254: 0.5})
        assert data is not None

    def test_init_with_volume(self):
        """InactivationData with volume should compute CADR columns."""
        data = InactivationData(fluence=1.0, volume_m3=50.0)
        assert data is not None


class TestInactivationDataSubset:
    """Tests for InactivationData.subset() filtering."""

    def test_subset_returns_self(self):
        """subset() should return self for chaining."""
        data = InactivationData()
        result = data.subset(medium="Aerosol")
        assert result is data

    def test_subset_by_medium(self):
        """subset(medium=...) should filter by medium."""
        data = InactivationData()
        data.subset(medium="Aerosol")
        # The filter is applied, internal state is set
        assert data._medium is not None

    def test_subset_by_category(self):
        """subset(category=...) should filter by category."""
        data = InactivationData()
        categories = InactivationData.get_valid_categories()
        if categories:
            data.subset(category=categories[0])
            assert data._category is not None

    def test_subset_by_species(self):
        """subset(species=...) should filter by species."""
        data = InactivationData()
        data.subset(species="coli")
        assert data._species is not None
        # Check that full_df is filtered (display_df may remove single-value columns)
        df = data.full_df
        assert all("coli" in s.lower() for s in df["Species"])

    def test_subset_by_strain(self):
        """subset(strain=...) should filter by strain."""
        data = InactivationData()
        data.subset(strain="ATCC")
        assert data._strain is not None

    def test_subset_by_condition(self):
        """subset(condition=...) should filter by condition."""
        data = InactivationData()
        data.subset(condition="stationary")
        assert data._condition is not None

    def test_subset_by_log_level(self):
        """subset(log=...) should set log reduction level."""
        data = InactivationData(fluence=1.0)
        data.subset(log=3)
        assert data._log == 3

    def test_subset_chaining(self):
        """Multiple subset calls should chain."""
        data = InactivationData()
        result = data.subset(medium="Aerosol").subset(log=2)
        assert result is data

    def test_subset_case_insensitive_medium(self):
        """subset() should match medium case-insensitively."""
        data = InactivationData().subset(medium="aerosol")
        df = data.full_df
        assert len(df) > 0
        assert all("Aerosol" in m for m in df["Medium"])

    def test_subset_alias_air_to_aerosol(self):
        """subset(medium='air') should match Aerosol."""
        data = InactivationData().subset(medium="air")
        df = data.full_df
        assert len(df) > 0
        assert all("Aerosol" in m for m in df["Medium"])

    def test_subset_partial_word_matching(self):
        """subset should match partial words."""
        data = InactivationData().subset(category="virus")
        df = data.full_df
        assert len(df) > 0
        assert all("virus" in c.lower() for c in df["Category"])

    def test_subset_empty_result_returns_empty_df(self):
        """subset with no matching rows should return empty DataFrame."""
        data = InactivationData()
        data.subset(species="nonexistent_species_xyz")
        # Result should be empty but not raise
        assert len(data.display_df) == 0


class TestInactivationDataDisplayDf:
    """Tests for InactivationData.display_df property."""

    def test_display_df_returns_dataframe(self):
        """display_df should return DataFrame."""
        data = InactivationData()
        df = data.display_df
        assert isinstance(df, pd.DataFrame)

    def test_display_df_has_species(self):
        """display_df should have Species column."""
        data = InactivationData()
        df = data.display_df
        assert "Species" in df.columns

    def test_display_df_filtered(self):
        """display_df should reflect subset filters."""
        data = InactivationData().subset(medium="Aerosol")
        df = data.display_df
        # All rows should be Aerosol medium
        if "Medium" in df.columns and len(df) > 0:
            assert all(df["Medium"] == "Aerosol")


class TestInactivationDataWithFluence:
    """Tests for InactivationData with fluence calculations."""

    def test_fluence_adds_each_column(self):
        """InactivationData with fluence should add eACH-UV column."""
        data = InactivationData(fluence=1.0)
        # The internal _full_df should have computed columns
        assert data._full_df is not None
        assert len(data._full_df) > 0

    def test_fluence_dict_multiple_wavelengths(self):
        """InactivationData with dict fluence should handle all wavelengths."""
        data = InactivationData(fluence={222: 0.5, 254: 0.5})
        # Should not raise error
        df = data.display_df
        assert df is not None


class TestInactivationDataMetadataProperties:
    """Tests for metadata properties (species, mediums, strains, etc.)."""

    def test_species_property(self):
        """species property should return list of unique species."""
        data = InactivationData()
        species = data.species
        assert isinstance(species, list)
        assert len(species) > 0

    def test_mediums_property(self):
        """mediums property should return list of unique mediums."""
        data = InactivationData()
        mediums = data.mediums
        assert isinstance(mediums, list)
        assert len(mediums) > 0

    def test_strains_property(self):
        """strains property should return list of unique strains or None."""
        data = InactivationData()
        strains = data.strains
        # May be None if no strain data, or list if present
        assert strains is None or isinstance(strains, list)

    def test_conditions_property(self):
        """conditions property should return list of unique conditions or None."""
        data = InactivationData()
        conditions = data.conditions
        # May be None if no condition data, or list if present
        assert conditions is None or isinstance(conditions, list)

    def test_metadata_sorted_alphabetically(self):
        """metadata lists should be sorted alphabetically."""
        data = InactivationData()
        species = data.species
        assert species == sorted(species)


class TestAverageValueParametric:
    """Tests for average_value with parametric inputs."""

    def test_average_value_single_function(self):
        """average_value with single function returns float."""
        data = InactivationData(fluence=0.5)
        result = data.average_value("log2", species="coli")
        assert isinstance(result, float)

    def test_average_value_function_list(self):
        """average_value with function list returns dict keyed by function."""
        data = InactivationData(fluence=0.5)
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
        data = InactivationData(fluence=0.5)
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
        data = InactivationData(fluence=0.5)
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


class TestWordsMatch:
    """Tests for words_match function."""

    def test_words_match_exact(self):
        """words_match should match exact strings."""
        assert words_match("Aerosol", "Aerosol")

    def test_words_match_case_insensitive(self):
        """words_match should be case-insensitive."""
        assert words_match("aerosol", "Aerosol")
        assert words_match("AEROSOL", "Aerosol")
        assert words_match("AeRoSoL", "Aerosol")

    def test_words_match_partial_word(self):
        """words_match should match partial words."""
        assert words_match("aero", "Aerosol")
        assert words_match("virus", "Viruses")

    def test_words_match_multiple_words(self):
        """words_match should require all query words to match."""
        assert words_match("ms2 phage", "Phage MS2")
        assert not words_match("ms2 coli", "Phage MS2")

    def test_words_match_alias_air(self):
        """words_match should translate 'air' to 'aerosol'."""
        assert words_match("air", "Aerosol")

    def test_words_match_alias_water(self):
        """words_match should translate 'water' to 'liquid'."""
        assert words_match("water", "Liquid")

    def test_words_match_no_match(self):
        """words_match should return False for non-matching strings."""
        assert not words_match("surface", "Aerosol")
        assert not words_match("xyz", "Aerosol")


class TestFilterByWords:
    """Tests for filter_by_words function."""

    def test_filter_by_words_basic(self):
        """filter_by_words should filter by word match."""
        df = pd.DataFrame({"Medium": ["Aerosol", "Surface", "Liquid"]})
        result = filter_by_words(df, "Medium", "aero")
        assert len(result) == 1
        assert result["Medium"].iloc[0] == "Aerosol"

    def test_filter_by_words_none(self):
        """filter_by_words should return unchanged df if value is None."""
        df = pd.DataFrame({"Medium": ["Aerosol", "Surface"]})
        result = filter_by_words(df, "Medium", None)
        assert len(result) == 2

    def test_filter_by_words_handles_nan(self):
        """filter_by_words should handle NaN values in column."""
        df = pd.DataFrame({"Medium": ["Aerosol", None, "Surface"]})
        result = filter_by_words(df, "Medium", "aero")
        assert len(result) == 1


class TestApplyRowFilters:
    """Tests for apply_row_filters function."""

    def test_apply_row_filters_single(self):
        """apply_row_filters should apply single filter."""
        df = pd.DataFrame({
            "Medium": ["Aerosol", "Surface"],
            "Category": ["Virus", "Bacteria"],
            "Species": ["MS2", "E. coli"],
            "Strain": ["ATCC", "K12"],
            "Condition": ["log", "stationary"],
        })
        result = apply_row_filters(df, medium="aero")
        assert len(result) == 1
        assert result["Medium"].iloc[0] == "Aerosol"

    def test_apply_row_filters_multiple(self):
        """apply_row_filters should apply multiple filters."""
        df = pd.DataFrame({
            "Medium": ["Aerosol", "Aerosol", "Surface"],
            "Category": ["Virus", "Bacteria", "Virus"],
            "Species": ["MS2", "E. coli", "Adeno"],
            "Strain": ["ATCC", "K12", ""],
            "Condition": ["", "", ""],
        })
        result = apply_row_filters(df, medium="aero", category="virus")
        assert len(result) == 1
        assert result["Species"].iloc[0] == "MS2"


class TestPlotFunction:
    """Tests for plot functionality."""

    def test_plot_returns_figure(self):
        """plot() should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        data = InactivationData(fluence=0.5).subset(medium="aerosol", species="ms2")
        fig = data.plot()
        assert fig is not None
        plt.close('all')

    def test_plot_survival_returns_figure(self):
        """plot_survival() should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        data = InactivationData(fluence=0.5).subset(medium="aerosol", species="ms2")
        fig = data.plot_survival()
        assert fig is not None
        plt.close('all')

    def test_plot_with_single_species_shows_strains(self):
        """When single species filtered, x-axis should show strains if multiple."""
        import matplotlib.pyplot as plt
        data = InactivationData(fluence=0.5).subset(medium="aerosol", species="ms2")
        fig = data.plot()
        # Just verify it doesn't crash - detailed axis checking would be brittle
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_returns_figure(self):
        """plot_wavelength() should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        data = InactivationData().subset(medium="aerosol")
        fig = data.plot_wavelength()
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_with_species_filter(self):
        """plot_wavelength() should work with species filter."""
        import matplotlib.pyplot as plt
        data = InactivationData().subset(species="coli")
        fig = data.plot_wavelength()
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_k2(self):
        """plot_wavelength(y='k2') should plot k2 values."""
        import matplotlib.pyplot as plt
        data = InactivationData()
        fig = data.plot_wavelength(y="k2")
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_log_scale(self):
        """plot_wavelength() should support log scale."""
        import matplotlib.pyplot as plt
        data = InactivationData()
        fig = data.plot_wavelength(yscale="log")
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_no_fit(self):
        """plot_wavelength(show_fit=False) should work without fit line."""
        import matplotlib.pyplot as plt
        data = InactivationData()
        fig = data.plot_wavelength(show_fit=False)
        assert fig is not None
        plt.close('all')
