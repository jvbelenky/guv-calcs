"""Tests for the efficacy module (InactivationData class, disinfection calculations)."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests

import pytest
import numpy as np
import pandas as pd
from guv_calcs import InactivationData
from guv_calcs.efficacy._kinetics import (
    parse_resistant,
    extract_kinetic_params,
    compute_row,
)
from guv_calcs.efficacy._filtering import (
    filter_by_column,
    words_match,
    filter_by_words,
    apply_row_filters,
    get_effective_wavelengths,
)
from guv_calcs.efficacy.math import survival_fraction


# Module-scoped fixtures to avoid rebuilding InactivationData per test
@pytest.fixture(scope="module")
def inact_data():
    """InactivationData with no fluence (base table only)."""
    return InactivationData()

@pytest.fixture(scope="module")
def inact_data_fluence():
    """InactivationData with fluence=0.5 (computed columns)."""
    return InactivationData(fluence=0.5)

@pytest.fixture(scope="module")
def inact_data_fluence_dict():
    """InactivationData with dict fluence."""
    return InactivationData(fluence={222: 0.5, 254: 0.5})


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
        df1["test_col"] = 1
        assert "test_col" not in df2.columns

    def test_get_valid_categories(self):
        categories = InactivationData.get_valid_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(c, str) for c in categories)

    def test_get_valid_mediums(self):
        mediums = InactivationData.get_valid_mediums()
        assert isinstance(mediums, list)
        assert len(mediums) > 0
        assert any(m in mediums for m in ["Aerosol", "Surface", "Liquid"])

    def test_get_valid_wavelengths(self):
        wavelengths = InactivationData.get_valid_wavelengths()
        assert isinstance(wavelengths, list)
        assert len(wavelengths) > 0
        assert all(isinstance(w, (int, float)) for w in wavelengths)


class TestInactivationDataInitialization:
    """Tests for InactivationData initialization."""

    def test_init_no_args(self, inact_data):
        assert inact_data is not None

    def test_init_with_fluence_float(self, inact_data_fluence):
        assert inact_data_fluence is not None

    def test_init_with_fluence_dict(self, inact_data_fluence_dict):
        assert inact_data_fluence_dict is not None

    def test_init_with_volume(self):
        """InactivationData with volume should compute CADR columns."""
        data = InactivationData(fluence=1.0, volume_m3=50.0)
        assert data is not None


class TestInactivationDataSubset:
    """Tests for InactivationData.subset() filtering."""

    def test_subset_returns_new_instance(self, inact_data):
        result = inact_data.subset(medium="Aerosol")
        assert result is not inact_data
        assert result._medium == "Aerosol"
        assert inact_data._medium is None

    def test_subset_by_medium(self, inact_data):
        result = inact_data.subset(medium="Aerosol")
        assert result._medium is not None

    def test_subset_by_category(self, inact_data):
        categories = InactivationData.get_valid_categories()
        if categories:
            result = inact_data.subset(category=categories[0])
            assert result._category is not None

    def test_subset_by_species(self, inact_data):
        result = inact_data.subset(species="coli")
        assert result._species is not None
        df = result.full_df
        assert all("coli" in s.lower() for s in df["Species"])

    def test_subset_by_strain(self, inact_data):
        result = inact_data.subset(strain="ATCC")
        assert result._strain is not None

    def test_subset_by_condition(self, inact_data):
        result = inact_data.subset(condition="stationary")
        assert result._condition is not None

    def test_subset_by_log_level(self, inact_data_fluence):
        result = inact_data_fluence.subset(log=3)
        assert result._log == 3

    def test_subset_chaining(self, inact_data):
        result = inact_data.subset(medium="Aerosol").subset(log=2)
        assert result is not inact_data
        assert result._medium == "Aerosol"
        assert result._log == 2
        assert inact_data._medium is None

    def test_subset_case_insensitive_medium(self, inact_data):
        data = inact_data.subset(medium="aerosol")
        df = data.full_df
        assert len(df) > 0
        assert all("Aerosol" in m for m in df["Medium"])

    def test_subset_alias_air_to_aerosol(self, inact_data):
        data = inact_data.subset(medium="air")
        df = data.full_df
        assert len(df) > 0
        assert all("Aerosol" in m for m in df["Medium"])

    def test_subset_partial_word_matching(self, inact_data):
        data = inact_data.subset(category="virus")
        df = data.full_df
        assert len(df) > 0
        assert all("virus" in c.lower() for c in df["Category"])

    def test_subset_category_list_with_aliases(self, inact_data):
        data = inact_data.subset(category=["bacteria", "virus"])
        df = data.full_df
        assert len(df) > 0
        assert set(df["Category"].unique()) == {"Bacteria", "Viruses"}

    def test_subset_medium_list_with_aliases(self, inact_data):
        data = inact_data.subset(medium=["air", "surface"])
        df = data.full_df
        assert len(df) > 0
        assert all(m in ("Aerosol", "Surface") for m in df["Medium"].unique())

    def test_subset_empty_result_returns_empty_df(self, inact_data):
        data = inact_data.subset(species="nonexistent_species_xyz")
        assert len(data.display_df) == 0


class TestInactivationDataDisplayDf:
    """Tests for InactivationData.display_df property."""

    def test_display_df_returns_dataframe(self, inact_data):
        df = inact_data.display_df
        assert isinstance(df, pd.DataFrame)

    def test_display_df_has_species(self, inact_data):
        df = inact_data.display_df
        assert "Species" in df.columns

    def test_display_df_filtered(self, inact_data):
        data = inact_data.subset(medium="Aerosol")
        df = data.display_df
        if "Medium" in df.columns and len(df) > 0:
            assert all(df["Medium"] == "Aerosol")


class TestInactivationDataWithFluence:
    """Tests for InactivationData with fluence calculations."""

    def test_fluence_adds_each_column(self, inact_data_fluence):
        assert inact_data_fluence._full_df is not None
        assert len(inact_data_fluence._full_df) > 0

    def test_fluence_dict_multiple_wavelengths(self, inact_data_fluence_dict):
        df = inact_data_fluence_dict.display_df
        assert df is not None


class TestInactivationDataMetadataProperties:
    """Tests for metadata properties (species, mediums, strains, etc.)."""

    def test_species_property(self, inact_data):
        species = inact_data.species
        assert isinstance(species, list)
        assert len(species) > 0

    def test_mediums_property(self, inact_data):
        mediums = inact_data.mediums
        assert isinstance(mediums, list)
        assert len(mediums) > 0

    def test_strains_property(self, inact_data):
        strains = inact_data.strains
        assert strains is None or isinstance(strains, list)

    def test_conditions_property(self, inact_data):
        conditions = inact_data.conditions
        assert conditions is None or isinstance(conditions, list)

    def test_metadata_sorted_alphabetically(self, inact_data):
        species = inact_data.species
        assert species == sorted(species)


class TestAverageValueParametric:
    """Tests for average_value with parametric inputs."""

    def test_average_value_single_function(self, inact_data_fluence):
        result = inact_data_fluence.average_value("log2", species="coli")
        assert isinstance(result, float)

    def test_average_value_function_list(self, inact_data_fluence):
        result = inact_data_fluence.average_value(["log2", "log3"], species="coli")
        assert isinstance(result, dict)
        assert "log2" in result
        assert "log3" in result
        assert isinstance(result["log2"], float)
        assert isinstance(result["log3"], float)
        assert result["log3"] > result["log2"]

    def test_average_value_function_list_with_species_list(self, inact_data_fluence):
        result = inact_data_fluence.average_value(["log2", "each"], species=["coli", "staph"])
        assert isinstance(result, dict)
        assert "log2" in result
        assert "each" in result
        assert isinstance(result["log2"], dict)
        assert "coli" in result["log2"]
        assert "staph" in result["log2"]

    def test_average_value_function_list_order(self, inact_data_fluence):
        result = inact_data_fluence.average_value(["log2", "log3"], species=["coli"])
        assert "log2" in result
        assert "coli" in result["log2"]

    def test_average_value_medium_alias(self, inact_data_fluence):
        result = inact_data_fluence.average_value("each", medium="air")
        assert isinstance(result, float)
        assert result > 0


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

    def test_parse_resistant_empty_string(self):
        """parse_resistant should return 0.0 for empty strings."""
        assert parse_resistant("") == 0.0
        assert parse_resistant("  ") == 0.0

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

    def test_extract_kinetic_params_float_resistant(self):
        """extract_kinetic_params should handle float resistant values."""
        row = pd.Series({
            "k1 [cm2/mJ]": 0.1,
            "k2 [cm2/mJ]": 0.01,
            "% resistant": 0.01
        })
        params = extract_kinetic_params(row)
        assert params["k1"] == 0.1
        assert params["k2"] == 0.01
        assert params["f"] == 0.01

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


class TestSurvivalFraction:
    """Tests for survival_fraction function."""

    def test_survival_fraction_at_t0(self):
        """survival_fraction at t=0 should be 1.0."""
        assert survival_fraction(0, irrad=100, k1=0.1) == pytest.approx(1.0)

    def test_survival_fraction_decreases(self):
        """survival_fraction should decrease over time."""
        s1 = survival_fraction(10, irrad=100, k1=0.1)
        s2 = survival_fraction(100, irrad=100, k1=0.1)
        assert s1 < 1.0
        assert s2 < s1

    def test_survival_fraction_vectorized(self):
        """survival_fraction should accept arrays."""
        t = np.array([0, 10, 100])
        result = survival_fraction(t, irrad=100, k1=0.1)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] < 1.0
        assert result[2] < result[1]

    def test_survival_fraction_multi_wavelength(self):
        """survival_fraction should handle multi-wavelength inputs."""
        result = survival_fraction(10, irrad=[50, 50], k1=[0.1, 0.05], k2=[0.01, 0.005], f=[0.01, 0.02])
        assert 0 < float(result) < 1


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

    def test_filter_by_words_list(self):
        """filter_by_words with list should match ANY element."""
        df = pd.DataFrame({"Medium": ["Aerosol", "Surface", "Liquid"]})
        result = filter_by_words(df, "Medium", ["air", "surface"])
        assert len(result) == 2
        assert set(result["Medium"]) == {"Aerosol", "Surface"}

    def test_filter_by_words_list_with_aliases(self):
        """filter_by_words with list should resolve aliases per element."""
        df = pd.DataFrame({"Category": ["Bacteria", "Viruses", "Bacterial spores"]})
        result = filter_by_words(df, "Category", ["bacteria", "virus"])
        assert len(result) == 2
        assert set(result["Category"]) == {"Bacteria", "Viruses"}


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
            "Category": ["Viruses", "Bacteria", "Viruses"],
            "Species": ["MS2", "E. coli", "Adeno"],
            "Strain": ["ATCC", "K12", ""],
            "Condition": ["", "", ""],
        })
        result = apply_row_filters(df, medium="aero", category="virus")
        assert len(result) == 1
        assert result["Species"].iloc[0] == "MS2"


class TestPlotFunction:
    """Tests for plot functionality."""

    def test_plot_returns_figure(self, inact_data_fluence):
        import matplotlib.pyplot as plt
        data = inact_data_fluence.subset(medium="aerosol", species="ms2")
        fig = data.plot()
        assert fig is not None
        plt.close('all')

    def test_plot_survival_returns_figure(self, inact_data_fluence):
        import matplotlib.pyplot as plt
        data = inact_data_fluence.subset(medium="aerosol", species="ms2")
        fig = data.plot_survival()
        assert fig is not None
        plt.close('all')

    def test_plot_with_single_species_shows_strains(self, inact_data_fluence):
        import matplotlib.pyplot as plt
        data = inact_data_fluence.subset(medium="aerosol", species="ms2")
        fig = data.plot()
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_returns_figure(self, inact_data):
        import matplotlib.pyplot as plt
        data = inact_data.subset(medium="aerosol")
        fig = data.plot_wavelength()
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_with_species_filter(self, inact_data):
        import matplotlib.pyplot as plt
        data = inact_data.subset(species="coli")
        fig = data.plot_wavelength()
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_k2(self, inact_data):
        import matplotlib.pyplot as plt
        fig = inact_data.plot_wavelength(y="k2")
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_log_scale(self, inact_data):
        import matplotlib.pyplot as plt
        fig = inact_data.plot_wavelength(yscale="log")
        assert fig is not None
        plt.close('all')

    def test_plot_wavelength_no_fit(self, inact_data):
        import matplotlib.pyplot as plt
        fig = inact_data.plot_wavelength(show_fit=False)
        assert fig is not None
        plt.close('all')


class TestTableFiltering:
    """Tests for table() with call-time filter arguments."""

    def test_table_with_species_filter(self, inact_data_fluence):
        df = inact_data_fluence.table(species="coronavirus")
        assert len(df) > 0
        assert all("coronavirus" in s.lower() for s in df["Species"])

    def test_table_with_category_filter(self, inact_data_fluence):
        all_rows = len(inact_data_fluence.table())
        filtered = inact_data_fluence.table(category="virus")
        assert len(filtered) > 0
        assert len(filtered) < all_rows

    def test_table_no_args_matches_display_df(self, inact_data_fluence):
        data = inact_data_fluence.subset(medium="aerosol")
        table_df = data.table()
        display_df = data.display_df
        pd.testing.assert_frame_equal(table_df, display_df)
