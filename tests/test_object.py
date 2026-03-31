"""Tests for the Object class."""

import pytest
import numpy as np
from guv_calcs import Object, Polygon2D


class TestBoxConstruction:

    def test_default_object(self):
        obj = Object()
        assert obj.id == "Object"
        assert obj.name == "Object"
        assert obj.width == 1
        assert obj.length == 1
        assert obj.height == 1
        assert obj.position == (0.0, 0.0, 0.0)
        assert obj.R == 0.0
        assert obj.T == 0.0
        assert obj.enabled is True

    def test_box_factory(self):
        obj = Object.box(2, 3, 4, object_id="desk")
        assert obj.width == 2
        assert obj.length == 3
        assert obj.height == 4
        assert obj.id == "desk"

    def test_box_has_six_faces(self):
        obj = Object.box(1, 1, 1)
        assert len(obj.face_ids) == 6
        assert "bottom" in obj.face_ids
        assert "top" in obj.face_ids
        assert "wall_0" in obj.face_ids
        assert "wall_3" in obj.face_ids

    def test_box_zero_dimension_rejected(self):
        with pytest.raises(ValueError):
            Object.box(0, 1, 1)
        with pytest.raises(ValueError):
            Object.box(1, -1, 1)

    def test_box_with_optical_properties(self):
        obj = Object.box(1, 1, 1, R=0.5, T=0.3)
        assert obj.R == 0.5
        assert obj.T == 0.3
        for fid in obj.face_ids:
            props = obj.get_face_properties(fid)
            assert props["R"] == 0.5
            assert props["T"] == 0.3


class TestExtrusionConstruction:

    def test_extrusion_from_list(self):
        verts = [(0, 0), (4, 0), (4, 3), (0, 3)]
        obj = Object.extrusion(verts, height=2)
        assert obj.height == 2
        assert len(obj.face_ids) == 6  # bottom + top + 4 walls

    def test_extrusion_from_polygon2d(self):
        poly = Polygon2D(vertices=((0, 0), (3, 0), (3, 2), (0, 2)))
        obj = Object.extrusion(poly, height=1.5)
        assert obj.height == 1.5

    def test_extrusion_l_shape(self):
        verts = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        obj = Object.extrusion(verts, height=1)
        assert len(obj.face_ids) == 8  # bottom + top + 6 walls

    def test_extrusion_zero_height_rejected(self):
        with pytest.raises(ValueError):
            Object.extrusion([(0, 0), (1, 0), (1, 1)], height=0)


class TestDimensionProperties:

    def test_box_dimensions(self):
        obj = Object.box(2, 3, 4)
        assert obj.width == 2
        assert obj.length == 3
        assert obj.height == 4

    def test_extrusion_bounding_box_dimensions(self):
        verts = [(0, 0), (6, 0), (6, 4), (0, 4)]
        obj = Object.extrusion(verts, height=2.5)
        assert abs(obj.width - 6) < 1e-9
        assert abs(obj.length - 4) < 1e-9
        assert obj.height == 2.5

    def test_extrusion_irregular_polygon_bbox(self):
        # L-shaped polygon
        verts = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        obj = Object.extrusion(verts, height=1)
        assert abs(obj.width - 4) < 1e-9
        assert abs(obj.length - 4) < 1e-9


class TestPositioning:

    def test_initial_position(self):
        obj = Object.box(1, 1, 1, position=(2, 3, 4))
        assert obj.x == 2
        assert obj.y == 3
        assert obj.z == 4

    def test_move(self):
        obj = Object()
        obj.move(x=5, y=6, z=7)
        assert obj.position == (5.0, 6.0, 7.0)

    def test_move_partial(self):
        obj = Object(position=(1, 2, 3))
        obj.move(z=10)
        assert obj.position == (1.0, 2.0, 10.0)

    def test_move_returns_self(self):
        obj = Object()
        result = obj.move(x=1)
        assert result is obj

    def test_world_surfaces_update_on_move(self):
        obj = Object.box(1, 1, 1, object_id="box")
        key = "box:bottom"
        origin_before = obj.surfaces[key].plane.geometry.origin
        obj.move(x=5)
        origin_after = obj.surfaces[key].plane.geometry.origin
        assert origin_after[0] != origin_before[0]


class TestRotation:

    def test_initial_rotation(self):
        obj = Object(yaw=45, pitch=10, roll=5)
        assert obj._yaw == 45
        assert obj._pitch == 10
        assert obj._roll == 5

    def test_rotate(self):
        obj = Object()
        obj.rotate(yaw=90)
        assert obj._yaw == 90

    def test_rotate_partial(self):
        obj = Object(yaw=10, pitch=20, roll=30)
        obj.rotate(pitch=0)
        assert obj._yaw == 10  # unchanged
        assert obj._pitch == 0
        assert obj._roll == 30  # unchanged

    def test_rotate_returns_self(self):
        obj = Object()
        result = obj.rotate(yaw=45)
        assert result is obj

    def test_90_degree_yaw_swaps_axes(self):
        obj = Object.box(2, 1, 1, object_id="box")
        obj.rotate(yaw=90)
        # After 90-degree yaw, world surfaces should be rotated
        # Gather all world vertices
        all_verts = []
        for s in obj.surfaces.values():
            all_verts.append(s.plane.geometry.boundary_vertices)
        coords = np.vstack(all_verts)
        # Bounding box in XY should be roughly swapped
        x_span = coords[:, 0].max() - coords[:, 0].min()
        y_span = coords[:, 1].max() - coords[:, 1].min()
        assert abs(x_span - 1) < 0.1  # was width=2, now ~1
        assert abs(y_span - 2) < 0.1  # was length=1, now ~2


class TestFaceProperties:

    def test_get_all_face_properties(self):
        obj = Object.box(1, 1, 1, R=0.3, T=0.2)
        props = obj.get_face_properties()
        assert len(props) == 6
        for fid, p in props.items():
            assert p["R"] == 0.3
            assert p["T"] == 0.2

    def test_get_single_face_properties(self):
        obj = Object.box(1, 1, 1, R=0.5)
        props = obj.get_face_properties("top")
        assert props == {"R": 0.5, "T": 0.0}

    def test_get_unknown_face_raises(self):
        obj = Object()
        with pytest.raises(KeyError):
            obj.get_face_properties("nonexistent")

    def test_set_face_properties_all(self):
        obj = Object()
        obj.set_face_properties(R=0.1, T=0.2)
        assert obj.R == 0.1
        assert obj.T == 0.2
        for fid in obj.face_ids:
            p = obj.get_face_properties(fid)
            assert p["R"] == 0.1
            assert p["T"] == 0.2

    def test_set_face_properties_single(self):
        obj = Object.box(1, 1, 1, R=0.0, T=0.0)
        obj.set_face_properties(R=0.3, T=0.6, face="top")
        assert obj.get_face_properties("top") == {"R": 0.3, "T": 0.6}
        assert obj.get_face_properties("bottom") == {"R": 0.0, "T": 0.0}

    def test_set_face_properties_validates_sum(self):
        obj = Object()
        with pytest.raises(ValueError):
            obj.set_face_properties(R=0.6, T=0.5)

    def test_set_face_properties_validates_range(self):
        obj = Object()
        with pytest.raises(ValueError):
            obj.set_face_properties(R=-0.1, T=0.0)
        with pytest.raises(ValueError):
            obj.set_face_properties(R=0.0, T=1.5)

    def test_set_face_properties_avoids_ordering_issue(self):
        """set_face_properties can change from high-R to high-T atomically."""
        obj = Object.box(1, 1, 1, R=0.8, T=0.0)
        # This would fail with separate set_reflectance/set_transmittance:
        # set_transmittance(0.9) would fail because R=0.8 + T=0.9 > 1
        obj.set_face_properties(R=0.0, T=0.9)
        assert obj.R == 0.0
        assert obj.T == 0.9

    def test_set_reflectance_all(self):
        obj = Object.box(1, 1, 1, R=0.0)
        obj.set_reflectance(0.5)
        assert obj.R == 0.5
        for fid in obj.face_ids:
            assert obj.get_face_properties(fid)["R"] == 0.5

    def test_set_reflectance_single(self):
        obj = Object()
        obj.set_reflectance(0.7, face="bottom")
        assert obj.get_face_properties("bottom")["R"] == 0.7
        assert obj.get_face_properties("top")["R"] == 0.0

    def test_set_transmittance_all(self):
        obj = Object.box(1, 1, 1)
        obj.set_transmittance(0.4)
        assert obj.T == 0.4
        for fid in obj.face_ids:
            assert obj.get_face_properties(fid)["T"] == 0.4

    def test_set_reflectance_unknown_face_raises(self):
        obj = Object()
        with pytest.raises(KeyError):
            obj.set_reflectance(0.5, face="nonexistent")

    def test_set_transmittance_unknown_face_raises(self):
        obj = Object()
        with pytest.raises(KeyError):
            obj.set_transmittance(0.5, face="nonexistent")


class TestSetDimensions:

    def test_set_box_width(self):
        obj = Object.box(1, 2, 3)
        obj.set_dimensions(width=5)
        assert obj.width == 5
        assert obj.length == 2
        assert obj.height == 3

    def test_set_box_all_dimensions(self):
        obj = Object.box(1, 1, 1)
        obj.set_dimensions(width=2, length=3, height=4)
        assert obj.width == 2
        assert obj.length == 3
        assert obj.height == 4

    def test_set_dimensions_preserves_face_properties(self):
        obj = Object.box(1, 1, 1, R=0.1, T=0.2)
        obj.set_face_properties(R=0.8, T=0.1, face="top")
        obj.set_dimensions(width=5)
        # Top should keep its custom properties
        assert obj.get_face_properties("top") == {"R": 0.8, "T": 0.1}
        # Others should keep the default
        assert obj.get_face_properties("bottom") == {"R": 0.1, "T": 0.2}

    def test_set_dimensions_preserves_position(self):
        obj = Object.box(1, 1, 1, position=(5, 6, 7))
        obj.set_dimensions(width=2)
        assert obj.position == (5.0, 6.0, 7.0)

    def test_set_dimensions_preserves_rotation(self):
        obj = Object.box(1, 1, 1, yaw=45)
        obj.set_dimensions(height=10)
        assert obj._yaw == 45

    def test_set_dimensions_returns_self(self):
        obj = Object()
        result = obj.set_dimensions(width=2)
        assert result is obj

    def test_set_dimensions_zero_rejected(self):
        obj = Object.box(1, 1, 1)
        with pytest.raises(ValueError):
            obj.set_dimensions(width=0)
        with pytest.raises(ValueError):
            obj.set_dimensions(height=-1)

    def test_set_extrusion_height(self):
        verts = [(0, 0), (4, 0), (4, 3), (0, 3)]
        obj = Object.extrusion(verts, height=2)
        obj.set_dimensions(height=5)
        assert obj.height == 5

    def test_set_extrusion_width_scales_polygon(self):
        verts = [(0, 0), (4, 0), (4, 3), (0, 3)]
        obj = Object.extrusion(verts, height=2)
        original_width = obj.width
        obj.set_dimensions(width=original_width * 2)
        assert abs(obj.width - original_width * 2) < 1e-9


class TestGridResolution:

    def test_set_num_points_global(self):
        obj = Object.box(2, 2, 2, num_points=5)
        assert obj._num_points == 5
        obj.set_num_points(10)
        assert obj._num_points == 10

    def test_set_num_points_preserves_face_properties(self):
        obj = Object.box(1, 1, 1, R=0.1, T=0.2)
        obj.set_face_properties(R=0.8, T=0.1, face="top")
        obj.set_num_points(10)
        assert obj.get_face_properties("top") == {"R": 0.8, "T": 0.1}
        assert obj.get_face_properties("bottom") == {"R": 0.1, "T": 0.2}

    def test_set_num_points_single_face(self):
        obj = Object.box(2, 2, 2, num_points=5)
        obj.set_num_points(20, face="top")
        top_plane = obj._local_surfaces["top"].plane
        assert top_plane.num_x == 20

    def test_set_num_points_unknown_face_raises(self):
        obj = Object()
        with pytest.raises(KeyError):
            obj.set_num_points(10, face="nonexistent")

    def test_set_spacing_all(self):
        obj = Object.box(4, 4, 2, num_points=5)
        obj.set_spacing(x_spacing=0.5, y_spacing=0.5)
        for s in obj._local_surfaces.values():
            # Spacing should be approximately 0.5
            assert abs(s.plane.x_spacing - 0.5) < 0.1 or abs(s.plane.y_spacing - 0.5) < 0.1

    def test_set_spacing_single_face(self):
        obj = Object.box(4, 4, 2, num_points=5)
        obj.set_spacing(x_spacing=0.25, face="floor" if "floor" in obj.face_ids else "bottom")
        bottom = obj._local_surfaces["bottom"]
        assert abs(bottom.plane.x_spacing - 0.25) < 0.05

    def test_set_spacing_unknown_face_raises(self):
        obj = Object()
        with pytest.raises(KeyError):
            obj.set_spacing(x_spacing=0.5, face="nonexistent")


class TestCopy:

    def test_copy_creates_equal_object(self):
        obj = Object.box(2, 3, 4, R=0.5, T=0.2, position=(1, 2, 3))
        clone = obj.copy()
        assert clone == obj
        assert clone is not obj

    def test_copy_with_overrides(self):
        obj = Object.box(1, 1, 1, object_id="original")
        clone = obj.copy(object_id="clone", R=0.9)
        assert clone.id == "clone"
        assert clone.R == 0.9
        assert clone.width == 1  # inherited

    def test_copy_preserves_face_properties(self):
        obj = Object.box(1, 1, 1)
        obj.set_face_properties(R=0.8, T=0.1, face="top")
        clone = obj.copy()
        assert clone.get_face_properties("top") == {"R": 0.8, "T": 0.1}

    def test_copy_is_independent(self):
        obj = Object.box(1, 1, 1, R=0.0)
        clone = obj.copy()
        clone.set_reflectance(0.9)
        assert obj.R == 0.0  # original unchanged


class TestSerialization:

    def test_to_dict_round_trip(self):
        obj = Object.box(2, 3, 4, object_id="test", R=0.3, T=0.1,
                         position=(1, 2, 3), yaw=45, pitch=10, roll=5)
        data = obj.to_dict()
        restored = Object.from_dict(data)
        assert restored == obj

    def test_to_dict_with_face_overrides(self):
        obj = Object.box(1, 1, 1, R=0.0, T=0.0)
        obj.set_face_properties(R=0.5, T=0.3, face="top")
        data = obj.to_dict()
        assert data["face_properties"]["top"] == {"R": 0.5, "T": 0.3}
        # faces matching the default should not appear
        assert "bottom" not in data["face_properties"]

    def test_from_dict_restores_face_overrides(self):
        obj = Object.box(1, 1, 1, R=0.0, T=0.0)
        obj.set_face_properties(R=0.5, T=0.3, face="top")
        data = obj.to_dict()
        restored = Object.from_dict(data)
        assert restored.get_face_properties("top") == {"R": 0.5, "T": 0.3}
        assert restored.get_face_properties("bottom") == {"R": 0.0, "T": 0.0}

    def test_extrusion_round_trip(self):
        verts = [(0, 0), (4, 0), (4, 3), (0, 3)]
        obj = Object.extrusion(verts, height=2, object_id="wall")
        data = obj.to_dict()
        restored = Object.from_dict(data)
        assert restored == obj

    def test_to_dict_keys(self):
        data = Object().to_dict()
        expected_keys = {
            "object_id", "name", "position", "yaw", "pitch", "roll",
            "R", "T", "enabled", "num_points", "shape", "face_properties",
        }
        assert set(data.keys()) == expected_keys


class TestCalcState:

    def test_calc_state_changes_on_move(self):
        obj = Object()
        state1 = obj.calc_state
        obj.move(x=5)
        state2 = obj.calc_state
        assert state1 != state2

    def test_calc_state_changes_on_rotate(self):
        obj = Object()
        state1 = obj.calc_state
        obj.rotate(yaw=45)
        state2 = obj.calc_state
        assert state1 != state2

    def test_calc_state_changes_on_reflectance(self):
        obj = Object()
        state1 = obj.calc_state
        obj.set_reflectance(0.5)
        state2 = obj.calc_state
        assert state1 != state2

    def test_calc_state_changes_on_disable(self):
        obj = Object()
        state1 = obj.calc_state
        obj.enabled = False
        state2 = obj.calc_state
        assert state1 != state2

    def test_update_state_is_empty(self):
        obj = Object()
        assert obj.update_state == ()


class TestUnitConversion:

    def test_convert_meters_to_feet(self):
        obj = Object.box(1, 2, 3, position=(1, 0, 0))
        obj.convert_units("meters", "feet")
        # 1 meter ~ 3.2808 feet
        assert abs(obj.width - 3.2808) < 0.01
        assert abs(obj.length - 2 * 3.2808) < 0.01
        assert abs(obj.height - 3 * 3.2808) < 0.01
        assert abs(obj.x - 3.2808) < 0.01

    def test_convert_preserves_face_properties(self):
        obj = Object.box(1, 1, 1, R=0.3, T=0.1)
        obj.set_face_properties(R=0.8, T=0.1, face="top")
        obj.convert_units("meters", "feet")
        assert obj.get_face_properties("top") == {"R": 0.8, "T": 0.1}
        assert obj.get_face_properties("bottom") == {"R": 0.3, "T": 0.1}


class TestIdentity:

    def test_default_id(self):
        obj = Object()
        assert obj.id == "Object"

    def test_custom_id(self):
        obj = Object(object_id="desk-1")
        assert obj.id == "desk-1"

    def test_name_defaults_to_id(self):
        obj = Object(object_id="partition")
        assert obj.name == "partition"

    def test_name_independent_of_id(self):
        obj = Object(object_id="obj-1", name="Main Partition")
        assert obj.name == "Main Partition"
        assert obj.id == "obj-1"

    def test_assign_id_updates_world_surface_keys(self):
        obj = Object(object_id="old")
        assert all(k.startswith("old:") for k in obj.surfaces.keys())
        obj._assign_id("new")
        assert all(k.startswith("new:") for k in obj.surfaces.keys())

    def test_assign_id_updates_name_if_matching(self):
        obj = Object(object_id="old")
        assert obj.name == "old"
        obj._assign_id("new")
        assert obj.name == "new"

    def test_assign_id_preserves_custom_name(self):
        obj = Object(object_id="old", name="Custom")
        obj._assign_id("new")
        assert obj.name == "Custom"


class TestWorldSurfaces:

    def test_world_surface_keys_namespaced(self):
        obj = Object(object_id="table")
        for key in obj.surfaces.keys():
            assert key.startswith("table:")

    def test_world_surfaces_shifted_by_position(self):
        obj = Object.box(1, 1, 1, object_id="box", position=(10, 0, 0))
        for s in obj.surfaces.values():
            verts = s.plane.geometry.boundary_vertices
            assert verts[:, 0].min() >= 9.4  # shifted by x=10 minus half-width

    def test_surface_count_matches_face_ids(self):
        obj = Object.box(1, 1, 1)
        assert len(obj.surfaces) == len(obj.face_ids)


class TestRepr:

    def test_box_repr(self):
        obj = Object.box(2, 3, 4, object_id="desk")
        r = repr(obj)
        assert "desk" in r
        assert "box(2" in r

    def test_extrusion_repr(self):
        obj = Object.extrusion([(0, 0), (1, 0), (1, 1)], height=2, object_id="tri")
        r = repr(obj)
        assert "tri" in r
        assert "extrusion" in r

    def test_equality(self):
        a = Object.box(1, 2, 3, R=0.5)
        b = Object.box(1, 2, 3, R=0.5)
        assert a == b

    def test_inequality_different_dimensions(self):
        a = Object.box(1, 2, 3)
        b = Object.box(1, 2, 4)
        assert a != b
