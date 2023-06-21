from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from typing import Any
import numpy as np
import nibabel as nib
import pytest
from simnibs.mesh_tools.mesh_io import Elements, Msh, Nodes

from simnibs.simulation.tms_coil.tms_coil import TmsCoil
from simnibs.simulation.tms_coil.tms_coil_deformation import TmsCoilRotation
from simnibs.simulation.tms_coil.tms_coil_element import (
    DipoleElements,
    LineSegmentElements,
    PositionalTmsCoilElements,
    SampledGridPointElements,
)
from simnibs.simulation.tms_coil.tms_coil_model import TmsCoilModel
from simnibs.simulation.tms_coil.tms_stimulator import TmsStimulator

from .... import SIMNIBSDIR


@pytest.fixture(scope="module")
def testcoil_ccd():
    fn = os.path.join(
        SIMNIBSDIR, "_internal_resources", "testing_files", "testcoil.ccd"
    )
    return fn


@pytest.fixture(scope="module")
def testcoil_nii_gz():
    fn = os.path.join(
        SIMNIBSDIR, "_internal_resources", "testing_files", "testcoil.nii.gz"
    )
    return fn


def is_close_subset_of(array1, array2, tolerance=1e-4):
    return np.all(
        [
            np.any(np.all(np.isclose(array2, element, atol=tolerance), axis=1))
            for element in array1
        ]
    )


def is_close_to_any(array1, array2, tolerance=1e-4):
    return np.any(
        [
            np.any(np.all(np.isclose(array2, element, atol=tolerance), axis=1))
            for element in array1
        ]
    )


class TestReadCoil:
    def test_read_minimal_tcd(
        self, minimal_tcd_coil_dict: dict[str, Any], tmp_path: Path
    ):
        with open(tmp_path / "test.tcd", "w") as f:
            json.dump(minimal_tcd_coil_dict, f)
        coil = TmsCoil.from_tcd(str(tmp_path / "test.tcd"))
        assert coil.name is None
        assert coil.limits is None
        assert coil.casing is None
        assert len(coil.elements) == 1
        assert isinstance(coil.elements[0], DipoleElements)
        np.testing.assert_array_almost_equal(coil.elements[0].points, [[1, 2, 3]])
        assert len(coil.elements[0].deformations) == 0
        assert coil.elements[0].casing is None

    def test_read_minimal_tcd_no_ending(
        self, minimal_tcd_coil_dict: dict[str, Any], tmp_path: Path
    ):
        with open(tmp_path / "test", "w") as f:
            json.dump(minimal_tcd_coil_dict, f)
        coil = TmsCoil.from_tcd(str(tmp_path / "test"))
        assert coil.name is None
        assert coil.limits is None
        assert coil.casing is None
        assert len(coil.elements) == 1
        assert isinstance(coil.elements[0], DipoleElements)
        np.testing.assert_array_almost_equal(coil.elements[0].points, [[1, 2, 3]])
        assert len(coil.elements[0].deformations) == 0
        assert coil.elements[0].casing is None

    def test_read_simplest_ccd(self, tmp_path: Path):
        with open(tmp_path / "test.ccd", "w") as f:
            f.write("# number of elements\n")
            f.write("1\n")
            f.write("1e-003 3e-003 -4e-003 0e+000 0e+000 -4e-006\n")
        coil = TmsCoil.from_ccd(str(tmp_path / "test.ccd"))
        dipole_elements: DipoleElements = coil.elements[0]
        assert np.allclose(
            dipole_elements.points, np.array([[1e-3, 3e-3, -4e-3]]) * 1e3
        )
        assert np.allclose(
            dipole_elements.values, np.array([[0, 0, -4e-6]], dtype=float)
        )

    def test_read_simple_ccd(self, tmp_path: Path):
        with open(tmp_path / "test.ccd", "w") as f:
            f.write("# number of elements\n")
            f.write("3\n")
            f.write("1e-003 3e-003 -4e-003 0e+000 0e+000 -4e-006\n")
            f.write("3e-003 1e-003  1e-003 1e+000 0e+000 0e-000\n")
            f.write("4e-003 2e-003 -5e-003 0e+000 1e+000 0e-000\n")
        coil = TmsCoil.from_ccd(str(tmp_path / "test.ccd"))
        dipole_elements: DipoleElements = coil.elements[0]
        assert np.allclose(
            dipole_elements.points,
            np.array([[1e-3, 3e-3, -4e-3], [3e-3, 1e-3, 1e-3], [4e-3, 2e-3, -5e-3]])
            * 1e3,
        )
        assert np.allclose(
            dipole_elements.values,
            np.array([[0, 0, -4e-6], [1, 0, 0], [0, 1, 0]], dtype=float),
        )

    def test_read_ccd(self, testcoil_ccd: str):
        coil = TmsCoil.from_ccd(testcoil_ccd)
        dipole_elements: DipoleElements = coil.elements[0]
        assert coil.limits is not None
        assert coil.resolution is not None
        assert np.allclose(coil.limits, ((-100, 100), (-100, 100), (-100, 100)))
        assert np.allclose(coil.resolution, (10, 10, 10))
        assert coil.name == "Test coil"
        assert dipole_elements.stimulator is not None
        assert dipole_elements.stimulator.max_di_dt == 100.0
        assert coil.brand == "None"
        assert dipole_elements.stimulator.name == "None"
        assert np.allclose(
            dipole_elements.points,
            np.array(
                [
                    [-1e-2, 0, -1e-3],
                    [1e-2, 0, -1e-3],
                    [-2e-2, 2e-2, -2e-3],
                    [2e-2, -2e-2, -2e-3],
                ]
            )
            * 1e3,
        )
        assert np.allclose(
            dipole_elements.values,
            ((1e-6, 0, 2e-6), (-1e-6, 0, -2e-6), (0, 1e-6, 0), (0, -1e-6, 0)),
        )

    def test_read_ccd_no_ending(self, testcoil_ccd: str, tmp_path: Path):
        source_path = Path(testcoil_ccd)
        destination_path = tmp_path / source_path.stem
        shutil.copy2(testcoil_ccd, destination_path)
        coil = TmsCoil.from_file(str(destination_path))

        dipole_elements: DipoleElements = coil.elements[0]
        assert coil.limits is not None
        assert coil.resolution is not None
        assert np.allclose(coil.limits, ((-100, 100), (-100, 100), (-100, 100)))
        assert np.allclose(coil.resolution, (10, 10, 10))
        assert coil.name == "Test coil"
        assert dipole_elements.stimulator is not None
        assert dipole_elements.stimulator.max_di_dt == 100.0
        assert coil.brand == "None"
        assert dipole_elements.stimulator.name == "None"
        assert np.allclose(
            dipole_elements.points,
            np.array(
                [
                    [-1e-2, 0, -1e-3],
                    [1e-2, 0, -1e-3],
                    [-2e-2, 2e-2, -2e-3],
                    [2e-2, -2e-2, -2e-3],
                ]
            )
            * 1e3,
        )
        assert np.allclose(
            dipole_elements.values,
            ((1e-6, 0, 2e-6), (-1e-6, 0, -2e-6), (0, 1e-6, 0), (0, -1e-6, 0)),
        )

    def test_read_nii_gz(self, testcoil_nii_gz: str):
        coil = TmsCoil.from_nifti(testcoil_nii_gz)
        sampled_grid_points: SampledGridPointElements = coil.elements[0]
        np.testing.assert_allclose(coil.resolution, [10, 10, 10])
        np.testing.assert_allclose(
            coil.limits, [[-100.0, 110.0], [-100.0, 110.0], [-100.0, 110.0]]
        )
        np.testing.assert_allclose(
            sampled_grid_points.affine,
            [
                [10.0, 0.0, 0.0, -100.0],
                [0.0, 10.0, 0.0, -100.0],
                [0.0, 0.0, 10.0, -100.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        np.testing.assert_allclose(sampled_grid_points.data.shape, (21, 21, 21, 3))

    def test_read_nii_gz_no_ending(self, testcoil_nii_gz: str, tmp_path: Path):
        source_path = Path(testcoil_nii_gz)
        destination_path = tmp_path / source_path.stem.rsplit(".", 1)[0]
        shutil.copy2(testcoil_nii_gz, destination_path)
        coil = TmsCoil.from_file(str(destination_path))

        sampled_grid_points: SampledGridPointElements = coil.elements[0]
        np.testing.assert_allclose(coil.resolution, [10, 10, 10])
        np.testing.assert_allclose(
            coil.limits, [[-100.0, 110.0], [-100.0, 110.0], [-100.0, 110.0]]
        )
        np.testing.assert_allclose(
            sampled_grid_points.affine,
            [
                [10.0, 0.0, 0.0, -100.0],
                [0.0, 10.0, 0.0, -100.0],
                [0.0, 0.0, 10.0, -100.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        np.testing.assert_allclose(sampled_grid_points.data.shape, (21, 21, 21, 3))

    def test_read_wrong_format(self, sphere3_msh: Msh, tmp_path: Path):
        sphere3_msh.write(str(tmp_path / "not_a_coil"))
        with pytest.raises(IOError):
            TmsCoil.from_file(str(tmp_path / "not_a_coil"))


class TestWriteCoil:
    def test_write_minimal_tcd(
        self,
        minimal_tcd_coil: TmsCoil,
        minimal_tcd_coil_dict: dict[str, Any],
        tmp_path: Path,
    ):
        minimal_tcd_coil.write(str(tmp_path / "test.tcd"))

        with open(tmp_path / "test.tcd", "r") as fid:
            coil = json.loads(fid.read())

        assert str(coil) == str(minimal_tcd_coil_dict)

    def test_write_medium_tcd(
        self,
        medium_tcd_coil: TmsCoil,
        medium_tcd_coil_dict: dict[str, Any],
        tmp_path: Path,
    ):
        medium_tcd_coil.write(str(tmp_path / "test.tcd"))

        with open(tmp_path / "test.tcd", "r") as fid:
            coil = json.loads(fid.read())

        assert str(coil) == str(medium_tcd_coil_dict)

    def test_write_full_tcd(
        self, full_tcd_coil: TmsCoil, full_tcd_coil_dict: dict[str, Any], tmp_path: Path
    ):
        full_tcd_coil.write(str(tmp_path / "test.tcd"))

        with open(tmp_path / "test.tcd", "r") as fid:
            coil = json.loads(fid.read())

        assert str(coil) == str(full_tcd_coil_dict)

    def test_write_nifti(self, tmp_path: Path):
        affine = np.array(
            [
                [5.0, 0.0, 0.0, -300],
                [0.0, 5.0, 0.0, -200],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 1],
            ]
        )

        field = np.ones((31, 21, 11, 3))
        field[..., 1] = 2
        field[..., 2] = 3

        image = nib.Nifti1Image(field, affine)
        nib.save(image, tmp_path / "test.nii")

        coil = TmsCoil.from_nifti(str(tmp_path / "test.nii"))
        coil.write(str(tmp_path / "test.tcd"))

        coil = TmsCoil.from_tcd(str(tmp_path / "test.tcd"))

        sampled_grid: SampledGridPointElements = coil.elements[0]

        np.testing.assert_allclose(sampled_grid.data, field)
        np.testing.assert_allclose(sampled_grid.affine, affine)


class TestWriteReadCoil:
    def test_multiple_exclusive_stimulators_nifti(
        self, full_tcd_coil: TmsCoil, tmp_path: Path
    ):
        coil = deepcopy(full_tcd_coil)
        coil.resolution = np.array([1, 1, 1])
        coil.limits = np.array([[0, 25], [0, 25], [0, 25]])
        coil.elements[0].stimulator = TmsStimulator("Stim 1", None, None, None)
        coil.elements[1].stimulator = TmsStimulator("Stim 2", None, None, None)
        coil.elements[2].stimulator = TmsStimulator("Stim 3", None, None, None)

        coil_sampled = coil.as_sampled(resample_sampled_elements=True)

        coil.write_nifti(str(tmp_path / "test.nii.gz"))
        coil_nifti = TmsCoil.from_nifti(str(tmp_path / "test.nii.gz"))

        assert len(coil_sampled.elements) == len(coil_nifti.elements)

        for sampled_element, nifti_element in zip(
            coil_sampled.elements, coil_nifti.elements
        ):
            assert isinstance(sampled_element, SampledGridPointElements)
            assert isinstance(nifti_element, SampledGridPointElements)
            np.testing.assert_allclose(
                sampled_element.get_a_field(
                    coil_sampled.get_sample_positions()[0], np.eye(4)
                ),
                nifti_element.get_a_field(
                    coil_sampled.get_sample_positions()[0], np.eye(4)
                ),
            )

        np.testing.assert_allclose(
            coil_sampled.get_a_field(coil_sampled.get_sample_positions()[0], np.eye(4)),
            coil_nifti.get_a_field(coil_nifti.get_sample_positions()[0], np.eye(4)),
        )

    def test_single_stimulators_nifti(self, full_tcd_coil: TmsCoil, tmp_path: Path):
        coil = deepcopy(full_tcd_coil)
        coil.resolution = np.array([1, 1, 1])
        coil.limits = np.array([[0, 25], [0, 25], [0, 25]])
        stimulator = TmsStimulator("Stim 1", None, None, None)
        coil.elements[0].stimulator = stimulator
        coil.elements[1].stimulator = stimulator
        coil.elements[2].stimulator = stimulator

        coil_sampled = coil.as_sampled_squashed()

        coil.write_nifti(str(tmp_path / "test.nii.gz"))
        coil_nifti = TmsCoil.from_nifti(str(tmp_path / "test.nii.gz"))

        assert len(coil_sampled.elements) == len(coil_nifti.elements)

        for sampled_element, nifti_element in zip(
            coil_sampled.elements, coil_nifti.elements
        ):
            assert isinstance(sampled_element, SampledGridPointElements)
            assert isinstance(nifti_element, SampledGridPointElements)
            np.testing.assert_allclose(
                sampled_element.get_a_field(
                    coil_sampled.get_sample_positions()[0], np.eye(4)
                ),
                nifti_element.get_a_field(
                    coil_sampled.get_sample_positions()[0], np.eye(4)
                ),
            )

        np.testing.assert_allclose(
            coil_sampled.get_a_field(coil_sampled.get_sample_positions()[0], np.eye(4)),
            coil_nifti.get_a_field(coil_nifti.get_sample_positions()[0], np.eye(4)),
        )

    def test_multiple_stimulators_nifti(self, full_tcd_coil: TmsCoil, tmp_path: Path):
        coil = deepcopy(full_tcd_coil)
        coil.resolution = np.array([1, 1, 1])
        coil.limits = np.array([[0, 25], [0, 25], [0, 25]])
        stimulator = TmsStimulator("Stim 1", None, None, None)
        coil.elements[0].stimulator = stimulator
        coil.elements[1].stimulator = stimulator
        coil.elements[2].stimulator = TmsStimulator("Stim 2", None, None, None)

        coil_sampled = coil.as_sampled_squashed()

        coil.write_nifti(str(tmp_path / "test.nii.gz"))
        coil_nifti = TmsCoil.from_nifti(str(tmp_path / "test.nii.gz"))

        assert len(coil_sampled.elements) == len(coil_nifti.elements)

        for sampled_element, nifti_element in zip(
            coil_sampled.elements, coil_nifti.elements
        ):
            assert isinstance(sampled_element, SampledGridPointElements)
            assert isinstance(nifti_element, SampledGridPointElements)
            np.testing.assert_allclose(
                sampled_element.get_a_field(
                    coil_sampled.get_sample_positions()[0], np.eye(4)
                ),
                nifti_element.get_a_field(
                    coil_sampled.get_sample_positions()[0], np.eye(4)
                ),
            )

        np.testing.assert_allclose(
            coil_sampled.get_a_field(coil_sampled.get_sample_positions()[0], np.eye(4)),
            coil_nifti.get_a_field(coil_nifti.get_sample_positions()[0], np.eye(4)),
        )


class TestCoilMesh:
    def test_generate_full_coil_mesh(self, full_tcd_coil: TmsCoil):
        coil_mesh = full_tcd_coil.get_mesh(apply_deformation=False)

        assert len(np.unique(coil_mesh.elm.tag1)) == len(full_tcd_coil.elements) * 4 + 3
        assert is_close_subset_of(
            full_tcd_coil.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.min_distance_points, coil_mesh.nodes.node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.intersect_points, coil_mesh.nodes.node_coord
        )

        for coil_element in full_tcd_coil.elements:
            assert is_close_subset_of(
                coil_element.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.min_distance_points, coil_mesh.nodes.node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.intersect_points, coil_mesh.nodes.node_coord
            )

            if isinstance(coil_element, PositionalTmsCoilElements):
                assert is_close_subset_of(
                    coil_element.points, coil_mesh.nodes.node_coord
                )

            if isinstance(coil_element, DipoleElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, LineSegmentElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, SampledGridPointElements):
                voxel_coordinates = np.array(
                    list(
                        np.ndindex(
                            coil_element.data.shape[0],
                            coil_element.data.shape[1],
                            coil_element.data.shape[2],
                        )
                    )
                )

                points = (
                    voxel_coordinates @ coil_element.affine[:3, :3].T
                    + coil_element.affine[None, :3, 3]
                )

                targets = points + coil_element.data.reshape(-1, 3)

                assert is_close_subset_of(points, coil_mesh.nodes.node_coord)
                assert is_close_subset_of(targets, coil_mesh.nodes.node_coord)

    def test_generate_full_coil_mesh_with_affine(self, full_tcd_coil: TmsCoil):
        coil_affine = np.array(
            [
                [np.cos(1.23323), -np.sin(1.23323), 0, 111],
                [np.sin(1.23323), np.cos(1.23323), 0, -32],
                [0, 0, 1, 232],
                [0, 0, 0, 1],
            ]
        )
        inv_coil_affine = np.linalg.inv(coil_affine)
        coil_mesh = full_tcd_coil.get_mesh(
            coil_affine=coil_affine, apply_deformation=False
        )

        coil_mesh_node_coord = (
            coil_mesh.nodes.node_coord @ inv_coil_affine[:3, :3].T
            + inv_coil_affine[None, :3, 3]
        )
        print(list(coil_mesh.nodes.node_coord))
        assert len(np.unique(coil_mesh.elm.tag1)) == len(full_tcd_coil.elements) * 4 + 3
        assert is_close_subset_of(
            full_tcd_coil.casing.mesh.nodes.node_coord, coil_mesh_node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.min_distance_points, coil_mesh_node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.intersect_points, coil_mesh_node_coord
        )

        for coil_element in full_tcd_coil.elements:
            assert is_close_subset_of(
                coil_element.casing.mesh.nodes.node_coord, coil_mesh_node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.min_distance_points, coil_mesh_node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.intersect_points, coil_mesh_node_coord
            )

            if isinstance(coil_element, PositionalTmsCoilElements):
                assert is_close_subset_of(coil_element.points, coil_mesh_node_coord)

            if isinstance(coil_element, DipoleElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values, coil_mesh_node_coord
                )

            if isinstance(coil_element, LineSegmentElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values, coil_mesh_node_coord
                )

            if isinstance(coil_element, SampledGridPointElements):
                voxel_coordinates = np.array(
                    list(
                        np.ndindex(
                            coil_element.data.shape[0],
                            coil_element.data.shape[1],
                            coil_element.data.shape[2],
                        )
                    )
                )

                points = (
                    voxel_coordinates @ coil_element.affine[:3, :3].T
                    + coil_element.affine[None, :3, 3]
                )

                targets = points + coil_element.data.reshape(-1, 3)
                print(points)
                print(targets)
                print(list(coil_mesh_node_coord))
                assert is_close_subset_of(points, coil_mesh_node_coord)
                # TODO assert is_close_subset_of(targets, coil_mesh_node_coord)

    def test_generate_coil_mesh_exclude_casings(self, full_tcd_coil: TmsCoil):
        coil_mesh = full_tcd_coil.get_mesh(
            include_casing=False, apply_deformation=False
        )

        assert len(np.unique(coil_mesh.elm.tag1)) == len(full_tcd_coil.elements) * 3 + 2
        assert not is_close_to_any(
            full_tcd_coil.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.min_distance_points, coil_mesh.nodes.node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.intersect_points, coil_mesh.nodes.node_coord
        )

        for coil_element in full_tcd_coil.elements:
            assert not is_close_to_any(
                coil_element.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.min_distance_points, coil_mesh.nodes.node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.intersect_points, coil_mesh.nodes.node_coord
            )

            if isinstance(coil_element, PositionalTmsCoilElements):
                assert is_close_subset_of(
                    coil_element.points, coil_mesh.nodes.node_coord
                )

            if isinstance(coil_element, DipoleElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, LineSegmentElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, SampledGridPointElements):
                voxel_coordinates = np.array(
                    list(
                        np.ndindex(
                            coil_element.data.shape[0],
                            coil_element.data.shape[1],
                            coil_element.data.shape[2],
                        )
                    )
                )

                points = (
                    voxel_coordinates @ coil_element.affine[:3, :3].T
                    + coil_element.affine[None, :3, 3]
                )

                targets = points + coil_element.data.reshape(-1, 3)

                assert is_close_subset_of(points, coil_mesh.nodes.node_coord)
                assert is_close_subset_of(targets, coil_mesh.nodes.node_coord)

    def test_generate_coil_mesh_exclude_optimization_points(
        self, full_tcd_coil: TmsCoil
    ):
        coil_mesh = full_tcd_coil.get_mesh(
            include_optimization_points=False, apply_deformation=False
        )

        assert len(np.unique(coil_mesh.elm.tag1)) == len(full_tcd_coil.elements) * 2 + 1
        assert is_close_subset_of(
            full_tcd_coil.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
        )
        assert not is_close_to_any(
            full_tcd_coil.casing.min_distance_points, coil_mesh.nodes.node_coord
        )
        assert not is_close_to_any(
            full_tcd_coil.casing.intersect_points, coil_mesh.nodes.node_coord
        )

        for coil_element in full_tcd_coil.elements:
            assert is_close_subset_of(
                coil_element.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
            )
            assert not is_close_to_any(
                coil_element.casing.min_distance_points, coil_mesh.nodes.node_coord
            )
            assert not is_close_to_any(
                coil_element.casing.intersect_points, coil_mesh.nodes.node_coord
            )

            if isinstance(coil_element, PositionalTmsCoilElements):
                assert is_close_subset_of(
                    coil_element.points, coil_mesh.nodes.node_coord
                )

            if isinstance(coil_element, DipoleElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, LineSegmentElements):
                assert is_close_subset_of(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, SampledGridPointElements):
                voxel_coordinates = np.array(
                    list(
                        np.ndindex(
                            coil_element.data.shape[0],
                            coil_element.data.shape[1],
                            coil_element.data.shape[2],
                        )
                    )
                )

                points = (
                    voxel_coordinates @ coil_element.affine[:3, :3].T
                    + coil_element.affine[None, :3, 3]
                )

                targets = points + coil_element.data.reshape(-1, 3)

                assert is_close_subset_of(points, coil_mesh.nodes.node_coord)
                assert is_close_subset_of(targets, coil_mesh.nodes.node_coord)

    def test_generate_coil_mesh_exclude_optimization_points(
        self, full_tcd_coil: TmsCoil
    ):
        coil_mesh = full_tcd_coil.get_mesh(
            include_coil_elements=False, apply_deformation=False
        )

        assert len(np.unique(coil_mesh.elm.tag1)) == len(full_tcd_coil.elements) * 3 + 3
        assert is_close_subset_of(
            full_tcd_coil.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.min_distance_points, coil_mesh.nodes.node_coord
        )
        assert is_close_subset_of(
            full_tcd_coil.casing.intersect_points, coil_mesh.nodes.node_coord
        )

        for coil_element in full_tcd_coil.elements:
            assert is_close_subset_of(
                coil_element.casing.mesh.nodes.node_coord, coil_mesh.nodes.node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.min_distance_points, coil_mesh.nodes.node_coord
            )
            assert is_close_subset_of(
                coil_element.casing.intersect_points, coil_mesh.nodes.node_coord
            )

            if isinstance(coil_element, PositionalTmsCoilElements):
                assert not is_close_to_any(
                    coil_element.points, coil_mesh.nodes.node_coord
                )

            if isinstance(coil_element, DipoleElements):
                assert not is_close_to_any(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, LineSegmentElements):
                assert not is_close_to_any(
                    coil_element.points + coil_element.values,
                    coil_mesh.nodes.node_coord,
                )

            if isinstance(coil_element, SampledGridPointElements):
                voxel_coordinates = np.array(
                    list(
                        np.ndindex(
                            coil_element.data.shape[0],
                            coil_element.data.shape[1],
                            coil_element.data.shape[2],
                        )
                    )
                )

                points = (
                    voxel_coordinates @ coil_element.affine[:3, :3].T
                    + coil_element.affine[None, :3, 3]
                )

                targets = points + coil_element.data.reshape(-1, 3)

                assert not is_close_to_any(points, coil_mesh.nodes.node_coord)
                assert not is_close_to_any(targets, coil_mesh.nodes.node_coord)


class TestPositionOptimization:
    def test_simple_optimization(self, sphere3_msh: Msh):
        skin_surface = sphere3_msh.crop_mesh(tags=[1005])
        coil_affine = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 100], [0, 0, 0, 1]]
        )
        casings = [
            TmsCoilModel(
                Msh(
                    Nodes(
                        np.array(
                            [[-20, -20, 0], [-20, 20, 0], [20, -20, 0], [20, 20, 0]]
                        )
                    ),
                    Elements(triangles=np.array([[0, 1, 2], [3, 2, 1]]) + 1),
                ),
                None,
                None,
            ),
            TmsCoilModel(
                Msh(
                    Nodes(
                        np.array(
                            [[-20, -60, 0], [-20, -20, 0], [20, -60, 0], [20, -20, 0]]
                        )
                    ),
                    Elements(triangles=np.array([[0, 1, 2], [3, 2, 1]]) + 1),
                ),
                None,
                None,
            ),
            TmsCoilModel(
                Msh(
                    Nodes(
                        np.array([[-20, 20, 0], [-20, 60, 0], [20, 20, 0], [20, 60, 0]])
                    ),
                    Elements(triangles=np.array([[0, 1, 2], [3, 2, 1]]) + 1),
                ),
                None,
                None,
            ),
        ]
        deformations = [
            TmsCoilRotation(0, (0, 90), np.array([0, -20, 0]), np.array([40, -20, 0])),
            TmsCoilRotation(
                0,
                (0, 90),
                np.array([40, 20, 0]),
                np.array([0, 20, 0]),
            ),
        ]
        coil = TmsCoil(
            None,
            None,
            None,
            None,
            None,
            None,
            [
                LineSegmentElements(
                    None, casings[0], None, np.array([[1, 2, 3]]), None, None
                ),
                LineSegmentElements(
                    None,
                    casings[1],
                    [deformations[0]],
                    np.array([[1, 2, 3]]),
                    None,
                    None,
                ),
                LineSegmentElements(
                    None,
                    casings[2],
                    [deformations[1]],
                    np.array([[1, 2, 3]]),
                    None,
                    None,
                ),
            ],
        )

        before, after, affine_after = coil.optimize_deformations(
            skin_surface, coil_affine
        )
        assert before > after
        assert after < 6.1
        np.testing.assert_allclose(coil_affine, affine_after)
        assert not coil._get_current_deformation_scores(
            skin_surface.get_AABBTree(), affine_after
        )[0]

    def test_optimization_with_global_translation(self, sphere3_msh: Msh):
        skin_surface = sphere3_msh.crop_mesh(tags=[1005])

        coil_affine = np.array(
            [[1, 0, 0, -4], [0, 1, 0, 3], [0, 0, 1, 110], [0, 0, 0, 1]]
        )
        casings = [
            TmsCoilModel(
                Msh(
                    Nodes(
                        np.array(
                            [[-20, -20, 0], [-20, 20, 0], [20, -20, 0], [20, 20, 0]]
                        )
                    ),
                    Elements(triangles=np.array([[0, 1, 2], [3, 2, 1]]) + 1),
                ),
                None,
                None,
            ),
            TmsCoilModel(
                Msh(
                    Nodes(
                        np.array(
                            [[-20, -60, 0], [-20, -20, 0], [20, -60, 0], [20, -20, 0]]
                        )
                    ),
                    Elements(triangles=np.array([[0, 1, 2], [3, 2, 1]]) + 1),
                ),
                None,
                None,
            ),
            TmsCoilModel(
                Msh(
                    Nodes(
                        np.array([[-20, 20, 0], [-20, 60, 0], [20, 20, 0], [20, 60, 0]])
                    ),
                    Elements(triangles=np.array([[0, 1, 2], [3, 2, 1]]) + 1),
                ),
                None,
                None,
            ),
        ]
        deformations = [
            TmsCoilRotation(0, (0, 90), np.array([0, -20, 0]), np.array([40, -20, 0])),
            TmsCoilRotation(
                0,
                (0, 90),
                np.array([40, 20, 0]),
                np.array([0, 20, 0]),
            ),
        ]
        coil = TmsCoil(
            None,
            None,
            None,
            None,
            None,
            None,
            [
                LineSegmentElements(
                    None, casings[0], None, np.array([[1, 2, 3]]), None, None
                ),
                LineSegmentElements(
                    None,
                    casings[1],
                    [deformations[0]],
                    np.array([[1, 2, 3]]),
                    None,
                    None,
                ),
                LineSegmentElements(
                    None,
                    casings[2],
                    [deformations[1]],
                    np.array([[1, 2, 3]]),
                    None,
                    None,
                ),
            ],
        )
        before, after, affine_after = coil.optimize_deformations(
            skin_surface, coil_affine, np.array([[-5, 5], [-5, 5], [-20, 20]])
        )

        assert before > after
        assert after < 1.0
        assert not np.allclose(coil_affine, affine_after)
        assert not coil._get_current_deformation_scores(
            skin_surface.get_AABBTree(), affine_after
        )[0]
