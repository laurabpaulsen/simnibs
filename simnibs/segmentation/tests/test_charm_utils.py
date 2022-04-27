import numpy as np
import pytest
import nibabel as nib
import os
from scipy import ndimage
from scipy.io import loadmat
from ... import SIMNIBSDIR
from .. import charm_utils
from ..samseg import initVisualizer
from ..samseg.io import kvlReadSharedGMMParameters
from ..samseg.simnibs_segmentation_utils import writeBiasCorrectedImagesAndSegmentation

@pytest.fixture(scope='module')
def testernie_nii():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'ernie_T1_ds5.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testmni_nii():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'MNI_test_ds5.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testtemplate_nii():
    fn = os.path.join(
        SIMNIBSDIR, 'segmentation', 'atlases', 'charm_atlas_mni', 'template.nii')
    return fn

@pytest.fixture(scope='module')
def testaffinemesh_msh():
    fn = os.path.join(
        SIMNIBSDIR, 'segmentation', 'atlases', 'charm_atlas_mni', 'affine_no_neck.txt.gz')
    return fn

@pytest.fixture(scope='module')
def testcubenoise_nii():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'cube_noise.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testcube_nii():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'cube.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testcubeatlas_path():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'cube_atlas')
    return fn


def _calc_dice(vol1, vol2):
    return np.sum(vol2[vol1])*2.0 / (np.sum(vol1) + np.sum(vol2))


def generate_label_arr(ndim=2):
    """Generate an array of labels."""
    size = ndim * (10,)
    arr = np.zeros(size, dtype=np.uint16)
    arr[5:] = 1
    arr[0, :5] = 2
    arr[5, 5] = 3
    arr[7:, 7:] = 5
    return arr


# all cases invoke remapping!
test_label_unassigned_elements_inputs = (
    (2, [0, 1, 2, 3, 5], 3, None, 0),
    (3, None, 3, None, 1),
    (3, None, 3, [1], 0),
    (5, None, 3, None, 1),
    (5, None, 3, [1], 5),  # should show a warning
)


# @pytest.mark.filterwarnings("ignore:Some elements could not be labeled")
@pytest.mark.parametrize(
    "label_unassign, labels, window_size, ignore_labels, expected_label",
    test_label_unassigned_elements_inputs,
)
def test_label_unassigned_elements(
    label_unassign,
    labels,
    window_size,
    ignore_labels,
    expected_label,
):
    label_arr = generate_label_arr(3)
    expected_arr = label_arr.copy()
    expected_arr[label_arr == label_unassign] = expected_label
    np.testing.assert_allclose(
        expected_arr,
        charm_utils.label_unassigned_elements(
            label_arr, label_unassign, labels, window_size, ignore_labels
        ),
    )

def test_sanlm(tmpdir, testernie_nii):
    denoised_scan = tmpdir.mkdir("denoised").join("denoised.nii.gz")
    input_scan = nib.load(testernie_nii)
    input_data = input_scan.get_fdata()
    charm_utils._denoise_input_and_save(testernie_nii, denoised_scan)
    output_scan = nib.load(denoised_scan)
    output_data = output_scan.get_fdata()
    assert input_data.var() > output_data.var()

def test_mni_affine(tmpdir, testmni_nii):
    trans_scan_name = tmpdir.mkdir("shifted").join("shifted_MNI.nii.gz")
    input_scan = nib.load(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat@input_scan.affine
    trans_mni = nib.Nifti1Image(input_scan.get_fdata(), trans_affine)
    nib.save(trans_mni, trans_scan_name)
    trans_settings = {"translation_scale": -100,
                      "max_iter": 10,
                      "shrink_factors": [0],
                      "smoothing_factors": [4.0],
                      "center_of_mass": True,
                      "samp_factor": 1.0,
                      "bg_value": 0}
    RAS2LPS = np.diag([-1, -1, 1, 1])
    estimated_trans_mat = charm_utils._init_atlas_affine(str(trans_scan_name),
                                                         testmni_nii,
                                                         trans_settings)
    np.testing.assert_allclose(trans_mat,
                               RAS2LPS@estimated_trans_mat@RAS2LPS)


def test_atlas_affine(tmpdir, testmni_nii, testtemplate_nii, testaffinemesh_msh):
    trans_scan_name = tmpdir.mkdir("shifted").join("shifted_MNI.nii.gz")
    template_coregistered_name = tmpdir.join('template_coregistered.mgz')
    input_scan = nib.load(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat@input_scan.affine
    trans_mni = nib.Nifti1Image(input_scan.get_fdata(), trans_affine)
    nib.save(trans_mni, trans_scan_name)
    init_atlas_settings = {"affine_scales": [[1, 1, 1]],
                           "affine_rotations": [0],
                           "affine_horizontal_shifts": [0],
                           "affine_vertical_shifts": [0],
                           "neck_search_bounds": [0, 0],
                           "downsampling_factor_affine": 1.0}
    visualizer = initVisualizer(False, False)
    charm_utils._register_atlas_to_input_affine(str(trans_scan_name),
                                                testtemplate_nii,
                                                testaffinemesh_msh,
                                                testaffinemesh_msh,
                                                testaffinemesh_msh,
                                                str(tmpdir),
                                                str(template_coregistered_name),
                                                init_atlas_settings,
                                                None,
                                                visualizer,
                                                True,
                                                init_transform=None,
                                                world_to_world_transform_matrix=None)

    matrices = loadmat(str(tmpdir.join('coregistrationMatrices.mat')))
    w2w = matrices['worldToWorldTransformMatrix']
    #I wouldn't expect the match to be as good as for the mni-mni reg above,
    #so relaxing the tolerances here
    np.testing.assert_allclose(trans_mat[:3,3],
                               w2w[:3,3], rtol=1e-5, atol=3)
    np.testing.assert_allclose(trans_mat[:3,:3], w2w[:3, :3], rtol=5e-2, atol=5e-2)

def test_T1T2(tmpdir, testmni_nii):
    trans_scan_name = tmpdir.mkdir("shifted").join("shifted_MNI.nii.gz")
    registered_scan_name = tmpdir.join("registered.nii.gz")
    input_scan = nib.load(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat@input_scan.affine
    trans_mni = nib.Nifti1Image(input_scan.get_fdata(), trans_affine)
    nib.save(trans_mni, trans_scan_name)
    charm_utils._registerT1T2(str(trans_scan_name),
                              testmni_nii,
                              str(registered_scan_name))
    reg_scan = nib.load(str(registered_scan_name))
    assert (np.corrcoef(reg_scan.get_fdata().flatten(), trans_mni.get_fdata().flatten()))[0,1] > 0.99

def test_largest_components():
    se = ndimage.generate_binary_structure(3, 3)
    test_array = np.zeros((5,5,5))
    test_array[0:2, 0:2, 0:2] = 1
    test_array[3:5, 3:5, 3:5] = 2

    assert charm_utils._get_largest_components(test_array, se).sum() == 16
    assert charm_utils._get_largest_components(test_array,se, vol_limit=9).sum() == 0
    assert charm_utils._get_largest_components(test_array, se, num_limit=1).sum() == 8

def test_smoothfill():
    test_array = generate_label_arr(3)
    unass = np.zeros_like(test_array, dtype=bool)
    unass[2:4,3:5,5:7] = True
    unass[6,6,6] = True
    unass[7:9, 6:8, 1:3] = True
    tissue_dict = {'bg': 0, 'first': 1, 'second': 2, 'WM': 3, 'fifth': 5}
    expected_array = test_array.copy()
    test_array[unass] = -1
    charm_utils._smoothfill(test_array, unass, tissue_dict)
    assert (test_array == 65535).sum() == 0
    np.testing.assert_allclose(expected_array == 3, test_array == 3)

def test_fill_missing():
    test_array = generate_label_arr(3)
    unass = np.zeros_like(test_array, dtype=bool)
    unass[2:4,3:5,5:7] = True
    unass[6,6,6] = True
    unass[7:9, 6:8, 1:3] = True
    expected_array = test_array.copy()
    test_array[unass] = -1
    charm_utils._fill_missing(test_array, unass)
    assert (test_array == 65535).sum() == 0
    np.testing.assert_allclose(expected_array == 3, test_array == 3)

def test_segmentation(tmpdir, testcube_nii, testcubenoise_nii, testcubeatlas_path):
    seg_dir = tmpdir.mkdir("segmentation")

    seg_settings = {"downsampling_targets": 1.0,
                    "bias_kernel_width": 100,
                    "background_mask_sigma": 1.0,
                    "background_mask_threshold": 0.001,
                    "mesh_stiffness": 0.1,
                    "diagonal_covariances": False}

    user_opts = {
        "multiResolutionSpecification": [
            {
                "atlasFileName": os.path.join(testcubeatlas_path,'atlas.txt.gz'),
                "targetDownsampledVoxelSpacing": 1.0,
                "maximumNuberOfIterations": 10,
                "estimateBiasField": False
            }
        ]
    }

    shared_gmm_params = kvlReadSharedGMMParameters(os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'))
    user_specs = {
            "atlasFileName": os.path.join(testcubeatlas_path, "atlas.txt.gz"),
            "biasFieldSmoothingKernelSize": seg_settings["bias_kernel_width"],
            "brainMaskingSmoothingSigma": seg_settings["background_mask_sigma"],
            "brainMaskingThreshold": seg_settings["background_mask_threshold"],
            "K": seg_settings["mesh_stiffness"],
            "useDiagonalCovarianceMatrices": seg_settings["diagonal_covariances"],
            "sharedGMMParameters": shared_gmm_params,
    }

    visualizer = initVisualizer(False, False)
    param_estimates = charm_utils._estimate_parameters(
                        str(seg_dir),
                        os.path.join(testcubeatlas_path, 'template.nii.gz'),
                        testcubeatlas_path,
                        [testcubenoise_nii],
                        seg_settings,
                        os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'),
                        visualizer,
                        user_optimization_options=user_opts,
                        user_model_specifications=user_specs)

    bias_corr = os.path.join(str(seg_dir), 'bias_temp.nii.gz')
    seg = os.path.join(str(seg_dir), 'seg.nii.gz')
    mock_tissue_settings = {"segmentation_tissues": {"CSF": -1}}
    writeBiasCorrectedImagesAndSegmentation([bias_corr], seg, param_estimates, mock_tissue_settings, 1)

    orig_cube = nib.load(testcube_nii)
    est_cube = nib.load(seg)
    dice = _calc_dice(orig_cube.get_fdata()==1, est_cube.get_fdata()==1)
    print("Dice score: "+str(dice))
    assert dice > 0.95