


import mne
from mne.utils import logger
import numpy as np

from typing import Union
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simnibs.utils.csv_reader import write_csv_positions
def prepare_meg_sensor_array(
        fname: Union[Path, str],
        info: Union[Path, str, mne.Info],
        trans_head_mr: Union[Path, str, mne.Transform],
        program = "MNE",
        accuracy = "accurate"
):  
    # load the mne info and trans objects if string or path was provided
    if not isinstance(info, mne.Info):
        info = mne.io.read_info(info)
    
    if not isinstance(trans_head_mr, mne.Transform):
        trans_head_mr = mne.read_trans(trans_head_mr)

    if program.lower() != "mne":
        msg = """Only implemented for MNE"""
        raise NotImplementedError(msg)
    
    fname = Path(fname)
    
    if fname.suffix != ".csv":
        fname = fname.with_suffix(".csv")

    # # sensors are in headspace and m so transform to MRI coordinates
    # and mm
    trans_dev_head = info["dev_head_t"]
    trans_head_mri = mne.transforms._ensure_trans(trans_head_mr, "head", "mri")    
    logger.info(f"Transforming from device to head using {trans_dev_head}")
    logger.info(f"Transforming from head to mr using {trans_head_mri}")

    trans_dev_mri = mne.transforms.combine_transforms(trans_dev_head, trans_head_mri, "meg", "mri")


    logger.info(f"Creating meg coil definitions in mri space using accuracy: {accuracy}")
    coilset = mne.forward._create_meg_coils(info["chs"], acc = accuracy, t = trans_dev_mri)

    logger.info(f"Writing coil postions and orientations to file: {fname}")

    if not fname.exists():
        fname.parent.mkdir(parents=True)
    write_csv_positions(
        filename = fname, 
        types = [coil["type"] for coil in coilset],
        coordinates = np.array([coil["rmag"][0] for coil in coilset]),
        extra = np.array([coil["cosmag"][0] for coil in coilset]),
        name = [coil["chname"] for coil in coilset],

        header=["type", "x", "y", "z", "ex", "ey", "ez", "ch_name"]
    
    )
    #return coilset


