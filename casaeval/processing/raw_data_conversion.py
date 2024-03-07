import timsrust_pyo3
import os
from pyteomics import mgf
from tqdm import tqdm
from pyteomics.auxiliary import unitfloat
from typing import Dict, List, Tuple

class TR_to_MGF_Spectrum:
    def __init__(self, title: str, pepmass: Tuple[float, float], charge: List[int], ion_mobility: float, scans: str, rtinseconds: unitfloat, casanovo_seq: str, casanovo_aa_scores: List[float], seq: str, mz_array: List[float], intensity_array: List[float]):
        self.params = {
            'title': title,
            'pepmass': pepmass,
            'charge': charge,
            'ion_mobility': ion_mobility,
            'scans': scans,
            'rtinseconds': rtinseconds,
            'casanovo_seq': casanovo_seq,
            'casanovo_aa_scores': [],
            'seq': seq
        }
        self.mz_array = mz_array
        self.intensity_array = intensity_array

    def to_dict(self) -> dict:
        return {
            'params': self.params,
            'm/z array': self.mz_array,
            'intensity array': self.intensity_array
        }


def save_file_with_path(file_name, output_path=None):
    if output_path:
        # Save the plot with the provided path
        return f"{output_path}/{file_name}.mgf"
    else:
        # Save the plot with the default filename "plot.png"
        return f"{file_name}.mgf"

def converter_timsd_to_mgf(
    file_path: str,
    output_path: str
) -> None:
    """
    Convert TIMS data to MGF format.

    Args:
    - file_path (str): Path to the TIMS data file.
    - output_path (str): Path to the output directory.

    Returns:
    - None

    Outputs:
    - [.mgf] file of the raw TIMS data file [.d].
    """
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(os.path.basename(file_path))[1]

    print('Extracting spectra and frames of raw TIMS/PASEF data...')
    reader = timsrust_pyo3.TimsReader(file_path)
    TR_all_spectra = reader.read_all_spectra()
    print('Spectra Data Extraction DONE.')
    TR_all_frames = reader.read_all_frames()
    print('Frame Data Extraction DONE.')
    TR_all_frames_indexcorrected = {}
    for frame in TR_all_frames:
        TR_all_frames_indexcorrected[frame.index] = frame

    with tqdm(total=2*len(TR_all_spectra), desc="Conversion Progress", unit="iteration") as progress_bar_universal:
        with tqdm(total=len(TR_all_spectra), desc="Restructuring Data", unit="spectrum", leave=False) as progress_bar_1:
            TR_to_mgf_spectra_list = []
            for spectrum in TR_all_spectra:
                spectrum_instance = TR_to_MGF_Spectrum(
                    title=spectrum.precursor.index,
                    #f'{file_name_without_extension}:Precursor_{spectrum.precursor.index}'
                    pepmass=(spectrum.precursor.mz, None),
                    charge=[spectrum.precursor.charge],
                    ion_mobility=spectrum.precursor.im,
                    scans=f'F{spectrum.precursor.frame_index}:{spectrum.precursor.intensity}',
                    rtinseconds=unitfloat(TR_all_frames_indexcorrected[spectrum.precursor.frame_index].rt, 'second'),
                    casanovo_seq='',
                    casanovo_aa_scores=[],
                    seq='',
                    mz_array=spectrum.mz_values,
                    intensity_array=spectrum.intensities
                )
                TR_to_mgf_spectra_list.append(spectrum_instance.to_dict())
                progress_bar_1.update(1)
                progress_bar_universal.update(1)
            progress_bar_1.set_postfix_str("Done!")
                
        with tqdm(total=len(TR_to_mgf_spectra_list), desc="Writing Data as .mgf", unit="spectrum", leave=False) as progress_bar_2:
            with open(save_file_with_path(file_name_without_extension, output_path), 'w') as mgf_file:
                for spectrum in TR_to_mgf_spectra_list:
                    mgf.write([spectrum], mgf_file)
                    progress_bar_2.update(1)
                    progress_bar_universal.update(1)
                progress_bar_2.set_postfix_str("Done!")