from typing import Optional
import os
import pandas as pd
from pyteomics import mztab
from pyteomics import mgf
from tqdm import tqdm
import fileinput

def get_line_count(file_path: str) -> int:
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

def add_predictions_to_mgf(
    mode: str,
    mgf_file_path: str,
    pred_path_1: Optional[str] = None,
    pred_path_2: Optional[str] = None
) -> None:
    """
    Add database and/or casanovo predictions to MGF file.

    Args:
    - mode (str): Mode of operation. Can be 'database', 'casanovo', or 'both'.
    - mgf_file_path (str): Path to the MGF file.
    - pred_path_1 (str, optional): Path to the first prediction file.
    - pred_path_2 (str, optional): Path to the second prediction file.

    Returns:
    - None 

    Outputs:
    - The predictions are added to the .mgf file.
    """
    # Read MGF file
    opened_MGF_spectra = mgf.read(mgf_file_path, convert_arrays=True)

    line_count = get_line_count(mgf_file_path)

    # Add database prediction if available
    if mode in ["database", "db", "both"]:
        database_df_origin = pd.read_csv(pred_path_1)
        database_df_origin = database_df_origin.sort_values(by='Precursor Id', ascending=True)
        precursor_id_list = database_df_origin['Precursor Id'].tolist()

        with tqdm(total=line_count, desc="Adding DB predictions", unit="line") as progress_bar:
            with fileinput.FileInput(mgf_file_path, inplace=True, backup='.bak') as file:
                precursor_id = [None]
                for line in file:
                    line = add_database_data(line, database_df_origin, precursor_id)
                    print(line, end='')
                    progress_bar.update(1)
                progress_bar.set_postfix_str("Done!")
                    
    
    # Add Casanovo prediction if available
    if mode in ["casanovo", "cas", "both"]:
        if mode in ["casanovo", "cas"]:
            casanovo_file = mztab.MzTab(pred_path_1)
        else:
            casanovo_file = mztab.MzTab(pred_path_2)
            
        casanovo_df_origin = pd.DataFrame(casanovo_file.spectrum_match_table)
        casanovo_df_origin['Precursor Id'] = casanovo_df_origin['spectra_ref'].str.extract(r'scan=(\d+)', expand=False).astype(int)

        with tqdm(total=line_count, desc="Adding CASA predictions", unit="line") as progress_bar:
            with fileinput.FileInput(mgf_file_path, inplace=True, backup='.bak') as file:
                precursor_id = [None]
                for line in file:
                    line = add_casanovo_data(line, casanovo_df_origin, precursor_id)
                    print(line, end='')
                    progress_bar.update(1)
                progress_bar.set_postfix_str("Done!")

def add_database_data(line, database_df, precursor_id):
    if line.startswith('TITLE='):
        precursor_id[0] = int(line.split('=')[1].strip())
    elif line.startswith('SEQ='):
        if precursor_id[0] in database_df['Precursor Id'].values:
            peptide = database_df.loc[database_df['Precursor Id'] == precursor_id[0], 'Peptide'].iloc[0]
            return f'SEQ={peptide}\n'
    return line

def add_casanovo_data(line, casanovo_df, precursor_id):
    if line.startswith('TITLE='):
        precursor_id[0] = int(line.split('=')[1].strip())
    elif line.startswith('CASANOVO_SEQ='):
        if precursor_id[0] in casanovo_df['Precursor Id'].values:
            peptide = casanovo_df.loc[casanovo_df['Precursor Id'] == precursor_id[0], 'sequence'].iloc[0]
            return f'CASANOVO_SEQ={peptide}\n'
    elif line.startswith('CASANOVO_AA_SCORES='):
        if precursor_id[0] in casanovo_df['Precursor Id'].values:
            aa_scores = casanovo_df.loc[casanovo_df['Precursor Id'] == precursor_id[0], 'opt_ms_run[1]_aa_scores'].iloc[0]
            return f'CASANOVO_AA_SCORES=[{aa_scores}]\n'
    return line