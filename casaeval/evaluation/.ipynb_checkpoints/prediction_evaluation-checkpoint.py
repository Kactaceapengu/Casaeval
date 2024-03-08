import os
import pandas as pd
from pandas import DataFrame, Series

import json
import re
from collections import defaultdict
from itertools import chain

from .evaluate import aa_match_batch, aa_match_metrics, aa_precision_recall # from casanovo/denovo/evaluate.py, modified functions for
from .masses import PeptideMass # get PeptideMass for a dictionary of all tokens
from pyteomics import mgf

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

################## Function Collections for Casanovo Evaluation

##############
############## READ AND STRUCTURE .mgf DATA
#############




###### READ AND STRUCTURE .mgf DATA containing database (true) sequences as 'SEQ' and casanovo predictions as 'CASANOVO_SEQ':

# Define a function to convert string representation of list to actual list of floats (Helper function)
def parse_float_list(string_list):
    if string_list == '[]':  # Check for empty string
        return []  # Return empty list
    else:
        return json.loads(string_list)  # Convert string to list of floats using json module

def mgf_to_df(
    mgf_path: str,
    modified_bool: bool
) -> pd.DataFrame:
    """
    Process MGF file and dataframes to return a matched dataframe.

    Args:
    - mgf_path (str): Path to the MGF file.
    - modified_bool (bool): Boolean to specify, if .mgf prediction data is modified or not.

    Returns:
    - pd.DataFrame: Dataframe containing matched data.
    """
    # Read MGF file
    opened_MGF_spectra_pred = mgf.read(mgf_path, convert_arrays=True)
    
    # Initialize empty lists to store data
    params_list = []
    mz_array_list = []
    intensity_array_list = []

    # Iterate through spectra and extract data
    with tqdm(total=len(opened_MGF_spectra_pred), desc="Reading Progress", unit="spectrum") as progress_bar:
        for spectrum in opened_MGF_spectra_pred:
            params = spectrum['params']
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            
            # Append data to lists
            params_list.append(params)
            mz_array_list.append(mz_array)
            intensity_array_list.append(intensity_array)
            progress_bar.update(1)
        progress_bar.set_postfix_str("Done!")

    # Create DataFrame for 'params' and expand the dictionary into columns
    params_df = pd.DataFrame(params_list)

    # Create DataFrame for 'm/z array' and 'intensity array'
    mz_intensity_df = pd.DataFrame({
        'm/z array': mz_array_list,
        'intensity array': intensity_array_list
    })

    # Concatenate all DataFrames
    df = pd.concat([params_df, mz_intensity_df], axis=1)

    # Apply the function to the entire column
    df.loc[:,'casanovo_aa_scores'] = df['casanovo_aa_scores'].apply(parse_float_list)

    if modified_bool:
        #Adapt modification tokens of database to casanovo
        database_modifications_df = df[df['seq'].str.contains(r'\d')]
    
        # Assuming database_modifications_df is your DataFrame
        modifications_dict = defaultdict(list)
    
        pattern_to_find = r'\((.*?)\)'
    
        # Adapt db token function:
        df.loc[:,'seq'] = modify_peptide_sequences(df.loc[:,'seq']).values
        
        # Add booleans to the df
        df.loc[:,'db_modified'] = df['seq'].str.contains(r'\d')
        df.loc[:,'cs_modified'] = df['casanovo_seq'].str.contains(r'\d')


    # Filter and return matched dataframe
    matched_df = df[(df['seq'] != '') & (df['casanovo_seq'] != '')]
    matched_df.reset_index(drop=True, inplace=True)
    
    casanovo_unmatched_df = df[(df['seq'] == '') & (df['casanovo_seq'] != '')].copy()
    casanovo_unmatched_df.loc[:,'mean_aa_score'] = [np.mean(scores) for scores in casanovo_unmatched_df.loc[:,'casanovo_aa_scores'].tolist()]

    return df, matched_df, casanovo_unmatched_df

####### MODIFICATIONS

def rearrange_modifications(peptide_sequence: str) -> str:
    """
    Rearrange modifications in a peptide sequence from the format 'A(+10)B(-20)C' to '+10A-20B-C'.

    Args:
        peptide_sequence (str): Peptide sequence containing modifications.

    Returns:
        str: Peptide sequence with rearranged modifications.
    """
    # Use regular expression to find all occurrences of the modification pattern
    matches = re.finditer(r'([A-Z])\(([-+]?\d+\.\d+)\)', peptide_sequence)

    # Iterate through matches and replace the original format with the rearranged format
    for match in matches:
        amino_acid = match.group(1)
        numeric_part = match.group(2)
        original_format = match.group(0)
        rearranged_format = f'{numeric_part}{amino_acid}'

        # If the match is at the beginning of the sequence, replace directly
        if peptide_sequence.startswith(original_format):
            peptide_sequence = peptide_sequence.replace(original_format, rearranged_format, 1)
        else:
            # Replace the original format with the rearranged format in the peptide_sequence
            peptide_sequence = peptide_sequence.replace(original_format, rearranged_format, 1)

    return peptide_sequence

def modify_peptide_sequences(column: Series) -> Series:
    """
    Modify peptide sequences in the specified column of a DataFrame.

    This function performs the following modifications to fit casanovo's tokens:
    1. Changes specific amino acid modifications.
    2. Rearranges N-terminal modifications.
    3. Changes specific N-terminal modifications.

    Args:
        column (Series): The column containing peptide sequences.

    Returns:
        Series: The modified column with peptide sequences.
    """

    # change all Amino acid modifications:
    aa_changes = [
        (r'\(\+57\.02\)', '+57.021'),
        (r'\(\+\.98\)', '+0.984'),
        (r'\(\+15\.99\)', '+15.995'),
    ]

    # change all N-term modifications:
    N_changes = [
        (r'\+42\.01', '+42.011'),
        (r'\+43\.01', '+43.006'),
        (r'\-17\.03', '-17.027'),
        (r'\+43\.01\(\-17\.03\)', '+43.006-17.027')
    ]

    modified_column = column.copy()
    
    for pattern, replacement in aa_changes:
        modified_column = modified_column.apply(lambda x: re.sub(pattern, replacement, x))

    # rearrange N-term modifications A(+42.01) -> +42.01A
    modified_column = modified_column.apply(lambda x: rearrange_modifications(x))

    for pattern, replacement in N_changes:
        modified_column = modified_column.apply(lambda x: re.sub(pattern, replacement, x))

    
    return modified_column




########## CALCULATE METRICS OF MATCHED AND UN-MATCHED CASANOVO PREDICTIONS


##########
######### ANALYSIS OF MATCHED PREDICTIONS
#########


def save_plot_with_path(plotname, plot_path=None):
    if plot_path:
        # Save the plot with the provided path
        plt.savefig(f"{plot_path}/{plotname}")
        print(f'Plot saved as {plotname} at {plot_path}')
    else:
        # Save the plot with the default filename "plot.png"
        plt.savefig(f"{plotname}")
        print(f'Plot saved as {plotname}')

def calculate_metrics(
    df: pd.DataFrame,
    aadict: Dict[str, float],
    cum_mass_threshold: float = np.inf
) -> Dict[str, Any]:
    """
    Calculate various metrics based on the dataframe containing predictions and reference sequences.

    Args:
        df (pd.DataFrame): DataFrame containing prediction sequences and reference sequences.
        aadict (Dict[str, float]): Dictionary containing amino acid masses.
        cum_mass_threshold (float, optional): Cumulative mass threshold. Defaults to np.inf.

    Returns:
        Dict[str, Any]: Dictionary containing calculated metrics.
    """
    casanovo_pred = df['casanovo_seq'].tolist()
    database_pred = df['seq'].tolist()
    aa_score_list = df['casanovo_aa_scores'].tolist()
    
    # Use casanovo's evaluation functions to get boolean lists and metrics:
    aa_bool = aa_match_batch(
                database_pred, casanovo_pred, aadict, aa_score_list, cum_mass_threshold
            )
    
    aa_precision, aa_recall, pep_precision = aa_match_metrics(*aa_bool)

    casanovo_pred = df['casanovo_seq']
    
    # Initialize counters
    n_total_aa = aa_bool[2]
    n_total_correct_aa = 0
    n_total_wrong_aa = 0
    
    n_total_peptide = len(aa_bool[0])
    n_total_correct_peptide = 0
    n_total_wrong_peptide = 0
    
    # Iterate through aa_bool[0]
    for array in aa_bool[0]:
        
        # Update total number of correct amino acids
        n_total_correct_aa += sum(array[0])

        if all(array[0]):
            # Update total number of correct peptides
            n_total_correct_peptide += 1
        else:
            # Update total number of wrong peptides
            n_total_wrong_peptide += 1

    # Update total number of wrong amino acids
    n_total_wrong_aa += n_total_aa - n_total_correct_aa

    
    return {
        'aa_precision of total pred. AA in %': round(aa_precision*100,2),
        'aa_recall of total true AA in %': round(aa_recall*100,2),
        'pep_precision in %': round(pep_precision*100,2),
        'n_total_aa': n_total_aa,
        'n_total_correct_aa': n_total_correct_aa,
        'correct_aa in %': round(n_total_correct_aa*100/n_total_aa,2),
        'n_total_wrong_aa': n_total_wrong_aa,
        'wrong_aa in %': round(n_total_wrong_aa*100/n_total_aa,2),
        'n_total_peptide': n_total_peptide,
        'n_total_correct_peptide': n_total_correct_peptide,
        'correct_peptide in %': round(n_total_correct_peptide*100/n_total_peptide,2),
        'n_total_wrong_peptide': n_total_wrong_peptide,
        'wrong_peptide in %': round(n_total_wrong_peptide*100/n_total_peptide,2),
    }

def plot_matched_metrics(
    df: DataFrame,
    matched_df: DataFrame,
    matched_metrics: dict,
    file_name:str,
    output: str
) -> None:
    """
    Plot matched metrics including overall statistics, peptide precision, and amino acid precision.

    Args:
        df (DataFrame): The DataFrame containing peptide sequences and predictions.
        matched_df (DataFrame): The DataFrame containing matched peptides.
        matched_metrics (dict): Dictionary containing matched metrics.

    Returns:
        None
    """

    # Retrieve data:
    predicted_matched = matched_df.shape[0]
    predicted_not_matched = df[(df['seq'] == '') & (df['casanovo_seq'] != '')].shape[0]
    not_predicted = df[(df['seq'] == '') & (df['casanovo_seq'] == '')].shape[0]

    rotation_angle = -40
    while True:
        # Build plot:
        fig = plt.figure(figsize=(12,5))
        left, bottom, width, height = 0.1, 0.1, 0.6, 0.6
        
        x_position = -0.1 # Fraction of figure width
        y_position = 0.15  # Fraction of figure height
        width = 0.8       # Fraction of figure width
        height = 0.6      # Fraction of figure height
        
        ax1 = fig.add_axes([x_position, y_position, width, height])
        
        size_scale = 0.5
        
        ax2 = fig.add_axes([0.45, 0.40, size_scale*width, size_scale*height])
        
        ax3 = fig.add_axes([0.59, 0.40, size_scale*width, size_scale*height])
        
        # Plot pie chart for the metrics
        overall_ratios = [predicted_matched, predicted_not_matched, not_predicted]
        labels = [f'Predicted & Matched\n({predicted_matched})',
                  f'Predicted & Not Matched\n({predicted_not_matched})',
                  f'Not Predicted\n({not_predicted})']
        explode = (0.1, 0, 0)
        angle = -40
        colors = ['cornflowerblue', 'lemonchiffon', 'dimgray']
        wedges, texts, _ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=rotation_angle, colors=colors, labels=labels, explode=explode, shadow = True, radius= 1.3)
        # radius=1.8
        ax1.set_title('Casanovo Peptide Annotation of HeLa Data (151503 spectra)', y=1.1) # y=1.3
        
        
        x,y = texts[2].get_position()
        texts[2].set_position((x - 0.3, y- 0.2))
        
        # Bar chart 1 for Peptide Precision
        pep_ratios = [matched_metrics['pep_precision in %']/100, 1-matched_metrics['pep_precision in %']/100]
        pep_labels = [f'Matched ({matched_metrics["n_total_correct_peptide"]})', f'Not matched ({matched_metrics["n_total_wrong_peptide"]})']
        bottom = 1
        width = .2
        pep_bar_colors = ['#60B5FE', '#FE6969']
        for j, (height, label, color) in enumerate(reversed([*zip(pep_ratios, pep_labels, pep_bar_colors)])):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom, color=color, label=label, alpha=0.5)
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
        
        ax2.set_title('Peptide Precision')
        ax2.legend(loc='best', bbox_to_anchor=(0.2, -0.4, 0.5, 0.5))
        ax2.axis('off')
        ax2.set_xlim(- 3.5 * width, 3.5 * width)
        
        # Bar chart 2 for AA Precision
        aa_ratios = [matched_metrics['aa_precision of total pred. AA in %']/100, 1-matched_metrics['aa_precision of total pred. AA in %']/100]
        aa_labels = [f"Matched ({matched_metrics['n_total_correct_aa']})", f"Not matched ({matched_metrics['n_total_aa']})"]
        bottom = 1
        width = .2
        aa_bar_colors = ['#60FE70', '#FE6969']
        for j, (height, label, color) in enumerate(reversed([*zip(aa_ratios, aa_labels, aa_bar_colors)])):
            bottom -= height
            bc = ax3.bar(0, height, width, bottom=bottom, color=color, label=label, alpha=0.5)
            ax3.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
        
        ax3.set_title('Aminoacid Precision')
        ax3.legend(loc='best', bbox_to_anchor=(0.3, -0.4, 0.5, 0.5))
        ax3.axis('off')
        ax3.set_xlim(- 3.5 * width, 3.5 * width)
        
        # Set up connections between the plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(pep_ratios)
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        con.set_linewidth(1)
        ax2.add_artist(con)
        
        # Draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(1)
        
        save_plot_with_path(f'{file_name}_matched_metrics_plot.png', output)
        plt.show()

        user_input = input("Enter rotation angle (or 'exit' to quit): ")
    
        if user_input.lower() == 'exit':
            break  
    
        # Update rotation angle with user input
        try:
            rotation_angle = float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid rotation angle.")

######### metrics vs modified aminoacids

def calculate_modified_metrics(matched_df: DataFrame, aadict: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate metrics for modified and non-modified peptides.

    Args:
        matched_df (DataFrame): DataFrame containing matched peptides.
        aadict (Dict[str, float]): Dictionary containing amino acid masses.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries of metrics for modified and non-modified peptides respectively.
    """
    mod_df = matched_df[(matched_df['db_modified'] == True) | (matched_df['cs_modified'] == True)]
    non_mod_df = matched_df[(matched_df['db_modified'] != True) & (matched_df['cs_modified'] != True)]
    mod_metrics = calculate_metrics(mod_df, aadict)
    non_mod_metrics = calculate_metrics(non_mod_df, aadict)

    return mod_metrics, non_mod_metrics

def plot_modified_metrics(
    matched_metrics: Dict[str, Any],
    mod_metrics: Dict[str, Any],
    non_mod_metrics: Dict[str, Any],
    file_name: str,
    output: str
) -> None:
    """
    Plot modified metrics including peptide prediction accuracy and amino acid prediction accuracy.

    Args:
        matched_metrics (Dict[str, Any]): Metrics for matched peptides.
        mod_metrics (Dict[str, Any]): Metrics for modified peptides.
        non_mod_metrics (Dict[str, Any]): Metrics for non-modified peptides.
    """
    peptides_total: int = matched_metrics['n_total_aa']
    peptides_unmodified: int = peptides_total - mod_metrics['n_total_peptide']
    peptides_modified: int = mod_metrics['n_total_peptide']
    peptides_modified_correct: int = mod_metrics['n_total_correct_peptide']
    peptides_unmodified_correct: int = non_mod_metrics['n_total_correct_peptide']
    peptides_modified_wrong: int = mod_metrics['n_total_wrong_peptide']
    peptides_unmodified_wrong: int = non_mod_metrics['n_total_wrong_peptide']

    amino_acids_total: int = matched_metrics['n_total_aa']
    amino_acids_of_modified_peptides: int = mod_metrics['n_total_aa']
    amino_acids_of_unmodified_peptides: int = non_mod_metrics['n_total_aa']
    amino_acids_of_modified_correct: int = mod_metrics['n_total_correct_aa']
    amino_acids_of_unmodified_correct: int = non_mod_metrics['n_total_correct_aa']
    amino_acids_of_modified_wrong: int = mod_metrics['n_total_wrong_aa']
    amino_acids_of_unmodified_wrong: int = non_mod_metrics['n_total_wrong_aa']

    # Plotting pie chart for peptides
    labels_peptides: List[str] = [f'Unmodified (Correct)\n({peptides_unmodified_correct})',
                                  f'Unmodified (Wrong)\n({peptides_unmodified_wrong})',
                                  f'Modified (Wrong)\n({peptides_modified_wrong})',
                                  f'Modified (Correct)\n({peptides_modified_correct})']
    sizes_peptides: List[int] = [peptides_unmodified_correct, peptides_unmodified_wrong, peptides_modified_wrong, peptides_modified_correct]
    colors_peptides: List[str] = ['#97E0F4', '#6FA4B3', '#AD68A4', '#F6AAFF']
    explode_peptides: Tuple[float, ...] = (0, 0, 0.1, 0.1)  # explode the modified (correct) section
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pie(sizes_peptides, explode=explode_peptides, labels=labels_peptides, colors=colors_peptides, autopct='%1.1f%%', shadow=True, startangle=140, labeldistance=1.15)
    plt.title(f'Peptides Prediction (in total: {peptides_total})', y=1.10)

    # Plotting pie chart for amino acids
    labels_aa: List[str] = [f'Unmodified (Correct)\n({amino_acids_of_unmodified_correct})',
                            f'Unmodified (Wrong)\n({amino_acids_of_unmodified_wrong})',
                            f'Modified (Wrong)\n({amino_acids_of_modified_wrong})',
                            f'Modified (Correct)\n({amino_acids_of_modified_correct})']
    sizes_aa: List[int] = [amino_acids_of_unmodified_correct, amino_acids_of_unmodified_wrong, amino_acids_of_modified_wrong, amino_acids_of_modified_correct]
    colors_aa: List[str] = ['#FEB45D', '#B9864A', '#C15151', '#FE6868']
    explode_aa: Tuple[float, ...] = (0, 0, 0.1, 0.1)  # explode the incorrect slice
    plt.subplot(1, 2, 2)
    plt.pie(sizes_aa, explode=explode_aa, labels=labels_aa, colors=colors_aa, autopct='%1.1f%%', shadow=True, startangle=140, labeldistance=1.2)
    plt.title(f'Amino Acids Prediction (in total: {amino_acids_total})', y=1.10)
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    
    save_plot_with_path(f'{file_name}_modified_metrics.png', output)
    plt.show()


######### metrics vs aminoacid score

def aa_metrics_at_score_threshold(
    df: pd.DataFrame,
    aadict: Dict[str, float],
    cum_mass_threshold: float,
    score_threshold: float
) -> Tuple[float, float]:
    """
    Calculate amino acid precision and recall at a given score threshold.

    Args:
        df (pd.DataFrame): DataFrame containing prediction data.
        aadict (Dict[str, float]): Dictionary containing amino acid masses.
        cum_mass_threshold (float): Cumulative mass threshold for matching amino acids.
        score_threshold (float): Threshold score for considering an amino acid match.

    Returns:
        Tuple[float, float]: A tuple containing amino acid precision and recall.
    """
    
    casanovo_pred = df['casanovo_seq']
    database_pred = df['seq']
    aa_score_list = df['casanovo_aa_scores'].tolist()

    aa_bool = aa_match_batch(database_pred, casanovo_pred, aadict, aa_score_list, cum_mass_threshold=np.inf)
    n_total_aa = aa_bool[1]
    
    aa_scores_correct = []
    aa_scores_all = []

    for array in aa_bool[0]:
        aa_scores_all.append(array[2])
        indices = np.argwhere(array[0]).flatten()
        for index in indices:
            aa_scores_correct.append(array[2][index])

    aa_scores_all = list(chain.from_iterable(aa_scores_all))

    aa_precision, aa_recall = aa_precision_recall(aa_scores_correct, aa_scores_all, n_total_aa, score_threshold)

    return aa_precision, aa_recall

def plot_precision_recall(
    df: pd.DataFrame,
    aadict: Dict[str, float],
    cum_mass_threshold: float,
    score_threshold_step: int = 10,
    file_name: str = '',
    output: str = None
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Plot precision and recall values against different score thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing prediction data.
        aadict (Dict[str, float]): Dictionary containing amino acid masses.
        cum_mass_threshold (float): Cumulative mass threshold for matching amino acids.
        score_threshold_step (int): Number of steps to divide the score threshold range.

    Returns:
        Tuple[np.ndarray, List[float], List[float]]: A tuple containing the score thresholds, precisions, and recalls.
    """
    precisions = []
    recalls = []

    score_thresholds = np.linspace(0, 0.99, score_threshold_step)

    with tqdm(total=len(score_thresholds), desc="Analyzing data with score thresholds [0-1]", unit="iteration") as progress_bar:
        for threshold in score_thresholds:
            aa_precision, aa_recall = aa_metrics_at_score_threshold(df, aadict, np.Inf, threshold)
            precisions.append(aa_precision)
            recalls.append(aa_recall)
            progress_bar.update(1)
        progress_bar.set_postfix_str("Done!")

    score_pr_df = pd.DataFrame({'score thresholds': score_thresholds, 'precisions': precisions, 'recalls': recalls})
    print(score_pr_df)
    
    plt.plot(score_thresholds, precisions, label='Precision')
    plt.plot(score_thresholds, recalls, label='Recall')
    plt.xlabel('Score Threshold')
    plt.ylabel('Value')
    plt.title('Precision and Recall vs. Score Threshold')
    plt.legend()
    
    save_plot_with_path(f'{file_name}_precision_recall_plot.png', output)
    plt.show()

    
    return score_pr_df

def plot_aa_scores_horizontal(
    df: pd.DataFrame,
    aadict: Dict[str, float],
    cum_mass_threshold: float = np.inf,
    file_name: str = '',
    output: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot a horizontal histogram of AA scores for true and false matches.

    Args:
        df (pd.DataFrame): DataFrame containing prediction data.
        aadict (Dict[str, float]): Dictionary containing amino acid masses.
        cum_mass_threshold (float): Cumulative mass threshold for matching amino acids.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of true counts and false counts.
    """
    # Extract relevant data from the DataFrame
    casanovo_pred = df['casanovo_seq']
    database_pred = df['seq']
    aa_score_list = df['casanovo_aa_scores'].tolist()

    # Perform AA matching batch
    aa_bool = aa_match_batch(database_pred, casanovo_pred, aadict, aa_score_list, cum_mass_threshold)

    # Extract match boolean and AA scores
    match_boolean_all = [array[0] for array in aa_bool[0]]
    aa_scores_all = [array[2] for array in aa_bool[0]]

    # Flatten lists
    aa_scores_all = list(chain.from_iterable(aa_scores_all))
    match_boolean_all = list(chain.from_iterable(match_boolean_all))

    # Convert lists to arrays
    aa_scores = np.array(aa_scores_all)
    match_boolean = np.array(match_boolean_all)

    # Define bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Calculate histograms for True and False counts
    true_counts, _ = np.histogram(aa_scores[match_boolean], bins=bins)
    false_counts, _ = np.histogram(aa_scores[~match_boolean], bins=bins)

    # Bin labels for df
    bin_labels = ['0 - 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', '0.9 - 1']
    aa_horiz_df = pd.DataFrame({'true_counts': true_counts, 'false_counts': false_counts}, index=bin_labels)
    print(aa_horiz_df)
    
    # Calculate bar heights
    bar_height = 0.2

    # Calculate bar positions
    y_positions = np.arange(len(bins[:-1]))

    plt.figure(figsize=(10, 6))
    # Plot histograms using Matplotlib
    plt.barh(y_positions - bar_height/2, true_counts, height=bar_height, color='#2BDD1F', alpha=0.6, label='True Counts')
    plt.barh(y_positions + bar_height/2, false_counts, height=bar_height, color='#FE6969', alpha=0.4, label='False Counts')

    # Add group labels
    tick_labels = [f'{bins[i]:.1f} - {bins[i+1]:.1f}' for i in range(len(bins)-1)]
    plt.yticks(y_positions, tick_labels)
    
    plt.xlabel('Counts')
    plt.ylabel('AA Scores')
    plt.title('Histogram of AA Scores')
    plt.xlim(0, max(max(true_counts), max(false_counts))+10)
    plt.legend()

    save_plot_with_path(f'{file_name}_aa_scores_histogram.png', output)
    plt.show()

    return aa_horiz_df


#############
############# ANALYSIS OF UNMATCHED DATA
############


def plot_dark_scores_horizontal(
    df: pd.DataFrame,
    aadict: Dict[str, float],
    cum_mass_threshold: float = np.inf,
    file_name: str = '',
    output: str = None
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Plot horizontal histograms of dark scores.

    Args:
        df (pd.DataFrame): DataFrame containing prediction data.
        aadict (Dict[str, float]): Dictionary containing amino acid masses.
        cum_mass_threshold (float): Cumulative mass threshold for matching amino acids.

    Returns:
        Tuple[List[float], np.ndarray, np.ndarray]: A tuple containing lists of mean scores of peptides, counts of peptides, and counts of amino acids.
    """
    # Extract relevant data from the DataFrame
    casanovo_pred = df['casanovo_seq']
    aa_score_list = df['casanovo_aa_scores'].tolist()

    aa_scores_all = list(chain.from_iterable(aa_score_list))
    # Calculate mean scores of peptides
    mean_scores_of_peptides = df['mean_aa_score'].tolist()
    
    # Define bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Calculate histograms for mean scores of peptides
    peptide_counts, _ = np.histogram(mean_scores_of_peptides, bins=bins)
    aa_counts, _ = np.histogram(aa_scores_all, bins=bins)

    bin_labels = ['0 - 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', '0.9 - 1']
    dark_hist_df = pd.DataFrame({'peptide_counts':peptide_counts, 'aa_counts':aa_counts}, index=bin_labels)
    print(dark_hist_df)
    
    # Calculate bar heights
    bar_height = 0.2

    # Calculate bar positions
    y_positions = np.arange(len(bins[:-1]))

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Plot histograms using Matplotlib
    axs[0].barh(y_positions, peptide_counts, height=bar_height, color='#1F90DD', alpha=0.8, label='Peptide Counts')
    axs[1].barh(y_positions, aa_counts, height=bar_height, color='#1FDD31', alpha=0.8, label='Amino Acid Counts')

    # Add group labels
    tick_labels = [f'{bins[i]:.1f} - {bins[i+1]:.1f}' for i in range(len(bins)-1)]
    axs[0].set_yticks(y_positions)
    axs[0].set_yticklabels(tick_labels)
    axs[1].set_yticks(y_positions)
    axs[1].set_yticklabels(tick_labels)
    
    axs[0].set_xlabel('Peptide Counts')
    axs[1].set_xlabel('Amino Acid Counts')
    axs[0].set_ylabel('Mean AA Scores')
    axs[1].set_ylabel('Individual AA Scores')
    axs[0].set_title('Dark Peptide Counts vs Mean Amino Acid Scores')
    axs[1].set_title('Dark Aminoacid Counts vs Amino Acid Scores')
    axs[0].set_xlim(0, max(peptide_counts) + 10)
    axs[1].set_xlim(0, max(aa_counts) + 10)

    plt.tight_layout()
    
    plt.savefig(f'{file_name}_dark_scores_histogram.png') 

    save_plot_with_path(f'{file_name}_dark_scores_histogram.png', output)
    plt.show()

    return mean_scores_of_peptides, peptide_counts, aa_counts

def plot_dark_metrics(df, matched_df, casanovo_unmatched_df, high_mean_aa_score_peptides, modified_bool,file_name, output):
    # Define your data
    predicted_matched = matched_df.shape[0]
    predicted_not_matched = df[(df['seq'] == '') & (df['casanovo_seq'] != '')].shape[0]
    not_predicted = df[(df['seq'] == '') & (df['casanovo_seq'] == '')].shape[0]
    total_dark_annotation = casanovo_unmatched_df.shape[0]
    high_score_dark_annotation = high_mean_aa_score_peptides.shape[0]
    low_score_dark_annotation = total_dark_annotation - high_mean_aa_score_peptides.shape[0]

    duplicate_sequences = high_mean_aa_score_peptides[high_mean_aa_score_peptides['casanovo_seq'].duplicated(keep=False)]
    duplicate_data = duplicate_sequences['casanovo_seq'].value_counts()
    unique_counts = len(high_mean_aa_score_peptides['casanovo_seq'].unique())
    duplicate_counts = high_score_dark_annotation - unique_counts
    duplicate_ratio = 1 - (unique_counts / high_score_dark_annotation)

    dark_metrics = {'total_predicted_not_matched':int(total_dark_annotation), 
                    'high_mean_score_annotations':int(high_score_dark_annotation), 'duplicates': int(duplicate_counts),
                            'duplicate_ratio': round(duplicate_ratio,2), 'unique_ratio': round(1-duplicate_ratio,2)}
    
    dark_df = pd.DataFrame.from_dict({'metrics_of_unmatched_data':dark_metrics}).reindex(index=['total_predicted_not_matched',
                                                                                             'high_mean_score_annotations',
                                                                                                'duplicates',
                                                                                             'duplicate_ratio', 
                                                                                             'unique_ratio'])
    
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    #fig.subplots_adjust(wspace=0)
    rotation_angle = 40
    while True:
        fig = plt.figure(figsize=(12,5))
        left, bottom, width, height = 0.1, 0.1, 0.6, 0.6
        
        x_position = -0.1 # Fraction of figure width
        y_position = 0.15  # Fraction of figure height
        width = 0.8       # Fraction of figure width
        height = 0.6      # Fraction of figure height
        
        ax1 = fig.add_axes([x_position, y_position, width, height])
        
    
        # Define data for pie chart
        overall_ratios = [predicted_matched, not_predicted, low_score_dark_annotation,  high_score_dark_annotation]
        labels = [f'Predicted & Matched\n({predicted_matched})',
                  f'Not Predicted\n({not_predicted})', 
                    f'Predicted \n & Not Matched \n & Mean AA Score < 90% \n({low_score_dark_annotation})',
                    f'Predicted \n & Not Matched \n & Mean AA Score > 90% \n({high_score_dark_annotation})',
              ]
        explode = (0.1, 0, 0, 0.1) 
        colors = ['cornflowerblue', 'dimgray',  'darkkhaki', 'lemonchiffon']
        wedges, texts, _ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=rotation_angle, colors=colors, labels=labels, explode=explode, radius=1.3, shadow = True)
        ax1.set_title('Casanovo Peptide Annotation of HeLa Data (151503 spectra)', y=1.2)
    
        x,y = texts[2].get_position()
        texts[2].set_position((x - 0.3, y- 0.2))
    
        if modified_bool:
    
            modified_sequences = high_mean_aa_score_peptides[high_mean_aa_score_peptides['cs_modified'] == True]
            modified_count = modified_sequences.shape[0]
            modified_ratio = modified_count/high_score_dark_annotation
            non_modified_sequences = high_mean_aa_score_peptides[high_mean_aa_score_peptides['cs_modified'] == False]
            non_modified_count = non_modified_sequences.shape[0]
            non_modified_ratio = non_modified_count/high_score_dark_annotation
    
            dark_metrics = {'total_predicted_not_matched':int(total_dark_annotation), 
                        'high_mean_score_annotations':int(high_score_dark_annotation), 'duplicates': int(duplicate_counts),
                                'duplicate_ratio': round(duplicate_ratio,2), 'unique_ratio': round(1-duplicate_ratio,2),
                           'modified': int(modified_count), 'modified_ratio':round(modified_ratio,2), 'non_modified_ratio': round(non_modified_ratio,2)}
        
            dark_df = pd.DataFrame.from_dict({'metrics_of_unmatched_data':dark_metrics}).reindex(index=['total_predicted_not_matched',
                                                                                                     'high_mean_score_annotations',
                                                                                                        'duplicates',
                                                                                                     'duplicate_ratio', 
                                                                                                     'unique_ratio', 'modified', 'modified_ratio',
                                                                                                       'non_modified_ratio'])
    
            size_scale = 0.5
        
            ax2 = fig.add_axes([0.45, 0.40, size_scale*width, size_scale*height])
            ax3 = fig.add_axes([0.59, 0.40, size_scale*width, size_scale*height])
            
            bbox_to_anchor_1 = (0.2, -0.4, 0.5, 0.5)
            bbox_to_anchor_2 = (0.4, -0.4, 0.5, 0.5)
    
          # Bar chart 1 for Duplication analysis
            pep_ratios = [1-duplicate_ratio, duplicate_ratio]
            pep_labels = [f'Unique peptides ({unique_counts})', f'Duplicate peptides ({duplicate_counts})']
            bottom = 1
            width = .2
            pep_bar_colors = ['#1560bd', 'lavender']
            for j, (height, label, color) in enumerate(reversed([*zip(pep_ratios, pep_labels, pep_bar_colors)])):
                bottom -= height
                bc = ax2.bar(0, height, width, bottom=bottom, color=color, label=label, alpha=0.5)
                ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
            
            ax2.set_title('Duplication Ratio')
            ax2.legend(loc='best', bbox_to_anchor=bbox_to_anchor_1)
            ax2.axis('off')
            ax2.set_xlim(- 3.5 * width, 3.5 * width)
        
                # Bar chart 2 for Modification analysis
            aa_ratios = [non_modified_ratio, modified_ratio,]
            aa_labels = [f"Not modified peptides ({non_modified_count})", f"Modified peptides ({modified_count})"]
            bottom = 1
            width = .2
            aa_bar_colors = ['#60FE70', 'rebeccapurple']
            for j, (height, label, color) in enumerate(reversed([*zip(aa_ratios, aa_labels, aa_bar_colors)])):
                bottom -= height
                bc = ax3.bar(0, height, width, bottom=bottom, color=color, label=label, alpha=0.5)
                ax3.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
            
            ax3.set_title('Modification Ratio')
            ax3.legend(loc='best', bbox_to_anchor=bbox_to_anchor_2)
            ax3.axis('off')
            ax3.set_xlim(- 3.5 * width, 3.5 * width)
        
    
        else:
            size_scale = 1
        
            ax2 = fig.add_axes([0.5, 0.4, 0.6, size_scale*height])
    
            bbox_to_anchor_1 =(0.15, -0.4, 0.5, 0.5)
    
            # Bar chart 1 for Duplication analysis
            pep_ratios = [1-duplicate_ratio, duplicate_ratio]
            pep_labels = [f'Unique ({unique_counts})', f'Duplicates ({duplicate_counts})']
            bottom = 1
            width = .2
            pep_bar_colors = ['#1560bd', 'lavender']
            for j, (height, label, color) in enumerate(reversed([*zip(pep_ratios, pep_labels, pep_bar_colors)])):
                bottom -= height
                bc = ax2.bar(0, height, width, bottom=bottom, color=color, label=label, alpha=0.5)
                ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
            
            ax2.set_title('Duplication Ratio')
            ax2.legend(bbox_to_anchor = bbox_to_anchor_1, loc='best')
            ax2.axis('off')
            ax2.set_xlim(- 3.5 * width, 3.5 * width)
    
        print(dark_df)
        connected_section = 3
        # Set up connections between the plots
        theta1, theta2 = wedges[connected_section].theta1, wedges[connected_section].theta2
        center, r = wedges[connected_section].center, wedges[0].r
        bar_height = sum(pep_ratios)
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        con.set_linewidth(1)
        ax2.add_artist(con)
        
        # Draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(1)
        
        save_plot_with_path(f'{file_name}_dark_metrics_plot.png', output)
        plt.show()
        
        user_input = input("Enter rotation angle (or 'exit' to quit): ")
    
        if user_input.lower() == 'exit':
            break  
    
        # Update rotation angle with user input
        try:
            rotation_angle = float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid rotation angle.")
    

    return dark_df

    


