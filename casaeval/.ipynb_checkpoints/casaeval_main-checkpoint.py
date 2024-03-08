"""The command line entry point for casaeval."""
import datetime
import functools
import logging
import os
import re
import shutil
import sys
import warnings
from typing import Optional, Tuple
from pathlib import Path 

warnings.filterwarnings("ignore", category=DeprecationWarning)

import appdirs
import rich_click as click

import timsrust_pyo3
import pyteomics
from pyteomics import mgf

import fileinput
import json
import re
import itertools
import collections

import numpy as np
import pandas as pd
import matplotlib


from .evaluation.masses import PeptideMass
from .processing.raw_data_conversion import converter_timsd_to_mgf # import TIMS to .mgf converter
from .processing.prediction_writer import add_predictions_to_mgf # import Writer for adding predictions
from .automation.SLURMscripter import generate_slurm_scripts # import scripter for automating casanovo runs

# Evaluation functions
import casaeval.evaluation.prediction_evaluation as predeval


logger = logging.getLogger("casaeval")
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True

class _SharedParams(click.RichCommand):
    """Options shared between most commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
            click.Option(
                ("--output", "-o"),
                help="The path, to which command output files are saved in.",
                type=click.Path(exists=False, file_okay=False, dir_okay=True),
            ), 
            click.Option(
                ("-v", "--verbosity"),
                help="""
                Set the verbosity of console logging messages. Log files are
                always set to 'debug'.
                """,
                type=click.Choice(
                    ["debug", "info", "warning", "error"],
                    case_sensitive=False,
                ),
                default="info",
            ),
        ]


@click.group(context_settings=dict(help_option_names=["-h", "-help"]))
def main():
    """
    \b
    Casaeval: Data Processing and Prediction Evaluation of Casanovo
    ================================================================================

    Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo
    mass spectrometry peptide sequencing with a transformer model. Proceedings
    of the 39th International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.

    Official code website of casanovo: https://github.com/Noble-Lab/casanovo \n
    Official code website of casaeval: https://github.com/Kactaceapengu/casaeval
    """
    pass


@main.command(cls=_SharedParams)
@click.option("-i", "-input_d", "input_file", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
def timsconvert(
    input_file: str,
    output: Optional[str],
    verbosity: Optional[str]  # Just for log data
) -> None:
    """Converts raw TIMS data [.d] to a .mgf file."""

    output = setup_logging(output, verbosity)
    file_name = os.path.basename(input_file)
    file_name_without_extension = os.path.splitext(os.path.basename(input_file))[0]
    file_extension = os.path.splitext(os.path.basename(input_file))[1]
    logger.info(f"Converting {file_name} to .mgf file.")
    converter_timsd_to_mgf(input_file, output)
    
    
    logger.info("DONE!")



@main.command(cls=_SharedParams)
@click.option(
    "-add", "--add_mode",
    help="Adds database or casanovo peptide predictions to spectra file [.mgf].",
    type=click.Choice(["database", "db", "casanovo", "cas", "both"]), required=True)
@click.option("-i", "--input_mgf", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-i_1", "--input_database", type=click.Path(exists=True, dir_okay=False), help="When adding both: Path to the database file.")
@click.option("-i_2", "--input_casanovo", type=click.Path(exists=True, dir_okay=False), help="When adding both: Path to the casanovo file.")
def predictions(
    add_mode: str, 
    input_mgf: str, 
    input_database: str, #if only one prediction data is added, input_predictions_db takes both kinds (db and cas)
    input_casanovo: Optional[str],
    output: Optional[str],
    verbosity: Optional[str] # Just for log data
) -> None:  
    """Add database and/or casanovo prediction to .mgf file. Provide database predictions as [.csv] and casanovo predictions as [.mztab]."""
    
    output = setup_logging(output, verbosity)
    if not input_database and not input_casanovo:
        click.echo("Please provide at least one prediction path.")
        return
        
    if add_mode in ["database", "db"]:
        text_db = f"Adding database predictions..."
        click.echo(text_db)
        logger.info(text_db)
    if add_mode in ["casanovo", "cas"]:
        text_cas = f"Adding casanovo predictions..."
        click.echo(text_cas)
        logger.info(text_cas)
    if add_mode == "both":
        text_both = f"Adding database predictions (1.input) and casanovo predictions (2.input)..."
        click.echo(text_both)
        logger.info(text_both)
        add_predictions_to_mgf
    
    add_predictions_to_mgf(add_mode, input_mgf, input_database, input_casanovo)
    logger.info("DONE!")



@main.command(cls=_SharedParams)
@click.option("--input_mgf", "-i", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-m", "--matched", help="Evaluate with 'matched' mode.", is_flag=True)
@click.option("-d", "--dark", "-nm", "-notmatched", help="Evaluate with 'dark' mode.", is_flag=True)
@click.option("--aadictmode", "-modification", "-mod", "-notmatched", help="Specify, if there are modifications in the predictions.",type=click.Choice(["y", "n", "modified", "not_modified", "only_aa"]), required=True) #type=click.Choice(["y", "n", "modified", "not_modified", "only_aa"])
def evaluate(
    input_mgf: str,
    matched: bool,
    dark: bool,
    aadictmode,
    output:Optional[str],
    verbosity: Optional[str] # Just for log data
):
    "With a given .mgf file containing both database (true) predictions and Casanovo predictions, the precision and amino acid prediction scores of casanovo is evaluated by plots."
    
    output = setup_logging(output, verbosity)
    file_name = os.path.basename(input_mgf)
    file_name_without_extension = os.path.splitext(os.path.basename(input_mgf))[0]
    file_extension = os.path.splitext(os.path.basename(input_mgf))[1]
    
    #### Get aadict with all tokens specified:
    if aadictmode in ["modified", "y"]:
        print(f'Predictions are evaluated with modification tokens.')
        
        aadict = PeptideMass(residues='massivekb').masses
        modified_bool = True
    elif aadictmode in ["not_modified", "only_aa", "n"]:
        print(f'Predictions are evaluated with no modification tokens.')
        aadict = PeptideMass(residues='canonical').masses
        modified_bool = False
    else:
        print("Please specify if the amino acids in the predictions are modified or not.")

    # Function to handle evaluation of matched prediction steps
    def evaluate_matched(file_name_without_extension, output):

        # Evaluate matched predictions:
        matched_metrics = predeval.calculate_metrics(matched_df, aadict)

        eval_df = pd.DataFrame.from_dict(
            {'all matched predictions': matched_metrics},
                 orient='columns'
                    ).reindex(index=['aa_precision of total pred. AA in %', 'aa_recall of total true AA in %', 'pep_precision in %', 'n_total_aa',
                                     'n_total_correct_aa', 'correct_aa in %', 'n_total_wrong_aa', 'wrong_aa in %', 'n_total_peptide', 'n_total_correct_peptide',
                                     'correct_peptide in %', 'n_total_wrong_peptide', 'wrong_peptide in %'])
        print(eval_df)
        predeval.plot_matched_metrics(df, matched_df, matched_metrics, file_name_without_extension, output)
        
        eval_df.to_csv(f"{file_name_without_extension}_matched_metrics.csv", index=False)
        print(f'Matched metric data were saved as {file_name_without_extension}_matched_metrics.csv at {output}')
        
        if modified_bool: # Check if modified predictions are specified
            proceed_mod_evaluation = input('Continue with modification evaluation? (y/n/a): ')
            if proceed_mod_evaluation.lower() == 'a':
                return  # Abort the whole process
        
            if proceed_mod_evaluation.lower() == 'y':
                # Evaluate modifications of the matched predictions
                mod_metrics, non_mod_metrics = predeval.calculate_modified_metrics(matched_df, aadict)
                mod_eval_df = pd.DataFrame.from_dict({'all matched predictions':matched_metrics,
                                                      'subset with modifications': mod_metrics, 
                                                      'subset without modifications': non_mod_metrics}, orient='columns').reindex(index=['aa_precision of total pred. AA in %',
                                                                                                                   'aa_recall of total true AA in %',
                                                                                                                   'pep_precision in %', 'n_total_aa',
                                                                                                                   'n_total_correct_aa', 'correct_aa in %',
                                                                                                                   'n_total_wrong_aa', 'wrong_aa in %',
                                                                                                                   'n_total_peptide', 'n_total_correct_peptide', 
                                                                                                                   'correct_peptide in %',
                                                                                                                   'n_total_wrong_peptide', 'wrong_peptide in %'])

                print(mod_eval_df)
                predeval.plot_modified_metrics(matched_metrics, mod_metrics, non_mod_metrics, file_name_without_extension, output)
                mod_eval_df.to_csv(f"{file_name_without_extension}_matched_metrics.csv", index=False)
        
        # Continue with amino acid precision and recall evaluation with different prediction score thresholds?
        proceed_aa_evaluation = input('Continue with amino acid precision and recall evaluation with different prediction score thresholds? (y/n/a): ')
        if proceed_aa_evaluation.lower() == 'a':
            return  # Abort the whole process
    
        if proceed_aa_evaluation.lower() == 'y':
            # Evaluate amino acid score distribution, precision, recall of matched predictions
            pred_score_precision_recall_data = predeval.plot_precision_recall(matched_df, aadict, np.Inf, score_threshold_step=10,
                                                                              file_name=file_name_without_extension, output=output)
    
        # Continue with correctly and incorrectly amino acid match count evaluation with different prediction score thresholds?
        proceed_count_evaluation = input('Continue with correctly and incorrectly amino acid match count evaluation with different prediction score thresholds? (y/n/a): ')
        if proceed_count_evaluation.lower() == 'a':
            return  # Abort the whole process
    
        if proceed_count_evaluation.lower() == 'y':
            # Evaluate correctly and incorrectly amino acid match count with different prediction score thresholds
            boolean_pred_score_distribution_data = predeval.plot_aa_scores_horizontal(matched_df, aadict, cum_mass_threshold=np.inf,
                                                                                      file_name=file_name_without_extension, output=output)

    def evaluate_dark(casanovo_unmatched_df):
        # Evaluate aminoacid score and peptide vs aa score distribution of the 'dark peptides':
        mean_scores_of_peptides, peptide_counts, aa_counts = predeval.plot_dark_scores_horizontal(casanovo_unmatched_df, aadict, cum_mass_threshold=np.inf,
                                                                                                      file_name=file_name_without_extension, output=output)
        
        
        dark_pred_score_threshold = float(input('Specify the score threshold to filter casanovo annotations of dark spectra [float, 0 - 1]. Default at 0.9: ') or 0.9)
    
        # Filter casanovo annotations with mean_aa_score >= 0.9
        high_mean_aa_score_peptides = casanovo_unmatched_df[casanovo_unmatched_df['mean_aa_score'] >= dark_pred_score_threshold]
    
        predeval.plot_dark_metrics(df, matched_df, casanovo_unmatched_df, high_mean_aa_score_peptides,
                                   modified_bool, file_name=file_name_without_extension, output=output)
    
        high_mean_aa_score_peptides_sorted = high_mean_aa_score_peptides.sort_values(by='mean_aa_score', ascending=False)
        high_mean_aa_score_peptides_sorted.rename(columns={'title': 'Precursor Id'}, inplace=True)
        
        # Get the integer-based index of the columns
        precursor_index = high_mean_aa_score_peptides_sorted.columns.get_loc('Precursor Id')
        casanovo_seq_index = high_mean_aa_score_peptides_sorted.columns.get_loc('casanovo_seq')
        mean_aa_score_index = high_mean_aa_score_peptides_sorted.columns.get_loc('mean_aa_score')
        
        # Select the first 20 rows and specific columns using integer-based indexers
        high_mean_aa_score_peptides_subset = high_mean_aa_score_peptides_sorted.iloc[0:20, [precursor_index, casanovo_seq_index, mean_aa_score_index]]
        print(high_mean_aa_score_peptides_subset)

        print(f'Saving dark annotations with mean prediction score greater or equal {dark_pred_score_threshold}...')
        high_mean_aa_score_peptides_sorted.to_csv(f"{file_name_without_extension}_casanovo_annotations_prediction_score_over_{dark_pred_score_threshold}.csv",
                                                  index=False)
        print(f'{file_name_without_extension}_casanovo_annotations_prediction_score_over_{dark_pred_score_threshold}.csv" at {output}')
        
    """Evaluates spectra file [.mgf] with 'matched' or 'dark' mode."""
    if matched and dark:
        (f"Evaluating spectra file {input_mgf} with both modes...")
        df, matched_df, casanovo_unmatched_df = predeval.mgf_to_df(input_mgf, modified_bool)  # Retrieve data
    
        evaluate_matched(file_name_without_extension, output)

        proceed_dark_evaluation = input('Continue with evaluation of prediction score distribution of unmatched (dark) spectra data? (y/n/a): ')
        if proceed_dark_evaluation.lower() == 'a':
            return # Abort the whole process

        if proceed_dark_evaluation.lower() == 'y':
            evaluate_dark(casanovo_unmatched_df)

            
            
    elif not matched and not dark:
        click.echo("Please specify either 'matched' or 'dark' mode or both.")
        return

    elif matched:
        click.echo(f"Evaluating spectra file {input_mgf} with 'matched' mode...")
        # Perform evaluation with 'matched' mode
        
        df, matched_df, _ = predeval.mgf_to_df(input_mgf, modified_bool)  # Retrieve data
        
        evaluate_matched(file_name_without_extension, output)
    
    elif dark:
        click.echo(f"Evaluating spectra file {input_mgf} with 'dark' mode...")
        # Perform evaluation with 'dark' mode
        
        df, matched_df, casanovo_unmatched_df = predeval.mgf_to_df(input_mgf, modified_bool)  # Retrieve data
        
        evaluate_dark(casanovo_unmatched_df)

@main.command(cls=_SharedParams)
@click.option("-i", "--input", "-input_folder", "-input_file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--command", "-c", help="Specify, which casanovo command will be automated.",type=click.Choice(["sequence", "train", "evaluate"]), required=True)
def automate(
    input: str,
    command: str,
    output: Optional[str],
    verbosity: Optional[str]  # Just for log data
) -> None:
    """Automates SLURM scripting and its execution for casanovo run on servers."""

    output = setup_logging(output, verbosity)
    
    logger.info(f"Generating scripts for casanovo SLURM runs.")

    if output == '':
        output = None
        
    generate_slurm_scripts(input, command, output)
    
    logger.info("DONE!")


def setup_logging(
    output: Optional[str],
    verbosity: str,
) -> Path:
    """Set up the logger.

    Logging occurs to the command-line and to the given log file.

    Parameters
    ----------
    output : Optional[str]
        The provided output file name.
    verbosity : str
        The logging level to use in the console.

    Return
    ------
    output : Path
        The output file path.
    """
    if output is None:
        output = f"casaeval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    output = Path(output).expanduser().resolve()

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

    # Formatters for file vs console:
    console_formatter = logging.Formatter("{levelname}: {message}", style="{")
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)
    file_handler = logging.FileHandler(output.with_suffix(".log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    warnings_logger.addHandler(file_handler)

    # Disable dependency non-critical log messages.
    logging.getLogger("timsrust_pyo3").setLevel(
        logging_levels[verbosity.lower()]
    )
    logging.getLogger("pyteomics").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)

    return output




if __name__ == "__main__":
    main()
