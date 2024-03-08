# Casaeval: Data Processing and Prediction Evaluation of Casanovo

Casaeval is a versatile tool designed to convert BRUKER TIMS data files (.d) into Mascot Generic Format (.mgf) files.
It also generates SLURM scripts for executing Casanovo on servers and includes evaluation scripts for analyzing the .mgf data.
These evaluation scripts compare Casanovo predictions with database predictions, enhancing data interpretation in mass spectrometry experiments for each measurement.
Additionally, Casaeval evaluates unmatched peptide annotations of unknown spectra based on their Casanovo prediction scores, saving those with high mean scores exceeding a specified threshold.
With seamless integration, Casaeval adds predictions directly to .mgf files, simplifying the analysis process for users.



**Installation:** Download dist folder, navigate into it and execute "pip install casaeval-0.0.1-py3-none-any.whl".
