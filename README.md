# Casaeval: Data Processing and Prediction Evaluation of Casanovo

Casaeval is a versatile tool designed to utilize TimsRust in converting BRUKER TIMS data files (.d) into Mascot Generic Format (.mgf) files.

It also generates SLURM scripts for executing Casanovo on servers and includes evaluation scripts for analyzing the .mgf data.

Additionally, provided evaluation scripts compare Casanovo predictions with database predictions, enhancing data interpretation in mass spectrometry experiments for each measurement.

Casaeval assists in evaluating unmatched peptide annotations of unknown spectra based on their Casanovo prediction scores and the degree of alignment with database annotations, saving those with high mean scores exceeding a specified threshold.

With seamless integration, Casaeval adds predictions directly to .mgf files, simplifying the analysis process for users.



**Installation:** Download dist folder, navigate into it and execute "pip install casaeval-0.0.1-py3-none-any.whl".

**Usage:** The command 'casaeval -h' shows all possible commands: 'automate', 'timsconvert', 'predictions' and 'evaluate'. By executing f.ex. 'casaeval evaluate -h' their needed input are explained and shown. A documentary of all scripts will follow.
