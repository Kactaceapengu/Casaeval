import os
import subprocess

def generate_slurm_script(command, input_file, run_time):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=casanovo
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --mem=4G
#SBATCH --time={run_time}
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

# Exit the SLURM script if a command fails
set -e

# Load the Conda module (if needed)
module load conda
module load cuda

# Activate your Conda environment
conda activate casanovo_env

# Run casanovo command with input file
srun python casanovo {command} {input_file}

# If we reached this point, the command succeeded. We clean up resources.
rm -rf $TMPDIR
"""
    return slurm_script
    
# Possibility to add casaeval commands for .mgf generation.


def write_slurm_script(slurm_script, output_filename):
    with open(output_filename, 'w') as f:
        f.write(slurm_script)

def generate_slurm_scripts(input_file_path, command, option, output_path=None):
    if input_file_path:
        if os.path.isfile(input_file_path):
            option = 'single'
        elif os.path.isdir(input_file_path):
            option = 'folder'
        else:
            raise ValueError("Invalid input path. Please provide a valid file or folder path.")

    if option == 'single':
        input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
        run_time = input("Enter run time (HH:MM:SS) [default: 1:30:00]: ") or "1:30:00"

        slurm_script = generate_slurm_script(command, input_file_path, run_time)

        if not output_path:
            output_path = input("Enter output path for SLURM script (leave blank for current directory): ").strip() or "."
            
        output_filename = os.path.join(output_path, f"{input_file_name}.slurm")

        write_slurm_script(slurm_script, output_filename)
        print(f"SLURM script generated: output_filename")

        execute = input("Execute the SLURM script with sbatch now? (y/n): ").strip().lower()
        if execute == 'y':
            subprocess.run(["sbatch", output_filename])
            print(f"{input_file_name}.slurm was sent to SLURM for execution.")

    elif option == 'folder':
        run_time = input("Enter run time (HH:MM:SS) [default: 1:30:00]: ") or "1:30:00"

        execute_all = input("Execute all generated SLURM scripts with sbatch now? (y/n): ").strip().lower()
        execute_scripts = execute_all == 'y'
        
        # Create output folder 
        if not output_path:
            output_folder = input("Enter output folder for SLURM scripts: ")
            
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over files in the folder and generate SLURM scripts
        for filename in os.listdir(folder_path):
            input_file = os.path.join(folder_path, filename)
            output_filename = os.path.join(output_folder, f"{filename}.slurm")
            slurm_script = generate_slurm_script(command, input_file, run_time)
            write_slurm_script(slurm_script, output_filename)
            print(f"SLURM script generated: {output_filename}")
            
            if execute_scripts:
                subprocess.run(["sbatch", output_filename])
                print(f"{filename}.slurm was sent to SLURM for execution.")

    else:
        print("Invalid option. Please enter 'single' or 'folder'.")