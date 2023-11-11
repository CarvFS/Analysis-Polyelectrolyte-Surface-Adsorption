#!/usr/bin/env bash
# Created by Alec Glisman (GitHub: @alec-glisman) on November 10, 2023.

# Node configuration
#SBATCH --partition=all --qos=dow --account=dow
#SBATCH --nodes=1 --ntasks=32
#SBATCH --mem=30G
#SBATCH --gres=gpu:0 --gpu-bind=closest

# Job information
#SBATCH --job-name=Analysis-Monomer
#SBATCH --time=2-0:00:00

# Runtime I/O
#SBATCH --mail-user=slurm.notifications@gmail.com --mail-type=END,FAIL
#SBATCH -o logs/jid_%j-node_%N-%x.log -e logs/jid_%j-node_%N-%x.log

# built-in shell options
set -o errexit  # exit when a command fails. Add || true to commands allowed to fail
set -o nounset  # exit when script tries to use undeclared variables
set -o pipefail # exit when a command in a pipe fails

# analysis method
python_script='mda_data_gen.py'
single_analysis='0'
sim_idx='0'

dir_sims_base="/home/aglisman/VSCodeProjects/Polyelectrolyte-Surface-Adsorption/data/completed"
dir_sims=(
    'sjobid_0-calcite-104surface-5nm_surface-8nm_vertical-1chain-PAcr-8mer-0Crb-0Ca-8Na-0Cl-300K-1bar-NVT'
)
n_sims="${#dir_sims[@]}"

# run analysis script
if [[ "${single_analysis}" != "1" ]]; then

    # for loop through all simulations
    for ((sim_idx = 0; sim_idx < n_sims; sim_idx++)); do
        echo "- Analysis on index $((sim_idx + 1))/${n_sims}..."
        cmd="python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
        echo "${cmd}"
        eval "${cmd}"
    done

# run single analysis job
else
    echo "- Single analysis on index $((sim_idx + 1))/${n_sims}..."
    cmd="python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
    echo "${cmd}"
    eval "${cmd}"
fi
