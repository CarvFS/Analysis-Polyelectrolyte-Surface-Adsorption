#!/usr/bin/env bash
# Created by Alec Glisman (GitHub: @alec-glisman) on July 22nd, 2022

# Node configuration
#SBATCH --partition=all --qos=dow --account=dow
#SBATCH --ntasks=32 --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:0 --gpu-bind=closest

# Job information
#SBATCH --job-name=Analysis
#SBATCH --time=2-0:00:00

# Runtime I/O
#SBATCH --mail-user=slurm.notifications@gmail.com --mail-type=END,FAIL
#SBATCH -o logs/jid_%j-node_%N-%x.log -e logs/jid_%j-node_%N-%x.log

# built-in shell options
set -o errexit # exit when a command fails. Add || true to commands allowed to fail
set -o nounset # exit when script tries to use undeclared variable

# analysis method
python_script='mda_analysis_4.py'
single_analysis='0'
sim_idx='6'

dir_sims_base='/nfs/zeal_nas/home_mount/aglisman/GitHub/Polyelectrolyte-Surface-Adsorption/data_archive/4_many_monomer_binding'

# dir sims is all subdirectories in the base directory
mapfile -t dir_sims < <(find "${dir_sims_base}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")
n_sims="${#dir_sims[@]}"

echo "Found ${#dir_sims[@]} simulations in ${dir_sims_base}"
for ((i = 0; i < ${#dir_sims[@]}; i++)); do
    echo "  ${dir_sims[${i}]}"
done

# run analysis script
mkdir -p "logs"
if [[ "${single_analysis}" != "1" ]]; then
    for ((sim_idx = 0; sim_idx < n_sims; sim_idx++)); do
        echo "- Analysis on index $((sim_idx + 1))/${n_sims}..."
        echo "python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
        {
            python3 "${python_script}" \
                --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}"
        } | tee "logs/${python_script%%.*}_idx_${sim_idx}.log" 2>&1
    done

# run single analysis job
else
    echo "- Single analysis on index $((sim_idx + 1))/${n_sims}..."
    echo "python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
    {
        python3 "${python_script}" \
            --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}"
    } | tee "logs/${python_script%%.*}_idx_${sim_idx}.log" 2>&1
fi
