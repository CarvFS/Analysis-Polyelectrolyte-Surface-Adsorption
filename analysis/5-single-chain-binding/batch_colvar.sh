#!/usr/bin/env bash
# Created by Alec Glisman (GitHub: @alec-glisman) on July 22nd, 2022

# Node configuration
#SBATCH --partition=all --qos=dow --account=dow
#SBATCH --ntasks=32 --nodes=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:0 --gpu-bind=closest

# Job information
#SBATCH --job-name=Colvar-Anl
#SBATCH --time=2-0:00:00

# Runtime I/O
#SBATCH --mail-user=slurm.notifications@gmail.com --mail-type=END,FAIL
#SBATCH -o logs/jid_%j-node_%N-%x.log -e logs/jid_%j-node_%N-%x.log

# built-in shell options
set -o errexit # exit when a command fails. Add || true to commands allowed to fail
set -o nounset # exit when script tries to use undeclared variable

# analysis method
gnu_parallel='0'
single_analysis='0'
sim_idx='0'
python_script='colvar_analysis.py'
dir_sims_base='/nfs/zeal_nas/home_mount/aglisman/GitHub/Polyelectrolyte-Surface-Adsorption/data_archive/6_single_chain_binding/cleaned'

# dir sims is all subdirectories in the base directory
mkdir -p "logs"
mapfile -t dir_sims < <(find "${dir_sims_base}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")
mapfile -t dir_sims < <(printf "%s\n" "${dir_sims[@]}" | sort)
n_sims="${#dir_sims[@]}"

echo "Found ${#dir_sims[@]} simulations in ${dir_sims_base}"
for ((i = 0; i < ${#dir_sims[@]}; i++)); do
    echo "  ${i}: ${dir_sims[${i}]}"
done

# if there is more than 1 command line argument, then the first argument is the index of the simulation to analyze
if [[ "${#}" -gt 0 ]]; then
    single_analysis='1'
    sim_idx="${1}"
    if [[ "${sim_idx}" -ge "${n_sims}" ]]; then
        echo "Error: Simulation index ${sim_idx} is out of range [0, ${n_sims})"
        exit 1
    fi
fi

# run analysis script
if [[ "${single_analysis}" != "1" ]]; then
    if [[ "${gnu_parallel}" != "1" ]]; then
        for ((sim_idx = 0; sim_idx < n_sims; sim_idx++)); do
            echo "- Analysis on index $((sim_idx + 1))/${n_sims}..."
            echo "python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
            {
                python3 "${python_script}" \
                    --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}"
            } | tee "logs/${python_script%%.*}_idx_${sim_idx}.log" 2>&1
        done

    else
        echo "- Running analysis in parallel..."
        parallel -j 32 --joblog "logs/${python_script%%.*}_parallel.log" --halt-on-error 2 --keep-order \
            python3 "${python_script}" --dir "${dir_sims_base}/{1}" \
            ::: "${dir_sims[@]}"
    fi

# run single analysis job
else
    echo "- Single analysis on index $((sim_idx + 1))/${n_sims}..."
    echo "python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
    {
        python3 "${python_script}" \
            --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}"
    } | tee "logs/${python_script%%.*}_idx_${sim_idx}.log" 2>&1
fi
