#!/usr/bin/env bash
# Created by Alec Glisman (GitHub: @alec-glisman) on July 22nd, 2022

# Node configuration
#SBATCH --partition=all --qos=dow --account=dow
#SBATCH --ntasks=16 --nodes=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:0 --gpu-bind=closest

# Job information
#SBATCH --job-name=MDA-Anl
#SBATCH --time=2-0:00:00

# Runtime I/O
#SBATCH --mail-user=slurm.notifications@gmail.com --mail-type=END,FAIL
#SBATCH -o logs/jid_%j-node_%N-%x.log -e logs/jid_%j-node_%N-%x.log

# built-in shell options
set -o errexit # exit when a command fails. Add || true to commands allowed to fail
set -o nounset # exit when script tries to use undeclared variable

# analysis method
dask='0'
gnu_parallel='0'
single_analysis='0'
sim_idx='0'
python_script='mda_analysis.py'
dir_sims_base='/nfs/zeal_nas/data_mount/aglisman-data/1-electronic-continuum-correction/7-single-chain-surface-binding/6_single_chain_binding/cleaned'

# on ctrl-c, kill the dask server
dask_pid=''
trap 'echo "Caught signal"; kill -9 ${dask_pid}' SIGINT SIGTERM

# dir sims is all subdirectories in the base directory
mkdir -p "logs"
mapfile -t dir_sims < <(find "${dir_sims_base}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")
mapfile -t dir_sims < <(printf "%s\n" "${dir_sims[@]}" | sort)
n_sims="${#dir_sims[@]}"

echo "Found ${#dir_sims[@]} simulations in ${dir_sims_base}"
for ((i = 0; i < ${#dir_sims[@]}; i++)); do
    echo "  ${i}: ${dir_sims[${i}]}"
done

# start a dask server
if [[ "${dask}" == "1" ]]; then
    echo "- Starting Dask server..."
    python3 ./../../src/utils/dask_helper.py &
    dask_pid="${!}"
    sleep 3
fi

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

    # run parallel analysis using GNU parallel
    if [[ "${gnu_parallel}" == "1" ]]; then
        echo "- Running analysis in parallel..."
        parallel -j 32 --joblog "data/${python_script%%.*}_parallel.log" --halt-on-error 2 --keep-order \
            python3 "${python_script}" --dir "${dir_sims_base}/{1}" \
            ::: "${dir_sims[@]}"

    # run analysis serially
    else
        for ((sim_idx = 0; sim_idx < n_sims; sim_idx++)); do
            echo "- Analysis on index $((sim_idx + 1))/${n_sims}..."
            echo "python3 ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}"
            {
                python3 "${python_script}" \
                    --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}"
            } | tee "logs/${python_script%%.*}_idx_${sim_idx}.log" 2>&1
        done
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

# kill the dask server
if [[ "${dask}" == "1" ]]; then
    echo "- Killing Dask server..."
    kill -9 "${dask_pid}"
fi
