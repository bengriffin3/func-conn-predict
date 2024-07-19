"""Submit jobs to the BMRC cluster.

"""

import os

def write_job_script(n_ICs, states, run, trans_prob_diag, n_chunks, model_mean, queue="short"):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"hcp-hmm-{n_ICs}-{states}-{run}-{trans_prob_diag}-{n_chunks}-{model_mean}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        file.write("#SBATCH -c 4\n")
        #file.write("#SBATCH --gres gpu:1\n")
        #file.write("source activate osld\n")
        file.write("source activate venv_nets\n")
        #file.write(f"python ../CH_01b_train_hmm_chunk_group_single_chunk.py {states} {run} {trans_prob_diag} {model_mean} {n_chunks} {chunk} \n")
        
        if model_mean==1:
            file.write(f"python ../scripts/00_train_hmm_chunk_group.py {n_ICs} {states} {run} {trans_prob_diag} {n_chunks} --model_mean\n")
        elif model_mean==0:
            file.write(f"python ../scripts/00_train_hmm_chunk_group.py {n_ICs} {states} {run} {trans_prob_diag} {n_chunks} --no-model_mean\n")

os.makedirs("logs", exist_ok=True)

n_ICs_vec = [25]
run_vec = [1]
states_vec = [8]
trans_prob_diag_vec = [10]
n_chunks = 12
model_mean_vec = [0, 1]

for n_ICs in n_ICs_vec:
    for states in states_vec:
        for run in run_vec:
            for trans_prob_diag in trans_prob_diag_vec:
                for model_mean in model_mean_vec:
                    write_job_script(n_ICs, states, run, trans_prob_diag, n_chunks, model_mean)
                    os.system("sbatch job.sh")
                    os.system("rm job.sh")
                    #exit()
