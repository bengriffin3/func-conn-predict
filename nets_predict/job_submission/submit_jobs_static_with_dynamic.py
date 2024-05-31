"""Submit jobs to the BMRC cluster.

"""

import os

def write_job_script(n_ICs, network_edge, network_matrix, prediction_matrix, n_chunks, features_to_use, run, n_states, trans_prob_diag, model_mean, queue="short"):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"edg-pre-{n_ICs}-{n_chunks}-{network_edge}-{network_matrix}-{prediction_matrix}-{features_to_use}-{run}-{n_states}-{trans_prob_diag}-{model_mean}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        #file.write("source activate osld\n")
        file.write("source activate venv_nets\n")

        if features_to_use=='static':
            file.write(f"python ../scripts/02_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} {features_to_use} \n")
        else:
            if model_mean==1:
                file.write(f"python ../scripts/02_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} {features_to_use} --run {run} --n_states {n_states} --trans_prob_diag {trans_prob_diag} --model_mean\n")
            elif model_mean==0:
                file.write(f"python ../scripts/02_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} {features_to_use} --run {run} --n_states {n_states} --trans_prob_diag {trans_prob_diag} --no-model_mean\n")


os.makedirs("logs", exist_ok=True)

prediction_matrix = 'icov'
network_matrix = 'icov'
n_ICs_vec = [50]
n_chunks_vec = [4]
run = 1
n_states = 8
trans_prob_diag = 10
#dynamic_add = 'fc'
feature_vec =  ['pc'] #, 'all']
model_mean = 1

####### input model mean if necessary

for n_ICs in n_ICs_vec:
    n_edges =  int((n_ICs*(n_ICs-1))/2)
    network_edge_range = range(0, n_edges, 5) #range(0, n_edges, 5)
    for features_to_use in feature_vec: 
        for network_edge in network_edge_range:
            for n_chunks in n_chunks_vec:
                write_job_script(n_ICs, network_edge, network_matrix, prediction_matrix, n_chunks, features_to_use, run, n_states, trans_prob_diag, model_mean)
                #os.system("sbatch --dependency=afterok:55634075 job.sh")
                os.system("sbatch job.sh")
                #os.system("rm job.sh")
                #exit()
