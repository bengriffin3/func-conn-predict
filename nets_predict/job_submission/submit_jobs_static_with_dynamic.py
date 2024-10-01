"""Submit jobs to the BMRC cluster.

"""

import os
import numpy as np

def write_job_script(n_ICs, network_edge, network_matrix, prediction_matrix, n_chunks, features_to_use, run, n_states, trans_prob_diag, model_mean, pca, apply_filter, freq, prediction_model, queue="long"):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"edg-pre-{n_ICs}-{n_chunks}-{network_edge}-{network_matrix}-{prediction_matrix}-{features_to_use}-{run}-{n_states}-{trans_prob_diag}-{model_mean}-{apply_filter}-{freq}-{prediction_model}".replace('.', '_')
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        file.write("source activate venv_nets\n") #file.write("source activate osld\n")

        proj_path = "/well/win-fmrib-analysis/users/psz102/nets-predict/nets_predict"

        if features_to_use=='static' or features_to_use=='static_pca' or features_to_use=='static_connecting_edges' or features_to_use=='static_self_edge_only':
            file.write(f"python {proj_path}/scripts/02_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} {features_to_use} --prediction_model {prediction_model} \n")
        else:
            if model_mean==1:
                file.write(f"python {proj_path}/scripts/02_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} {features_to_use} --prediction_model {prediction_model} --run {run} --n_states {n_states} --trans_prob_diag {trans_prob_diag} --model_mean --pca\n")
            elif model_mean==0:
                file.write(f"python {proj_path}/scripts/02_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} {features_to_use} --prediction_model {prediction_model}  --run {run} --n_states {n_states} --trans_prob_diag {trans_prob_diag} --no-model_mean --pca\n")


os.makedirs("logs", exist_ok=True)


prediction_matrix = 'icov'
network_matrix = 'cov'
n_ICs_vec = [25]
n_chunks_vec = [4, 10]
run = 1
n_states = 8
trans_prob_diag = 10
feature_vec =  ['static'] # 'pc', 'fc', 'all', 'tpms_ss_only', 'tpms_ss', 'static', 'means']
model_mean = 1
pca = 0
prediction_model = 'xgboost'

####### input model mean if necessary

for n_ICs in n_ICs_vec:
    n_edges =  int((n_ICs*(n_ICs-1))/2)
    network_edge_range = range(0, n_edges, 2)
    for features_to_use in feature_vec: 
        for network_edge in network_edge_range:
            for n_chunks in n_chunks_vec:
                write_job_script(n_ICs, network_edge, network_matrix, prediction_matrix, n_chunks, features_to_use, run, n_states, trans_prob_diag, model_mean, pca, apply_filter, freq, prediction_model)
                #os.system("sbatch --dependency=afterok:55634075 job.sh")
                os.system("sbatch job.sh")
                os.system("rm job.sh")
                #exit()
