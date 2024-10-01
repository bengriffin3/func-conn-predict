"""Submit jobs to the BMRC cluster.

"""

import os

def write_job_script(n_ICs, network_edge, network_matrix, prediction_matrix, n_chunks, queue="short"):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"hcp-hmm-{n_ICs}-{network_edge}-{network_matrix}-{prediction_matrix}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        #file.write("source activate osld\n")
        file.write("source activate venv_nets\n")
        file.write(f"../python CH_04_netmats_edge_prediction.py {n_ICs} {network_edge} {network_matrix} {prediction_matrix} {n_chunks} \n")


os.makedirs("logs", exist_ok=True)

prediction_matrix = 'icov'
network_edge_range = range(0, 325)
network_matrix = 'icov'
n_ICs = 25
n_chunks = 8

for network_edge in network_edge_range:
    write_job_script(n_ICs, network_edge, network_matrix, prediction_matrix, n_chunks)
    os.system("sbatch job.sh")
    os.system("rm job.sh")
    #exit()
