import pickle
import numpy as np
import os


n_ICs = 25
run = 1
n_states = 8
trans_prob_diag = 10

for model_mean in [True, False]:
    for n_chunk in [4, 12]:

        proj_dir = "/well/win-fmrib-analysis/users/psz102/nets-predict/nets_predict"
        results_dir = f"{proj_dir}/results/ICA_{n_ICs}"
        dynamic_dir = f"{results_dir}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunk}_chunks"

        with open(f'{dynamic_dir}/hmm_features_{n_chunk}_chunks.pickle', 'rb') as file:
            hmm_features_dict = pickle.load(file)

        # hmm_features_dict is a list of length n_chunks where each element is a dictonary made up of:
        # dict_keys(['fo_chunk', 'lt_chunk', 'intv_chunk', 'sr_chunk', 'means_chunk', 'covs_chunk', 'trans_prob_chunk', 'icovs_chunk'])


        for chunk in range(n_chunk):
            # note FOs and covs
            fractional_occupancies = hmm_features_dict[chunk]['fo_chunk']
            covariances = hmm_features_dict[chunk]['covs_chunk']

            # reshape FOs to same as covs
            fractional_occupancies_reshaped = fractional_occupancies[:, :, np.newaxis, np.newaxis]

            # Perform the multiplication using broadcasting
            weighted_covariances = fractional_occupancies_reshaped * covariances

            # save new weighted covs to dictionary
            hmm_features_dict[chunk]['weighted_covs_chunk'] = weighted_covariances


        with open(os.path.join(dynamic_dir, f"hmm_features_{n_chunk}_chunks.pickle"), 'wb') as file:
            pickle.dump(hmm_features_dict, file)

'''
with for loop

# Initialize an array to store the results
weighted_covariances = np.zeros_like(covariances)

# Iterate over each subject
for i in range(fractional_occupancies.shape[0]):
    # Iterate over each state
    for j in range(fractional_occupancies.shape[1]):
        # Multiply the covariance matrix by the respective fractional occupancy value
        weighted_covariances[i, j, :, :] = fractional_occupancies[i, j] * covariances[i, j, :, :]

# Print the shape of the result to verify
print(weighted_covariances.shape)


'''