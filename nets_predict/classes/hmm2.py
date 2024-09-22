import numpy as np
from osl_dynamics.inference import modes
import logging
from osl_dynamics.data import Data
from osl_dynamics.analysis.modes import calc_trans_prob_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

_logger = logging.getLogger("Chunk_project")

class HMMInferenceClass:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def get_summary_stats(self, stc):
        """Calculate and return summary statistics from state time courses."""
        return {
            'fo': modes.fractional_occupancies(stc),
            'lt': modes.mean_lifetimes(stc),
            'intv': modes.mean_intervals(stc),
            'sr': modes.switching_rates(stc)
        }


def get_inferred_parameters(self, time_series):
        """Infer parameters and return dynamic features."""
        data = Data(time_series)
        _logger.info("Infer state probabilities")
        alpha = self.model.get_alpha(data)

        n_ICs, n_sub, n_states = data.n_channels, len(data.time_series()), self.model.config.n_states
        
        # Infer state time courses and calculate summary statistics
        _logger.info("Calculating summary statistics")
        stc = modes.argmax_time_courses(alpha)
        summary_stats = self.get_summary_stats(stc)

        _logger.info("Dual-estimating subject-specific HMMs")
        # Dual-estimation: means, covariances, and transition probabilities
        means_dual, covs_dual = self.model.dual_estimation(data, alpha)
        trans_prob_chunk = calc_trans_prob_matrix(stc, n_states=n_states)

        # Weighted covariances
        weighted_covs_dual = summary_stats['fo'][:, :, np.newaxis, np.newaxis] * covs_dual

        # Calculate partial correlations
        icovs_chunk = np.array([
            [
                PartialCorrClass.partial_corr(covs_dual[sub, state])[1]
                for state in range(n_states)
            ]
            for sub in range(n_sub)
        ])

        # Return dictionary of all features
        return {
            "fo_chunk": summary_stats['fo'],
            "lt_chunk": summary_stats['lt'],
            "intv_chunk": summary_stats['intv'],
            "sr_chunk": summary_stats['sr'],
            "means_chunk": means_dual,
            "covs_chunk": covs_dual,
            "trans_prob_chunk": trans_prob_chunk,
            "icovs_chunk": icovs_chunk,
            "weighted_covs_chunk": weighted_covs_dual
        }


class FeatureEngineeringClass:

    def reshape_summary_stats(self, fo, intv, lt, sr):
        n_sub = fo.shape[0]
        
        # List of summary stats to be reshaped and concatenated
        stats = [fo, intv, lt, sr]
        
        # Reshape each summary stat and concatenate them along axis 1
        feat = np.hstack([np.transpose(stat.reshape(-1, n_sub)) for stat in stats])
        
        return feat

    def reshape_dynamic_features(self, hmm_features_dict, dynamic_add):

        # Unpack features from the dictionary
        icovs = hmm_features_dict['icovs_chunk']
        covs = hmm_features_dict['covs_chunk']
        means = hmm_features_dict['means_chunk']
        trans_prob = hmm_features_dict['trans_prob_chunk']
        fo = hmm_features_dict['fo_chunk']
        intv = hmm_features_dict['intv_chunk']
        lt = hmm_features_dict['lt_chunk']
        sr = hmm_features_dict['sr_chunk']
        weighted_covs =  hmm_features_dict['weighted_covs_chunk']
        weighted_icovs =  hmm_features_dict['weighted_icovs_chunk']

        # Handle reshaping based on the dynamic_add argument
        if dynamic_add == 'fc':
            feat = self.reshape_cov_features(covs)
        elif dynamic_add == 'weighted_covs':
            feat = self.reshape_cov_features(weighted_covs)
        elif dynamic_add == 'weighted_icovs':
            feat = self.reshape_icovs(weighted_icovs, nan_replace=True)
        elif dynamic_add == 'pc':
            feat = self.reshape_icovs(icovs)
        elif dynamic_add == 'means':
            feat = means.reshape(means.shape[0], -1)
        elif dynamic_add in ['tpms_ss', 'tpms_ss_only']:
            feat = self.concatenate_tpms_ss(trans_prob, fo, intv, lt, sr)
        elif dynamic_add == 'all':
            feat = self.concatenate_all(icovs, covs, means, trans_prob, fo, intv, lt, sr)
        else:
            raise ValueError(f"Unknown dynamic_add option: {dynamic_add}")

        return feat

    def reshape_icovs(self, icovs, nan_replace=False):
        """
        Reshape icovs by extracting upper off-diagonal elements and reshaping.
        Optionally replace NaNs with zeros.
        """
        icovs_off_diag = PartialCorrClass.extract_upper_off_main_diag(icovs)
        if nan_replace:
            icovs_off_diag = np.nan_to_num(icovs_off_diag)
        return icovs_off_diag.reshape(icovs_off_diag.shape[0], -1)

    def concatenate_tpms_ss(self, trans_prob, fo, intv, lt, sr):
        """
        Concatenate trans_prob and summary statistics (fo, intv, lt, sr).
        """
        feat_tpm = trans_prob.reshape(trans_prob.shape[0], -1)
        feat_ss = self.reshape_summary_stats(fo, intv, lt, sr)
        return np.concatenate([feat_tpm, feat_ss], axis=1)

    def concatenate_all(self, icovs, covs, means, trans_prob, fo, intv, lt, sr):
        """
        Concatenate all features including icovs, covs, means, trans_prob, and summary stats.
        """
        feat_icovs = self.reshape_icovs(icovs)
        feat_covs = self.reshape_cov_features(covs)
        feat_means = means.reshape(means.shape[0], -1)
        feat_tpm = trans_prob.reshape(trans_prob.shape[0], -1)
        feat_ss = self.reshape_summary_stats(fo, intv, lt, sr)
        return np.concatenate([feat_icovs, feat_covs, feat_means, feat_tpm, feat_ss], axis=1)

    def reshape_cov_features(self, netmats):
        # input is a netmats of n_sub x n_IC x n_IC
        m, n = np.triu_indices(netmats.shape[-1])
        feat = netmats[..., m, n] # flatten covs 

        # flatten across states (if just static just doesn't do anything)
        feat = feat.reshape(feat.shape[0], -1)

        return feat

    def determine_n_features(self, features_to_use, n_ICs, n_states=0):
        if n_states == 0 and 'static' not in features_to_use:
            raise ValueError("Using dynamic features but n_states not specified")

        match features_to_use:
            case "fc" | "weighted_covs":
                return self._calculate_feature_size(n_ICs, n_states, True)
            case "pc" | "weighted_icovs":
                return self._calculate_feature_size(n_ICs, n_states, False)
            case "means":
                return n_states * n_ICs
            case "tpms_ss":
                return self._tpms_summary_stats_size(n_ICs, n_states)
            case "static" | "static_pca" | "static_connecting_edges" | "static_self_edge_only":
                return self._static_feature_size(n_ICs)
            case _:
                raise ValueError("Unknown feature type")

    def _calculate_feature_size(self, n_ICs, n_states, include_diag):
        n_upper_diag = (n_ICs * (n_ICs + 1)) // 2 if include_diag else (n_ICs * (n_ICs - 1)) // 2
        return n_states * n_upper_diag
    
    def _tpms_summary_stats_size(self, n_ICs, n_states):
        n_summary_stats = n_states * 4
        n_tpms = n_states * n_states
        return n_summary_stats + n_tpms
    
    def _static_feature_size(self, n_ICs):
        return (n_ICs * (n_ICs - 1)) // 2        
    
    def get_predictor_features(self, netmats, hmm_features_dict, features_to_use, edge_index):

        if features_to_use=='static' or features_to_use=='static_pca' or features_to_use=='static_self_edge_only':
            X = PartialCorrClass.extract_upper_off_main_diag(netmats)
        elif features_to_use=='static_connecting_edges':
            row, col = PartialCorrClass.find_original_indices(edge_index, netmats)
            X_incoming = netmats[:, row, :]
            X_incoming = np.delete(X_incoming, col, axis=1) # delete the self edge to avoid duplication
            X_outgoing = netmats[:, :, col]
            X = np.concatenate([X_incoming, X_outgoing],axis=1) 
        else:
            if features_to_use=='tpms_ss_only':
                X = self.reshape_dynamic_features(hmm_features_dict, features_to_use)

            else:
                X_static = PartialCorrClass.extract_upper_off_main_diag(netmats)

                # Form the design matrix of dynamic features (depending on features select in 'dynamic_add')
                X_dynamic = self.reshape_dynamic_features(hmm_features_dict, features_to_use)
                
                # Combine static and dynamic
                X = np.concatenate([X_static, X_dynamic],axis=1) 

        return X

    def intialise_trans_prob(self, trans_prob_diag, n_states):
            
        initial_trans_prob = np.ones([n_states, n_states])
        np.fill_diagonal(initial_trans_prob, trans_prob_diag)
        initial_trans_prob /= np.sum(initial_trans_prob, axis=1, keepdims=True)

        return initial_trans_prob


class TimeSeriesProcessingClass:
    def split_time_series(self, data, n_chunks, n_sub=None, n_ICs=None):

        if n_sub is None:
            n_sub = len(data.time_series())
        else:
            if n_sub != len(data.time_series()):
                raise ValueError("Incorrect number of subjects")

        if n_ICs is None:
            n_ICs = data.n_channels
        else:
            if n_ICs != data.n_channels:
                raise ValueError("Incorrect number of channels selected")

        n_timepoints = data.time_series()[0].shape[0]

        time_series_chunk = np.zeros((n_sub, n_chunks, int(n_timepoints/n_chunks),n_ICs))

        # split data into chunks
        for sub in range(n_sub):
            # get data for 1 subject and split it into 'n' chunks
            time_series_sub = np.split(data.time_series()[sub],n_chunks,axis=0)

            # for given subject, store the chunked up data
            for chunk in range(n_chunks):
                time_series_chunk[sub,chunk,:,:] = time_series_sub[chunk]

        return time_series_chunk


    def NormalizeData(self, data):
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))



class Analysis:
    def __init__(self):
        # Store variable names for easier reference in generalized functions
        self.var_names = ['fo', 'lt', 'sr', 'intv', 'means', 'covs', 'trans_prob', 'icovs']

    def get_std_across(self, axis, *args):
        # Generalized function to calculate standard deviation across a given axis
        return [np.std(data, axis=axis) for data in args]

    def get_mean_std(self, *args):
        # Generalized function to calculate the mean across the first axis
        return [np.mean(data, axis=0) for data in args]

    def get_std_across_chunks(self, *args):
        return self.get_std_across(axis=0, *args)

    def get_std_across_subs(self, *args):
        return self.get_std_across(axis=1, *args)

    def get_mean_std_chunks(self, *args):
        return self.get_mean_std(*args)

    def get_mean_std_subs(self, *args):
        return self.get_mean_std(*args)


class HMMFeatures:
    def __init__(self):
        # Define feature names to generalize operations
        self.feature_names = ['fo', 'lt', 'sr', 'intv', 'means', 'covs', 'trans_prob', 'icovs']

    def organise_hmm_features_across_chunks(self, hmm_features_list):
        n_chunk = len(hmm_features_list)
        n_sub = hmm_features_list[0]['fo_chunk'].shape[0]
        n_states = hmm_features_list[0]['fo_chunk'].shape[1]
        n_ICs = hmm_features_list[0]['means_chunk'].shape[2]

        # Initialize a dictionary to hold all feature arrays
        features = {
            'fo': np.zeros((n_chunk, n_sub, n_states)),
            'lt': np.zeros((n_chunk, n_sub, n_states)),
            'sr': np.zeros((n_chunk, n_sub, n_states)),
            'intv': np.zeros((n_chunk, n_sub, n_states)),
            'means': np.zeros((n_chunk, n_sub, n_states, n_ICs)),
            'covs': np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs)),
            'trans_prob': np.zeros((n_chunk, n_sub, n_states, n_states)),
            'icovs': np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))
        }

        # Populate the feature arrays
        for chunk in range(n_chunk):
            for feature in self.feature_names:
                features[feature][chunk] = hmm_features_list[chunk][f'{feature}_chunk']

        # Return feature arrays as a tuple
        return tuple(features[feature] for feature in self.feature_names)

    def normalize_hmm_features(self, *features):
        # Normalize each feature using the provided 'normalize_feature' function
        return tuple(self.normalize_feature(feature) for feature in features)

    def  normalize_feature(self, feature):

        feature_shape = feature.shape
        feature.shape = (-1) #flatten
        feature_norm = self.NormalizeData(feature)
        feature_norm.shape = (feature_shape) #unflatten
    
        return feature_norm

class Prediction:

    def standardise_train_apply_to_test(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test
    
    def evaluate_model(self, model, X_test, y_test, my):
        # move to a prediction class and move correlation to another function called 'get_stats' and get other stats like R^2 (save in list or dictionary probs)
        y_pred = model.predict(X_test)
        y_pred = y_pred + my
        correlation = pearsonr(y_test.squeeze(), y_pred.squeeze())[0]
        return y_pred, correlation

    
    def pca_dynamic_only(self, X_train, X_test, n_ICs):
        
         # we keep the static features as they are so let's separate them out
        n_static = self.determine_n_features('static', n_ICs)
        X_train_static = X_train[:, 0:n_static]
        X_test_static = X_test[:, 0:n_static]
        X_train_dynamic = X_train[:, n_static:]
        X_test_dynamic = X_test[:, n_static:]

        # perform PCA
        pca_model = PCA(0.9999999, svd_solver='full')
        pca_model.fit(X_train_dynamic)
        print(f"Number of new features (i.e., PCs): {np.argmax(np.cumsum(pca_model.explained_variance_ratio_)>0.99)}")
        X_train_dynamic = pca_model.transform(X_train_dynamic)
        X_test_dynamic = pca_model.transform(X_test_dynamic)

        # combine static and dynamic features
        X_train = np.concatenate((X_train_static, X_train_dynamic), axis=1)
        X_test = np.concatenate((X_test_static, X_test_dynamic), axis=1)

        return X_train, X_test, pca_model

    def self_predict_plus_pca(self, X, edge_index):
        # 1) note down edge C_ii (i.e. self edge)
        X_self_edge = X[:, edge_index].reshape(-1, 1)

        # 2) perform PCA on all the remaining edges
        X_without_self_edge = np.delete(X, edge_index, axis=1)
        pca_model = PCA(0.85, svd_solver='full')
        X_without_self_edge_pca = pca_model.fit_transform(X_without_self_edge)

        # 3) combine into new X
        X = np.concatenate((X_self_edge, X_without_self_edge_pca), axis=1)

        return X


    # def get_std_across_chunks(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

    #     fo_std_across_chunks = np.std(fo_all, axis = 0)
    #     lt_std_across_chunks = np.std(lt_all, axis = 0)
    #     sr_std_across_chunks = np.std(sr_all, axis = 0)
    #     intv_std_across_chunks = np.std(intv_all, axis = 0)
    #     trans_prob_std_across_chunks = np.std(trans_prob_all, axis = 0)
    #     means_std_across_chunks = np.std(means_all, axis = 0)
    #     covs_std_across_chunks = np.std(covs_all, axis = 0)
    #     icovs_std_across_chunks = np.std(icovs_all, axis = 0)

    #     return fo_std_across_chunks, lt_std_across_chunks, sr_std_across_chunks, intv_std_across_chunks, means_std_across_chunks, covs_std_across_chunks, trans_prob_std_across_chunks, icovs_std_across_chunks

    # def get_std_across_subs(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

    #     fo_std_across_subs = np.std(fo_all, axis = 1)
    #     lt_std_across_subs = np.std(lt_all, axis = 1)
    #     sr_std_across_subs = np.std(sr_all, axis = 1)
    #     intv_std_across_subs = np.std(intv_all, axis = 1)
    #     trans_prob_std_across_subs = np.std(trans_prob_all, axis = 1)
    #     means_std_across_subs = np.std(means_all, axis = 1)
    #     covs_std_across_subs = np.std(covs_all, axis = 1)
    #     icovs_std_across_subs = np.std(icovs_all, axis = 1)

    #     return fo_std_across_subs, lt_std_across_subs, sr_std_across_subs, intv_std_across_subs, means_std_across_subs, covs_std_across_subs, trans_prob_std_across_subs, icovs_std_across_subs

    # def get_mean_std_chunks(self, fo_std_across_chunks, lt_std_across_chunks, sr_std_across_chunks, intv_std_across_chunks, means_std_across_chunks, covs_std_across_chunks, trans_prob_std_across_chunks, icovs_std_across_chunks):

    #     fo_std_mean_chunks = np.mean(fo_std_across_chunks, axis=0)
    #     lt_std_mean_chunks = np.mean(lt_std_across_chunks, axis=0)
    #     sr_std_mean_chunks = np.mean(sr_std_across_chunks, axis=0)
    #     intv_std_mean_chunks = np.mean(intv_std_across_chunks, axis=0)
    #     means_std_mean_chunks = np.mean(means_std_across_chunks, axis=0)
    #     covs_std_mean_chunks = np.mean(covs_std_across_chunks, axis=0)
    #     trans_prob_std_mean_chunks = np.mean(trans_prob_std_across_chunks, axis=0)
    #     icovs_std_mean_chunks = np.mean(icovs_std_across_chunks, axis=0)

    #     return fo_std_mean_chunks, lt_std_mean_chunks, sr_std_mean_chunks, intv_std_mean_chunks, means_std_mean_chunks, covs_std_mean_chunks, trans_prob_std_mean_chunks, icovs_std_mean_chunks 

    # def get_mean_std_subs(self, fo_std_across_subs, lt_std_across_subs, sr_std_across_subs, intv_std_across_subs, means_std_across_subs, covs_std_across_subs, trans_prob_std_across_subs, icovs_std_across_subs):

    #     fo_std_mean_subs = np.mean(fo_std_across_subs, axis=0)
    #     lt_std_mean_subs = np.mean(lt_std_across_subs, axis=0)
    #     sr_std_mean_subs = np.mean(sr_std_across_subs, axis=0)
    #     intv_std_mean_subs = np.mean(intv_std_across_subs, axis=0)
    #     means_std_mean_subs = np.mean(means_std_across_subs, axis=0)
    #     covs_std_mean_subs = np.mean(covs_std_across_subs, axis=0)
    #     trans_prob_std_mean_subs = np.mean(trans_prob_std_across_subs, axis=0)
    #     icovs_std_mean_subs = np.mean(icovs_std_across_subs, axis=0)

    #     return fo_std_mean_subs, lt_std_mean_subs, sr_std_mean_subs, intv_std_mean_subs, means_std_mean_subs, covs_std_mean_subs, trans_prob_std_mean_subs, icovs_std_mean_subs




#    def  organise_hmm_features_across_chunks(self, hmm_features_list):

#         n_chunk = len(hmm_features_list)
#         n_sub = hmm_features_list[0]['fo_chunk'].shape[0]
#         n_states = hmm_features_list[0]['fo_chunk'].shape[1]
#         n_ICs = hmm_features_list[0]['means_chunk'].shape[2]

#         fo_all = np.zeros((n_chunk, n_sub, n_states))
#         lt_all = np.zeros((n_chunk, n_sub, n_states))
#         intv_all = np.zeros((n_chunk, n_sub, n_states))
#         sr_all = np.zeros((n_chunk, n_sub, n_states))
#         means_all =np.zeros((n_chunk, n_sub, n_states, n_ICs))
#         covs_all =np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))
#         trans_prob_all =np.zeros((n_chunk, n_sub, n_states, n_states))
#         icovs_all =np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))

#         for chunk in range(n_chunk):
#             fo_all[chunk, :, :] = hmm_features_list[chunk]['fo_chunk']
#             lt_all[chunk, :, :] = hmm_features_list[chunk]['lt_chunk']
#             sr_all[chunk, :, :] = hmm_features_list[chunk]['sr_chunk']
#             intv_all[chunk, :, :] = hmm_features_list[chunk]['intv_chunk']
#             means_all[chunk, :, :, :] = hmm_features_list[chunk]['means_chunk']
#             covs_all[chunk, :, :, :, :] = hmm_features_list[chunk]['covs_chunk']
#             trans_prob_all[chunk, :, :, :] = hmm_features_list[chunk]['trans_prob_chunk']
#             icovs_all[chunk, :, :, :, :] = hmm_features_list[chunk]['icovs_chunk']


#         return fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all

#     def  normalize_hmm_features(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

#         fo_all_norm = self.normalize_feature(fo_all)
#         lt_all_norm = self.normalize_feature(lt_all)
#         sr_all_norm = self.normalize_feature(sr_all)
#         intv_all_norm = self.normalize_feature(intv_all)
#         means_all_norm = self.normalize_feature(means_all)
#         covs_all_norm = self.normalize_feature(covs_all)
#         trans_prob_all_norm = self.normalize_feature(trans_prob_all)
#         icovs_all_norm = self.normalize_feature(icovs_all)

#         return fo_all_norm, lt_all_norm, sr_all_norm, intv_all_norm, means_all_norm, covs_all_norm, trans_prob_all_norm, icovs_all_norm




    # def determine_n_features(self, features_to_use, n_ICs, n_states=0):

    #     if (n_states == 0 and (features_to_use != 'static' and features_to_use != 'static_pca' and features_to_use != 'static_connecting_edges' and features_to_use != 'static_self_edge_only')):
    #     #if (n_states == 0 and ('static' not in features_to_use)):
    #         _logger.exception("Using dynamic features but n_states not specified", stack_info=True, exc_info=True)
    #         exit()

    #     n_upper_diag = n_ICs*(n_ICs+1)/2 
    #     n_static_features = n_ICs*(n_ICs-1)/2

    #     # change to MATCH and CASE
    #     match features_to_use:
    #         case "fc" | "weighted_covs":
    #             n_fc = n_states * n_upper_diag
    #             n_features = int(n_fc + n_static_features) # add 1 to n_states for the statics
    #         case "pc" | "weighted_icovs":
    #             n_pc = n_states * n_static_features # also ignore main diagonal for PC here
    #             n_features = int(n_pc + n_static_features)
    #         case "means":
    #             n_means = n_states * n_ICs
    #             n_features = int(n_means + n_static_features)
    #         case "tpms_ss":
    #             n_tpms = n_states * n_states
    #             n_summary_stats = n_states * 4
    #             n_features = int(n_tpms + n_summary_stats + n_static_features) # 4 summary states for each state
    #         case "all":
    #             n_fc = n_states * n_upper_diag
    #             n_pc = n_states * n_static_features # also ignore main diagonal for PC here
    #             n_means = n_states * n_ICs
    #             n_tpms = n_states * n_states
    #             n_summary_stats = n_states * 4 
    #             n_features = int(n_fc + n_pc + n_means + n_tpms +  n_summary_stats + n_static_features)
    #         case "tpms_ss_only":
    #             n_tpms = n_states * n_states
    #             n_summary_stats = n_states * 4 
    #             n_features = int(n_tpms + n_summary_stats) # 4 summary states for each state
    #         case "static" | "static_pca" | "static_connecting_edges" | "static_self_edge_only":
    #             n_features = int(n_static_features)
    #         case _:
    #             raise ValueError('The inputted static/dynamic features are not a valid option)')


    #     return n_features


        # def reshape_summary_stats(self, fo, intv, lt, sr):
        # n_sub = fo.shape[0]

        # # reshape summary stats
        # fo_sub = np.transpose(fo.reshape(-1, n_sub))
        # intv_sub = np.transpose(intv.reshape(-1, n_sub))
        # lt_sub = np.transpose(lt.reshape(-1, n_sub))
        # sr_sub = np.transpose(sr.reshape(-1, n_sub))

        # # combine in feature matrix
        # feat = fo_sub
        # feat = np.concatenate([feat, intv_sub],axis=1)
        # feat = np.concatenate([feat, lt_sub],axis=1)
        # feat = np.concatenate([feat, sr_sub],axis=1)

        # return feat