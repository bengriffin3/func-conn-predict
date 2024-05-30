import numpy as np
from osl_dynamics.inference import modes
import logging
from sys import exit
from osl_dynamics.data import Data
from osl_dynamics.analysis.modes import calc_trans_prob_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from nets_predict.classes.partial_correlation import PartialCorrelationClass

_logger = logging.getLogger("Chunk_project")
PartialCorrClass = PartialCorrelationClass()

class HiddenMarkovModelClass:
    """Class for running functions related to the HMM.

    Parameters
    ----------
    Xxx : Xxx
        Xxx.
    """ 

    def intialise_trans_prob(self, trans_prob_diag, n_states):
            
        initial_trans_prob = np.ones([n_states, n_states])
        np.fill_diagonal(initial_trans_prob, trans_prob_diag)
        initial_trans_prob /= np.sum(initial_trans_prob, axis=1, keepdims=True)

        return initial_trans_prob

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
    
    def reshape_cov_features(self, netmats):
        """Extract upper diagonal from HMM covariances.

        Parameters
        ----------
        netmats : np.ndarray
            netmats for given subjects (e.g. partial correlation from fMRI time series). Shape is (n_sub, n_ICs, n_ICs).

        Returns
        -------
        feat : np.ndarray
            reshaped netmats. Shape is (n_sub, (n_ICs * (n_ICs + 1))/2)
        """
        # input is a netmats of n_sub x n_IC x n_IC
        m, n = np.triu_indices(netmats.shape[-1])
        feat = netmats[..., m, n] # flatten covs 

        # flatten across states (if just static just doesn't do anything)
        feat = feat.reshape(feat.shape[0], -1)

        return feat


    def reshape_summary_stats(self, fo, intv, lt, sr):
        n_sub = fo.shape[0]

        # reshape summary stats
        fo_sub = np.transpose(fo.reshape(-1, n_sub))
        intv_sub = np.transpose(intv.reshape(-1, n_sub))
        lt_sub = np.transpose(lt.reshape(-1, n_sub))
        sr_sub = np.transpose(sr.reshape(-1, n_sub))

        # combine in feature matrix
        feat = fo_sub
        feat = np.concatenate([feat, intv_sub],axis=1)
        feat = np.concatenate([feat, lt_sub],axis=1)
        feat = np.concatenate([feat, sr_sub],axis=1)

        return feat

    def reshape_dynamic_features(self, hmm_features_dict, dynamic_add):

        # verify shape of inputs
        icovs = hmm_features_dict['icovs_chunk']
        covs = hmm_features_dict['covs_chunk']
        means = hmm_features_dict['means_chunk']
        trans_prob = hmm_features_dict['trans_prob_chunk']
        fo = hmm_features_dict['fo_chunk']
        intv = hmm_features_dict['intv_chunk']
        lt = hmm_features_dict['lt_chunk']
        sr = hmm_features_dict['sr_chunk']

        if dynamic_add=='fc':
            feat = self.reshape_cov_features(covs)
        elif dynamic_add=='pc':
            #feat = self.reshape_cov_features(icovs)
            icovs_off_diag = PartialCorrClass.extract_upper_off_main_diag(icovs)
            feat = icovs_off_diag.reshape(icovs_off_diag.shape[0], -1)
        elif dynamic_add=='means':
            feat = means.reshape(means.shape[0], -1)
        elif dynamic_add=='tpms_ss' or dynamic_add=='tpms_ss_only':
            feat_tpm = trans_prob.reshape(trans_prob.shape[0], -1)
            feat_ss = self.reshape_summary_stats(fo, intv, lt, sr)
            feat = np.concatenate([feat_tpm, feat_ss],axis=1)
        elif dynamic_add=='all':
            #feat_icovs = self.reshape_cov_features(icovs)
            icovs_off_diag = PartialCorrClass.extract_upper_off_main_diag(icovs)
            feat_icovs = icovs_off_diag.reshape(icovs_off_diag.shape[0], -1)
            feat_covs = self.reshape_cov_features(covs)
            feat_means = means.reshape(means.shape[0], -1)
            feat_tpm = trans_prob.reshape(trans_prob.shape[0], -1)
            feat_ss = self.reshape_summary_stats(fo, intv, lt, sr)
            feat = np.concatenate([feat_icovs, feat_covs, feat_means, feat_tpm, feat_ss],axis=1)        

        return feat
    
    
    # def load_dynamic_features_chunk(self, proj_dir, n_ICs, n_sub, n_chunk, run, n_states, trans_prob_diag, model_mean):
        
    #     #results_dir = f"{proj_dir}/chunk_analysis/results/ICA_{n_ICs}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}/{n_chunk}_chunks"
    #     results_dir = f"{proj_dir}/chunk_analysis/results/ICA_{n_ICs}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunk}_chunks"
    #     icovs = np.load(f"{results_dir}/dual_estimates/icovs_{n_chunk}_chunks.npy") 
    #     covs = np.load(f"{results_dir}/dual_estimates/covs_{n_chunk}_chunks.npy") 
    #     means = np.load(f"{results_dir}/dual_estimates/means_{n_chunk}_chunks.npy") 
    #     trans_prob = np.load(f"{results_dir}/dual_estimates/trans_prob_{n_chunk}_chunks.npy") 
    #     fo = np.load(f"{results_dir}/summary_stats/fo_{n_chunk}_chunks_group.npy") 
    #     intv = np.load(f"{results_dir}/summary_stats/intv_{n_chunk}_chunks_group.npy") 
    #     lt = np.load(f"{results_dir}/summary_stats/lt_{n_chunk}_chunks_group.npy") 
    #     sr = np.load(f"{results_dir}/summary_stats/sr_{n_chunk}_chunks_group.npy") 

    #     return icovs, covs,means,trans_prob,fo,intv,lt,sr

    def get_summary_stats(self, stc):

        # validate stc is legit shape etc.


        fo = modes.fractional_occupancies(stc)
        lt = modes.mean_lifetimes(stc)
        intv = modes.mean_intervals(stc)
        sr = modes.switching_rates(stc)

        return fo, lt, intv, sr
    

    def determine_n_features(self, features_to_use, n_ICs, n_states=0):

        if n_states == 0 and features_to_use != 'static':
            _logger.exception("Using dynamic features but n_states not specified", stack_info=True, exc_info=True)
            exit()

        n_upper_diag = n_ICs*(n_ICs+1)/2 
        n_static_features = n_ICs*(n_ICs-1)/2

        # change to MATCH and CASE
        match features_to_use:
            case "fc":
                n_fc = n_states * n_upper_diag
                n_features = int(n_fc + n_static_features) # add 1 to n_states for the statics    
            case "pc":
                n_pc = n_states * n_static_features # also ignore main diagonal for PC here
                n_features = int(n_pc + n_static_features)
            case "means":
                n_means = n_states * n_ICs
                n_features = int(n_means + n_static_features)
            case "tpms_ss":
                n_tpms = n_states * n_states
                n_summary_stats = n_states * 4
                n_features = int(n_tpms + n_summary_stats + n_static_features) # 4 summary states for each state
            case "all":
                n_fc = n_states * n_upper_diag
                n_pc = n_states * n_static_features # also ignore main diagonal for PC here
                n_means = n_states * n_ICs
                n_tpms = n_states * n_states
                n_summary_stats = n_states * 4 
                n_features = int(n_fc + n_pc + n_means + n_tpms +  n_summary_stats + n_static_features)
            case "tpms_ss_only":
                n_tpms = n_states * n_states
                n_summary_stats = n_states * 4 
                n_features = int(n_tpms + n_summary_stats) # 4 summary states for each state
            case "static":
                n_features = int(n_static_features)
            case _:
                raise ValueError('The inputted static/dynamic features are not a valid option)')


        return n_features
    
    def get_inferred_parameters(self, model, time_series):
        """Infer HMM parameters.

        Parameters
        ----------
        model : OSLD object 
            HMM model containing fully_trained HMM.

        time_series:
            Time series to which the HMM was applied to


        Returns
        -------
        Xxx

        """  
      
        #
        data = Data(time_series)
        _logger.info("Infer state probabilities")
        alpha = model.get_alpha(data)
        n_ICs = data.n_channels
        n_sub = len(data.time_series())
        n_states = model.config.n_states

        #Calculate summary statistics
        _logger.info("Calculating summary statistics")
        stc = modes.argmax_time_courses(alpha)
        fo, lt, intv, sr = self.get_summary_stats(stc)

        _logger.info("Dual-estimating subject-specific HMMs")
        # Calculate subject-specific means, covariances, and TPMs
        means_dual, covs_dual = model.dual_estimation(data, alpha)
        trans_prob_chunk = calc_trans_prob_matrix(stc, n_states=n_states)

        # calculate partial correlations 
        icovs_chunk = np.zeros((n_sub, n_states, n_ICs, n_ICs))
        for sub in range(n_sub):
            for state in range(n_states):
                icovs_chunk[sub,state,:,:] = PartialCorrClass.partial_corr(covs_dual[sub, state, : :])[1]

        # return dictionary of dynamic features
        return {
            "fo_chunk": fo,
            "lt_chunk": lt,
            "intv_chunk": intv,
            "sr_chunk": sr,
            "means_chunk": means_dual,
            "covs_chunk": covs_dual,
            "trans_prob_chunk": trans_prob_chunk,
            "icovs_chunk": icovs_chunk
        }
    
    def get_predictor_features(self, netmats, hmm_features_dict, features_to_use):

        if features_to_use!='static': 
            if features_to_use=='tpms_ss_only':
                X = self.reshape_dynamic_features(hmm_features_dict, features_to_use)

            else:
                pass
                X_static = PartialCorrClass.extract_upper_off_main_diag(netmats)

                # Form the design matrix of dynamic features (depending on features select in 'dynamic_add')
                X_dynamic = self.reshape_dynamic_features(hmm_features_dict, features_to_use)
                
                # Combine static and dynamic
                X = np.concatenate([X_static, X_dynamic],axis=1)      
        else: # use static features only
            X = PartialCorrClass.extract_upper_off_main_diag(netmats)

        return X 

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
    
    def get_std_across_chunks(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

        fo_std_across_chunks = np.std(fo_all, axis = 0)
        lt_std_across_chunks = np.std(lt_all, axis = 0)
        sr_std_across_chunks = np.std(sr_all, axis = 0)
        intv_std_across_chunks = np.std(intv_all, axis = 0)
        trans_prob_std_across_chunks = np.std(trans_prob_all, axis = 0)
        means_std_across_chunks = np.std(means_all, axis = 0)
        covs_std_across_chunks = np.std(covs_all, axis = 0)
        icovs_std_across_chunks = np.std(icovs_all, axis = 0)

        return fo_std_across_chunks, lt_std_across_chunks, sr_std_across_chunks, intv_std_across_chunks, means_std_across_chunks, covs_std_across_chunks, trans_prob_std_across_chunks, icovs_std_across_chunks

    def get_std_across_subs(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

        fo_std_across_subs = np.std(fo_all, axis = 1)
        lt_std_across_subs = np.std(lt_all, axis = 1)
        sr_std_across_subs = np.std(sr_all, axis = 1)
        intv_std_across_subs = np.std(intv_all, axis = 1)
        trans_prob_std_across_subs = np.std(trans_prob_all, axis = 1)
        means_std_across_subs = np.std(means_all, axis = 1)
        covs_std_across_subs = np.std(covs_all, axis = 1)
        icovs_std_across_subs = np.std(icovs_all, axis = 1)

        return fo_std_across_subs, lt_std_across_subs, sr_std_across_subs, intv_std_across_subs, means_std_across_subs, covs_std_across_subs, trans_prob_std_across_subs, icovs_std_across_subs

    def get_mean_std_chunks(self, fo_std_across_chunks, lt_std_across_chunks, sr_std_across_chunks, intv_std_across_chunks, means_std_across_chunks, covs_std_across_chunks, trans_prob_std_across_chunks, icovs_std_across_chunks):

        fo_std_mean_chunks = np.mean(fo_std_across_chunks, axis=0)
        lt_std_mean_chunks = np.mean(lt_std_across_chunks, axis=0)
        sr_std_mean_chunks = np.mean(sr_std_across_chunks, axis=0)
        intv_std_mean_chunks = np.mean(intv_std_across_chunks, axis=0)
        means_std_mean_chunks = np.mean(means_std_across_chunks, axis=0)
        covs_std_mean_chunks = np.mean(covs_std_across_chunks, axis=0)
        trans_prob_std_mean_chunks = np.mean(trans_prob_std_across_chunks, axis=0)
        icovs_std_mean_chunks = np.mean(icovs_std_across_chunks, axis=0)

        return fo_std_mean_chunks, lt_std_mean_chunks, sr_std_mean_chunks, intv_std_mean_chunks, means_std_mean_chunks, covs_std_mean_chunks, trans_prob_std_mean_chunks, icovs_std_mean_chunks 

    def get_mean_std_subs(self, fo_std_across_subs, lt_std_across_subs, sr_std_across_subs, intv_std_across_subs, means_std_across_subs, covs_std_across_subs, trans_prob_std_across_subs, icovs_std_across_subs):

        fo_std_mean_subs = np.mean(fo_std_across_subs, axis=0)
        lt_std_mean_subs = np.mean(lt_std_across_subs, axis=0)
        sr_std_mean_subs = np.mean(sr_std_across_subs, axis=0)
        intv_std_mean_subs = np.mean(intv_std_across_subs, axis=0)
        means_std_mean_subs = np.mean(means_std_across_subs, axis=0)
        covs_std_mean_subs = np.mean(covs_std_across_subs, axis=0)
        trans_prob_std_mean_subs = np.mean(trans_prob_std_across_subs, axis=0)
        icovs_std_mean_subs = np.mean(icovs_std_across_subs, axis=0)

        return fo_std_mean_subs, lt_std_mean_subs, sr_std_mean_subs, intv_std_mean_subs, means_std_mean_subs, covs_std_mean_subs, trans_prob_std_mean_subs, icovs_std_mean_subs



    def  organise_hmm_features_across_chunks(self, hmm_features_list):

        n_chunk = len(hmm_features_list)
        n_sub = hmm_features_list[0]['fo_chunk'].shape[0]
        n_states = hmm_features_list[0]['fo_chunk'].shape[1]
        n_ICs = hmm_features_list[0]['means_chunk'].shape[2]

        fo_all = np.zeros((n_chunk, n_sub, n_states))
        lt_all = np.zeros((n_chunk, n_sub, n_states))
        intv_all = np.zeros((n_chunk, n_sub, n_states))
        sr_all = np.zeros((n_chunk, n_sub, n_states))
        means_all =np.zeros((n_chunk, n_sub, n_states, n_ICs))
        covs_all =np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))
        trans_prob_all =np.zeros((n_chunk, n_sub, n_states, n_states))
        icovs_all =np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))

        for chunk in range(n_chunk):
            fo_all[chunk, :, :] = hmm_features_list[chunk]['fo_chunk']
            lt_all[chunk, :, :] = hmm_features_list[chunk]['lt_chunk']
            sr_all[chunk, :, :] = hmm_features_list[chunk]['sr_chunk']
            intv_all[chunk, :, :] = hmm_features_list[chunk]['intv_chunk']
            means_all[chunk, :, :, :] = hmm_features_list[chunk]['means_chunk']
            covs_all[chunk, :, :, :, :] = hmm_features_list[chunk]['covs_chunk']
            trans_prob_all[chunk, :, :, :] = hmm_features_list[chunk]['trans_prob_chunk']
            icovs_all[chunk, :, :, :, :] = hmm_features_list[chunk]['icovs_chunk']


        return fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all


    def  normalize_hmm_features_OLD(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

        n_chunk = fo_all.shape[0]
        n_sub = fo_all.shape[1]
        n_states = fo_all.shape[2]
        n_ICs = covs_all.shape[3]


        fo_all_norm = np.zeros((n_chunk, n_sub, n_states))
        lt_all_norm = np.zeros((n_chunk, n_sub, n_states))
        intv_all_norm = np.zeros((n_chunk, n_sub, n_states))
        sr_all_norm = np.zeros((n_chunk, n_sub, n_states))
        means_all_norm =np.zeros((n_chunk, n_sub, n_states, n_ICs))
        covs_all_norm =np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))
        trans_prob_all_norm =np.zeros((n_chunk, n_sub, n_states, n_states))
        icovs_all_norm =np.zeros((n_chunk, n_sub, n_states, n_ICs, n_ICs))

        for chunk in range(n_chunk):
            for k1 in range(n_states):
                fo_all_norm[chunk, :, k1] = self.NormalizeData(fo_all[chunk, :, k1])
                lt_all_norm[chunk, :, k1] = self.NormalizeData(lt_all[chunk, :, k1])
                intv_all_norm[chunk, :, k1] = self.NormalizeData(intv_all[chunk, :, k1])
                sr_all_norm[chunk, :, k1] = self.NormalizeData(sr_all[chunk, :, k1])
                
                for k2 in range(n_states):
                    trans_prob_all_norm[chunk, :, k1, k2] = self.NormalizeData(trans_prob_all[chunk, :, k1, k2])
                
                for parcel1 in range(n_ICs):
                    means_all_norm[chunk, :, k1, parcel1] = self.NormalizeData(means_all[chunk, :, k1, parcel1])
                    
                    for parcel2 in range(n_ICs):
                        covs_all_norm[chunk, :, k1, parcel1, parcel2] = self.NormalizeData(covs_all[chunk, :, k1, parcel1, parcel2])
                        icovs_all_norm[chunk, :, k1, parcel1, parcel2] = self.NormalizeData(icovs_all[chunk, :, k1, parcel1, parcel2])

        return fo_all_norm, lt_all_norm, sr_all_norm, intv_all_norm, means_all_norm, covs_all_norm, trans_prob_all_norm, icovs_all_norm
    
    def  normalize_hmm_features(self, fo_all, lt_all, sr_all, intv_all, means_all, covs_all, trans_prob_all, icovs_all):

        fo_all_norm = self.normalize_feature(fo_all)
        lt_all_norm = self.normalize_feature(lt_all)
        sr_all_norm = self.normalize_feature(sr_all)
        intv_all_norm = self.normalize_feature(intv_all)
        means_all_norm = self.normalize_feature(means_all)
        covs_all_norm = self.normalize_feature(covs_all)
        trans_prob_all_norm = self.normalize_feature(trans_prob_all)
        icovs_all_norm = self.normalize_feature(icovs_all)

        return fo_all_norm, lt_all_norm, sr_all_norm, intv_all_norm, means_all_norm, covs_all_norm, trans_prob_all_norm, icovs_all_norm


    def  normalize_feature(self, feature):

        feature_shape = feature.shape
        feature.shape = (-1) #flatten
        feature_norm = self.NormalizeData(feature)
        feature_norm.shape = (feature_shape) #unflatten
    
        return feature_norm
    

    def NormalizeData(self, data):
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
