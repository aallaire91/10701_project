import numpy as np
import pandas as pd
import constants
from load_data import load_jigsaws_data_from_pkl,load_results_index,load_features_index,load_segmentation_from_pkl,load_features_from_exp_name
from load_data import load_annotations_from_exp_name
import segmentation
import os
import plot_results
import  metrics

feature_experiments = {'all_kin': constants.slave_cols_quat, 'pos': constants.slave_pos_cols_quat, 'vel':constants.slave_vel_cols_quat}
preprocess_default = ['standardize',]
gmm_preprocess_experiments = {0:['window_slds_2'],
                              1:['low_pass_1','window_slds_2'],
                              2:['low_pass_5','window_slds_2'],
                              3: ['low_pass_5','window_slds_2', 'pca_90'],
                              4: ['low_pass_5','window_slds_2', 'pca_95'],
                              5: ['low_pass_5','window_slds_2', 'pca_99'],
                              }
hmm_preprocess_default = ['standardize']
hmm_preprocess_experiments = {0:[],
                              1: ['low_pass_1'],
                              2: ['low_pass_5' ],
                              3: ['low_pass_5', 'pca_90'],
                              4: ['low_pass_5', 'pca_95'],
                              5: ['low_pass_5', 'pca_99']
                              }

gmm_func_plot_labels = ["DP_GMM", "BIC_GMM", "GMeans_GMM","GMM"]
hmm_func_plot_labels = ["AR_HDP_HSMM","HDP_HSMM"]

segment_colors_dict = {}

# def run_gt_evaluation(kin_df,ann_df):
#     demo_inds = pd.unique(ann_df.index.droplevel(level=2))
#     demo_list = []
#     ann_df_copy = ann_df.copy()
#     ann_df_copy.iloc[:,0:2] = ann_df_copy.iloc[:,0:2] - [1,1]
#
#     for i in range(0, demo_inds.shape[0]):
#         this_ann = ann_df_copy.loc[demo_inds[i]].copy()
#         this_kin = kin_df.loc[demo_inds[i], :].copy()
#         this_dem_df = pd.DataFrame()
#         for j in range(0,this_ann.shape[0]):
#             start_i = this_ann.iloc[j, 0].copy()
#             stop_i = this_ann.iloc[j,1].copy()
#             this_dem_df = this_dem_df.append(this_kin.iloc[start_i:stop_i + 1, :])
#         demo_list.append(this_dem_df.to_numpy())
#
#     demo_list_proc = preprocess_demos(demo_list,['standardize'])
#
#     total_demo_df_proc = pd.DataFrame()
#     for i in range(0, demo_inds.shape[0]):
#
#         demo_df_proc = pd.DataFrame(demo_list_proc[i])  # columns unknown depending on features used
#         demo_df_proc['task'] = [demo_inds[i][0]] * demo_list_proc[i].shape[0]
#         demo_df_proc['id'] = [demo_inds[i][1]] * demo_list_proc[i].shape[0]
#         demo_df_proc['i'] = range(0, demo_list_proc[i].shape[0])
#         demo_df_proc = demo_df_proc.set_index(['task', 'id', 'i'])
#
#         total_demo_df_proc = total_demo_df_proc.append(demo_df_proc)
#
#     metrics_df = metrics.compute_intrinsic_metrics(total_demo_df_proc,ann_df_copy)
#     return metrics_df


def run_evaluation_all(results_path="./results/"):
    global segment_colors_dict
    kin_df, ann_df = load_jigsaws_data_from_pkl()
    kin_df = kin_df[constants.slave_cols_quat]

    segment_colors_dict = plot_results.get_segment_colors_dict(ann_df)
    if os.path.exists(results_path) == False:
        os.mkdir(results_path)
    if os.path.exists(results_path+"plots/") == False:
        os.mkdir(results_path+"plots/")
    for cur_task in constants.tasks:
        run_task_evaluation(kin_df,ann_df,cur_task,results_path,results_path+"plots/")


# leave one out
def run_task_evaluation(kin_df,ann_df,task,save_filepath,plot_filepath):
    leave_out = []
    task_kin_df = kin_df.loc[task,:].copy()
    task_ann_df = ann_df.loc[task,:].copy()
    task_ids = pd.unique(task_kin_df.index.droplevel(level=1))

    avg_feature_metrics_df = pd.DataFrame()
    avg_gt_metrics_df = pd.DataFrame()
    avg_gmm_seg_metrics_df = pd.DataFrame()
    for it in range(0,len(constants.val_seeds)):
        # get validation sample (leave 1 out)
        np.random.seed(constants.val_seeds[it])
        task_val_inds = np.random.choice(range(0,task_ids.size),(10,),replace=False)

        task_val_kin_df = task_kin_df.loc[task_ids[task_val_inds],:].copy()
        task_val_ann_df = task_ann_df.loc[task_ids[task_val_inds],:].copy()
        task_val_kin_df['task']= [task]*task_val_kin_df.shape[0]
        task_val_ann_df['task'] = [task]*task_val_ann_df.shape[0]
        task_val_kin_df = task_val_kin_df.set_index(['task'],append=True)
        task_val_ann_df = task_val_ann_df.set_index(['task'],append=True)
        task_val_kin_df = task_val_kin_df.reorder_levels(['task', 'id', 'i'],)
        task_val_ann_df = task_val_ann_df.reorder_levels(['task', 'id', 'segment_no'],)

        # gt_metrics_df = run_gt_evaluation(task_val_kin_df,task_val_ann_df)
        # if avg_gt_metrics_df.size == 0:
        #     avg_gt_metrics_df = gt_metrics_df
        # else:
        #     avg_gt_metrics_df += gt_metrics_df

        # feature_metrics_df = run_feature_evaluation(task_val_kin_df,task_val_ann_df)
        # if avg_feature_metrics_df.size == 0:
        #     avg_feature_metrics_df = feature_metrics_df
        # else:
        #     avg_feature_metrics_df += feature_metrics_df

        gmm_seg_metrics_df = run_gmm_seg_evaluation(task_val_kin_df,task_val_ann_df)
        if avg_feature_metrics_df.size == 0:
            avg_gmm_seg_metrics_df = gmm_seg_metrics_df
        else:
            avg_gmm_seg_metrics_df += gmm_seg_metrics_df
    # avg_gt_metrics_df = avg_gt_metrics_df.div(len(constants.val_seeds))
    # metrics.save_intrinsic_metrics(avg_gt_metrics_df, save_filepath +task+ "_gt")
    # avg_feature_metrics_df = avg_feature_metrics_df.div(len(constants.val_seeds))
    # metrics.save_intrinsic_metrics(avg_feature_metrics_df, save_filepath+task+"_features")
    avg_gmm_seg_metrics_df = avg_gmm_seg_metrics_df.div(len(constants.val_seeds))
    metrics.save_intrinsic_metrics(avg_gmm_seg_metrics_df, save_filepath+task+"_gmm_seg_prune")


def run_gmm_seg_evaluation(kin_df, ann_df):
    if not os.path.exists("./results/plots/gmm_seg_eval_prune/"):
        os.mkdir("./results/plots/gmm_seg_eval_prune/")

    seg_funcs_dict = segmentation.get_segmentation_functions('gmm_')

    prune = True

    demo_df = kin_df[feature_experiments['pos']]

    preprocess_funcs = preprocess_default + ['low_pass_5', 'window_slds_2']
    exp_metrics_df = pd.DataFrame()
    exp_seg_dfs = []
    for seg_key in seg_funcs_dict.keys():
        exp_name = seg_key
        seg_df,demo_df_proc = segmentation.segment_demos(demo_df,ann_df,seg_funcs_dict[seg_key],preprocess_funcs,prune)
        metrics_df = metrics.compute_intrinsic_metrics(demo_df_proc,seg_df)

        metrics_df = metrics_df.rename(columns={0:exp_name})
        exp_metrics_df = pd.concat((exp_metrics_df,metrics_df),axis=1)
        exp_seg_dfs.append(seg_df)

    demo_inds  = pd.unique(demo_df.index.droplevel(level=2))
    for i in range(0,demo_inds.shape[0]):
        seg_df_list = []
        this_demo_ann = ann_df.loc[demo_inds[i]].copy()
        start_i = this_demo_ann.iloc[0]["i0"] - 1
        stop_i = this_demo_ann.iloc[-1]["i1"] - 1
        for j in range(0,len(gmm_func_plot_labels)):
            seg_df_list.append(exp_seg_dfs[j].loc[demo_inds[i],["i0","i1"]] + start_i)

        plot_results.plot_demo_seg_results(demo_df.loc[demo_inds[i]], ann_df.loc[demo_inds[i]], demo_inds[i],
                                           seg_df_list, gmm_func_plot_labels, segment_colors_dict,
                                           savedir = "./results/plots/gmm_seg_eval_prune/" + demo_inds[i][0] + "_")

    return exp_metrics_df


def run_feature_evaluation(kin_df, ann_df):
    seg_funcs_dict = segmentation.get_segmentation_functions()
    prune = False

    exp_metrics_df = pd.DataFrame()
    for exp_key in feature_experiments.keys():
        demo_df = kin_df[feature_experiments[exp_key]]

        for i in range(0, 3):
            preprocess_funcs = gmm_preprocess_experiments[i]
            exp_name = exp_key + "_"
            for pp_f in preprocess_funcs:
                exp_name += pp_f + "_"
            preprocess_funcs = preprocess_default + preprocess_funcs
            seg_func = 'gmm_seg_k_fixed'

            seg_df,demo_df_proc = segmentation.segment_demos(demo_df,ann_df,seg_funcs_dict[seg_func],preprocess_funcs,prune)
            metrics_df = metrics.compute_intrinsic_metrics(demo_df_proc,seg_df)

            metrics_df = metrics_df.rename(columns={0:exp_name})
            exp_metrics_df = pd.concat((exp_metrics_df,metrics_df),axis=1)

    return exp_metrics_df


def save_segmentation_df(seg_df,save_filename):
    pd.to_pickle(seg_df,save_filename + "_segmentation.pkl")




def evaluate_seg_results():
    results_index_df = load_results_index()
    results_index_df = results_index_df.reset_index().drop(["index"],axis=1)
    results_index_df = results_index_df.set_index(["task", "skill", "lp", "win", "seg_func","cv_fold"])
    cv_exps = np.unique(results_index_df.index.droplevel(level=5))
    orig_features = pd.read_pickle("./results/data/features/features_Suturing_E_lp1.5_dec3_std.pkl")
    cv_metrics_df = pd.DataFrame()
    gt_metrics_df = pd.DataFrame()
    seg_dfs_list = []
    seg_df_labels = []
    data_df_best_j =[]
    ann_df_best_j = []
    for i in range(0,cv_exps.shape[0]):
        this_cv_exp_index = results_index_df.loc[cv_exps[i]]
        features_dfs = load_features_from_exp_name(this_cv_exp_index.iloc[0]["exp_name"])
        ann_dfs  = load_annotations_from_exp_name(this_cv_exp_index.iloc[0]["exp_name"])



        seg_dfs_cv_list = []
        data_dfs_cv_list = []
        ann_dfs_cv_list = []
        best_j = 4
        for j in range(this_cv_exp_index.shape[0]):
            seg_dfs = load_segmentation_from_pkl(this_cv_exp_index.iloc[j]["exp_name"])
            demo_inds = np.unique(seg_dfs.index.droplevel(level=2))

            this_ann_df = pd.DataFrame()
            this_feat_df = pd.DataFrame()
            for k in range(0,demo_inds.shape[0]):
                start_i = int(seg_dfs.xs(demo_inds[k],drop_level=False).iloc[0,0])
                end_i = int(seg_dfs.xs(demo_inds[k],drop_level=False).iloc[-1,1]+1)
                this_feat_df = this_feat_df.append(features_dfs.xs(demo_inds[k],drop_level=False).iloc[start_i:end_i])
                seg_dfs.loc(axis=0)[demo_inds[k]].iloc[-1,1] =  ann_dfs.loc(axis=0)[demo_inds[k]].iloc[-1,1]
                seg_dfs.loc(axis=0)[demo_inds[k]].iloc[0,0] =  ann_dfs.loc(axis=0)[demo_inds[k]].iloc[0,0]
                this_ann_df =this_ann_df.append(ann_dfs.xs(demo_inds[k],drop_level=False))
                extrinsic_metrics_df = metrics.compute_extrinsic_metrics(features_dfs.xs(demo_inds[k],drop_level=False), seg_dfs.xs(demo_inds[k],drop_level=False), ann_dfs.xs(demo_inds[k],drop_level=False))
            metric_df = metrics.compute_metrics(this_feat_df,seg_dfs,this_ann_df)
            metric_df["exp_name"] = this_cv_exp_index.iloc[j]["exp_name"]
            col_list = ["task", "skill", "lp", "win", "seg_func","cv_fold"]
            for col in range(len(cv_exps[i])):
                metric_df[col_list[col]] = cv_exps[i][col]
            metric_df["cv_fold"]  = int(j)
            metric_df= metric_df.set_index(col_list)
            cv_metrics_df = cv_metrics_df.append(metric_df.copy()   )

            seg_dfs_cv_list.append(seg_dfs)
            data_dfs_cv_list.append(this_feat_df)
            ann_dfs_cv_list.append(this_ann_df)

            metric_df = metrics.compute_intrinsic_metrics(this_feat_df,this_ann_df)
            gt_metrics_df  = gt_metrics_df.append(metric_df.copy())


        seg_dfs_list.append(seg_dfs_cv_list[best_j])
        seg_df_labels.append(cv_exps[i][4])
        data_df_best_j = data_dfs_cv_list[best_j]
        ann_df_best_j = ann_dfs_cv_list[best_j]


    demo_inds =  np.unique(seg_dfs_list[0].index.droplevel(level=2))
    for ind in demo_inds:
        this_seg_list = []
        for i in range(0,len(seg_dfs_list)):
            this_seg_list.append(seg_dfs_list[i].loc[ind])
        plot_results.plot_gt_px_map_seg_v_time(orig_features.loc[ind],ann_df_best_j.loc[ind],ind,savedir="./results/plots/presentation/")

        plot_results.plot_demo_seg_results(data_df_best_j.loc[ind],ann_df_best_j.loc[ind],ind,this_seg_list,seg_df_labels,"./results/plots/presentation/")
    cv_metrics_df.to_csv("./results/data/metrics.csv")
    gt_metrics_df.to_csv("./results/data/gt.csv")


if __name__ == "__main__":
    evaluate_seg_results()


