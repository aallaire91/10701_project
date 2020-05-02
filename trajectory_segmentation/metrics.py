from sklearn import metrics
from sklearn import mixture
import numpy as np
import pandas as pd
from munkres import Munkres,make_cost_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import contingency_matrix


def compute_metrics(data_df,seg_df,ann_df):
    intrinsic_metrics = compute_intrinsic_metrics(data_df,seg_df)
    extrinsic_metrics = compute_extrinsic_metrics(data_df, seg_df, ann_df)
    metrics_df = pd.concat((extrinsic_metrics,intrinsic_metrics),axis=1)
    return metrics_df

def compute_intrinsic_metrics(data_df,seg_df):

    # trim data
    demo_inds = np.unique(data_df.index.droplevel(level=2))
    data_df_trimmed = pd.DataFrame()
    for i in range(0,demo_inds.shape[0]):
        this_demo_df = data_df.xs(demo_inds[i],drop_level=False)
        this_seg_df = seg_df.xs(demo_inds[i],drop_level=False)
        start_i = int(this_seg_df.iloc[0,0])
        end_i = int(this_seg_df.iloc[-1,1])
        data_df_trimmed = data_df_trimmed.append(this_demo_df.iloc[start_i:end_i+1])

    labels = convert_segments_to_labels(seg_df)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    ss = compute_silhouette_score(data_df_trimmed.to_numpy(),labels)
    dbi = compute_davies_bouldin_index(data_df_trimmed.to_numpy(),labels)


    metrics_df = pd.DataFrame([[ss,dbi]],columns=["si","dbi"] )

    return metrics_df


def save_intrinsic_metrics(metrics_df,save_filename):
    pd.to_pickle(metrics_df,save_filename + "_intrinsic_metrics.pkl")
    metrics_df.to_csv(save_filename + "_intrinsic_metrics.csv")


def compute_silhouette_score(data,pred_labels):
    # computes density
    # -1 bad, 1 good

    score = metrics.silhouette_score(data,pred_labels)
    return score


def compute_davies_bouldin_index(data,pred_labels):
    # 0 is good
    # measures cluster similarity
    score = metrics.davies_bouldin_score(data,pred_labels)
    return score


def compute_extrinsic_metrics(data_df,seg_df,ann_df):
    gt_labels = convert_segments_to_labels(ann_df)
    pred_labels = convert_segments_to_labels(seg_df)

    le = LabelEncoder()
    pred_labels = le.fit_transform(pred_labels)
    ari = compute_adjusted_rand_score(gt_labels, pred_labels)
    munk = munkres_score(gt_labels,pred_labels)
    metrics_df = pd.DataFrame([[ari,munk]],columns=["ari",'munkres'])
    return metrics_df


def compute_adjusted_rand_score(gt_labels, pred_labels):
    score = metrics.adjusted_rand_score(gt_labels, pred_labels)
    return score

def compute_contingency_matrix(gt_labels,pred_labels):
    return metrics.contingency_matrix(gt_labels, pred_labels)

def convert_segments_to_labels(seg_df):
    labels = []
    for i in range(0,seg_df.shape[0]):
        start_i = seg_df.iloc[i,0]
        stop_i = seg_df.iloc[i,1]

        labels += [seg_df.iloc[i,2]]*int(stop_i - start_i + 1)

    return labels


def split_data_into_cluster_dict(data_df,seg_df):
    demo_inds = pd.unique(data_df.index.droplevel(level=2))

    cluster_dict = {}
    for i in range(0,demo_inds.shape[0]):
        this_seg_df = seg_df.loc[demo_inds[i],:]
        this_data_df = data_df.loc[demo_inds[i],:]
        for j in range(0,this_seg_df.shape[0]):
            start_i = int(this_seg_df.iloc[j,0])
            stop_i = int(this_seg_df.iloc[j,1])
            cluster_dict[this_seg_df.iloc[j,2]] = {this_seg_df.iloc[j,2]:this_data_df.iloc[start_i:stop_i+1].to_numpy()}

    return cluster_dict


def map_pred_labels_to_gt(gt_label_df, pred_label_df,max_gt_label):
    primary_temporal_clustering = [int(l[1:])-1 for l in convert_segments_to_labels(gt_label_df)]
    secondary_temporal_clustering = convert_segments_to_labels(pred_label_df)
    # First make sure we relabel everything with 0-indexed, continuous cluster labels
    le2 = LabelEncoder()
    secondary_temporal_clustering = le2.fit_transform(secondary_temporal_clustering)
    le1 = LabelEncoder()
    primary_temporal_clustering = le1.fit_transform(primary_temporal_clustering)
    # Build out the contingency matrix
    mat = contingency_matrix(primary_temporal_clustering, secondary_temporal_clustering)

    # Create the cost matrix
    cost_mat = make_cost_matrix(mat, lambda x: len(primary_temporal_clustering) - x)

    # Apply the Hungarian method to find the optimal cluster correspondence
    m = Munkres()
    indexes = m.compute(cost_mat)

    # Create the correspondences between secondary clusters and primary labels
    correspondences = {b: a for a, b in indexes}

    # What're the labels in the primary and secondary clusterings
    primary_labels, secondary_labels = set(np.unique(primary_temporal_clustering)), set(
        np.unique(secondary_temporal_clustering))
    correspondences_orig = {}
    proposed_label = max_gt_label+1
    for label in secondary_labels:

        if label not in correspondences:
            correspondences_orig[le2.inverse_transform([0,label])[1] ]= 'G'+str(proposed_label)
            proposed_label += 1
        else:
            correspondences_orig[le2.inverse_transform([0,label])[1]] = 'G' + str(le1.inverse_transform([0,correspondences[label]])[1]+1)

    # # Relabel the temporal clustering
    # relabeled_secondary_temporal_clustering = [correspondences[e] for e in secondary_temporal_clustering]
    # le1.inverse_transform(self)
    return correspondences_orig

def munkres_score(gt, pred):
    """
    :param gt: a list of lists, each containing ints
    :param pred: a list of lists, each containing ints
    :return: accuracy
    """

    # Combine all the sequences into one long sequence for both gt and pred
    gt_combined = np.array(gt)
    pred_combined = np.array(pred)

    # Make sure we're comparing the right shapes
    assert(gt_combined.shape == pred_combined.shape)

    # Build out the contingency matrix
    # This follows the methodology suggested by Zhou, De la Torre & Hodgkins, PAMI 2013.
    mat = contingency_matrix(gt_combined, pred_combined)

    # We need to make the cost matrix
    # We use the fact that no entry can exceed the total length of the sequence
    cost_mat = make_cost_matrix(mat, lambda x: gt_combined.shape[0] - x)

    # Apply the Munkres method (also called the Hungarian method) to find the optimal cluster correspondence
    m = Munkres()
    indexes = m.compute(cost_mat)

    # Pull out the associated 'costs' i.e. the cluster overlaps for the correspondences we've found
    cluster_overlaps = mat[list(zip(*indexes))]

    # Now compute the accuracy
    accuracy = np.sum(cluster_overlaps)/float(np.sum(mat))

    return accuracy