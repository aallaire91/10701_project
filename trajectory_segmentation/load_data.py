import pandas as pd
import numpy as np
import os
import constants
from scipy.spatial.transform import Rotation as R
from preprocessing import preprocess_demos_df
from os import listdir
from os.path import isfile, join

def save_jigsaws_data_to_pkl():
    '''

    :param filepath: path to jigsaws data directory
    :type filepath: str
    :param load_pkl: load from pkl instead of raw data if available
    :type load_pkl: bool
    :param save_pkl: saves raw data as pkl
    :type save_pkl: bool
    :return:
    :rtype: pd.DataFrame
    '''

    annotation_types_dict = dict(zip(constants.annotation_cols, constants.annotation_types))
    kin_df = pd.DataFrame()
    annotations_df = pd.DataFrame()

    for task in constants.tasks:
        kin_path_full = os.path.join(constants.source_filepath, task, constants.kin_path)
        annotations_path_full = os.path.join(constants.source_filepath, task, constants.annotations_path)

        demo_files = os.listdir(annotations_path_full)
        for file in demo_files:
            id = file.split("_")[-1][:-4]

            # this_kin_df = pd.read_csv(os.path.join(kin_path_full, file), delim_whitespace=True, header=None,
            #                           names=constants.kin_cols, usecols=constants.kin_cols, dtype=np.float64)
            #
            # this_kin_df['task'] = [task] * this_kin_df.shape[0]
            # this_kin_df['id'] = [id] * this_kin_df.shape[0]
            # this_kin_df['i'] = range(0, this_kin_df.shape[0])
            # this_kin_df = this_kin_df.set_index(['task', 'id', 'i'])

            #
            this_ann_df = pd.read_csv(os.path.join(annotations_path_full, file), delim_whitespace=True, header=None,
                                      names=constants.annotation_cols, dtype=annotation_types_dict)
            this_ann_df['task'] = [task] * this_ann_df.shape[0]
            this_ann_df['id'] = [id] * this_ann_df.shape[0]
            this_ann_df['segment_no'] = range(0, this_ann_df.shape[0])
            this_ann_df = this_ann_df.set_index(['task', 'id', 'segment_no'])





            # kin_df = kin_df.append(this_kin_df)
            annotations_df = annotations_df.append(this_ann_df)
    annotations_df.loc[:,["i0","i1"]] -= 1
    kin_df.to_pickle(os.path.join(constants.pkl_path, constants.kin_pkl_filename))
    annotations_df.to_pickle(os.path.join(constants.pkl_path, constants.annotations_pkl_filename))

    return kin_df, annotations_df

def create_jigsaws_kin_quat_pkl():
    kin_df,ann_df = load_jigsaws_data_from_pkl()
    kin_df_quat = convert_R_features_to_quat(kin_df)
    kin_df_quat.to_pickle(os.path.join(constants.pkl_path, constants.kin_quat_pkl_filename))
    return kin_df_quat

def end_effector_dist(demo_df):
    demo_df_proc = demo_df.copy()
    dx = demo_df["RH_x"] - demo_df["LH_x"]
    dy = demo_df["RH_y"] - demo_df["LH_y"]
    dz = demo_df["RH_z"] - demo_df["LH_z"]
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    demo_df_proc["EE_dx"] = dx
    demo_df_proc["EE_dy"] = dy
    demo_df_proc["EE_dz"] = dz
    demo_df_proc["EE_d"] = d
    return demo_df_proc

def sig3d(ps):
    pm2 = 1e4*ps[0].astype(np.float64)
    pm1 = 1e4*ps[1].astype(np.float64)
    p = 1e4*ps[2].astype(np.float64)
    pp1 = 1e4*ps[3].astype(np.float64)
    pp2 = 1e4*ps[4].astype(np.float64)

    a = np.linalg.norm(pm1 - p,axis=1)
    b = np.linalg.norm(pp1 - p,axis=1)
    c = np.linalg.norm(pm2 - p,axis=1)
    d = np.linalg.norm(pp2 - pp1,axis=1)
    e = np.linalg.norm(pp2 - p,axis=1)
    f = np.linalg.norm(pp2 - pm1,axis=1)
    g = np.linalg.norm(pm2 - pm1,axis=1)
    s = (a+b+c)/2.0
    h = np.divide(2.0*np.sqrt(s*np.fabs(s-a)*np.fabs(s-b)*np.fabs(s-c)),c,out=np.zeros_like(s),where=c!=0)
    m = np.linalg.norm(pp1 - pm2,axis=1)
    n = np.linalg.norm(p - pm2,axis=1)



    delta_abc = np.sqrt(s*np.fabs(s-a)*np.fabs(s-b)*np.fabs(s-c))
    curvature = np.divide(4*delta_abc,(a*b*c),out=np.zeros_like(delta_abc),where=(a*b*c)!=0)
    Hp = np.array([])
    Hm = np.array([])
    for i in range(0,p.shape[0]):
        Vp = np.hstack((p[i,:],[1]))
        Vp = np.vstack((Vp,np.hstack((pm1[i,:],[1]))))
        Vp = np.vstack((Vp,np.hstack((pp1[i,:],[1]))))
        Vp = np.vstack((Vp,np.hstack((pp2[i,:],[1]))))
        Vp = np.linalg.det(Vp)/6.0

        hp = Vp*3.0/delta_abc[i] if np.fabs(delta_abc[i])>1e-10 else 0

        Vm = np.hstack((p[i,:],[1]))
        Vm = np.vstack((Vm,np.hstack((pm2[i,:],[1]))))
        Vm = np.vstack((Vm,np.hstack((pm1[i,:],[1]))))
        Vm = np.vstack((Vm,np.hstack((pp1[i,:],[1]))))
        Vm = np.linalg.det(Vm)/6

        hm = Vm*3.0/delta_abc[i] if np.fabs(delta_abc[i])>1e-10 else 0

        Hp = np.concatenate((Hp,[hp]),axis=0)
        Hm = np.concatenate((Hm,[hm]),axis=0)

    term1 = np.divide(6*Hp,(d*e*f*curvature),out=np.zeros_like(6*Hp), where=(d*e*f*curvature)!=0)
    term2 = np.divide(6*Hm,(g*n*m*curvature),out=np.zeros_like(6*Hm), where=(g*n*m*curvature)!=0)
    torsion =0.5*(term1 + term2)
    r = 2*a[1:-1] + 2*b[1:-1] - 2*d[1:-1] - 3*h[1:-1] + g[1:-1]

    dcurvature = np.divide(3*(curvature[2:] - curvature[0:-2]),(2*a[1:-1] + 2*b[1:-1] + d[1:-1] + g[1:-1]),out=np.zeros_like(3*(curvature[2:] - curvature[0:-2])),where=(2*a[1:-1] + 2*b[1:-1] + d[1:-1] + g[1:-1])!=0)
    dtorsion = 4*(torsion[2:] - torsion[0:-2] + r*np.divide(torsion[1:-1]*dcurvature,(6*curvature[1:-1]),out = np.zeros_like(torsion[1:-1]*dcurvature),where=(6*curvature[1:-1])!=0))
    dtorsion = np.divide(dtorsion,(2*a[1:-1] + 2*b[1:-1] + 2*d[1:-1] + h[1:-1] + g[1:-1]),out=np.zeros_like(dtorsion),where= (2*a[1:-1] + 2*b[1:-1] + 2*d[1:-1] + h[1:-1] + g[1:-1])!=0)

    return curvature[1:-1],torsion[1:-1],dcurvature,dtorsion


def invariant_euclidean_signature(demo_df,ann_df):
    pm2 = (demo_df.iloc[0:-4]).loc(axis=1)[["LH_x","LH_y","LH_z"]].values
    pm1 = (demo_df.iloc[1:-3]).loc(axis=1)[["LH_x","LH_y","LH_z"]].values
    p = (demo_df.iloc[2:-2]).loc(axis=1)[["LH_x","LH_y","LH_z"]].values
    pp1 = (demo_df.iloc[3:-1]).loc(axis=1)[["LH_x","LH_y","LH_z"]].values
    pp2  = (demo_df.iloc[4:]).loc(axis=1)[["LH_x","LH_y","LH_z"]].values

    curvature,torsion,dcurvature,dtorsion = sig3d([pm2,pm1,p,pp1,pp2])

    demo_df_proc = demo_df.iloc[3:-3].copy()
    demo_df_proc["LH_kappa"] = curvature
    demo_df_proc["LH_tau"] = torsion
    demo_df_proc["LH_dkappa"] = dcurvature
    demo_df_proc["LH_dtau"] = dtorsion

    pm2 = (demo_df.iloc[0:-4]).loc(axis=1)[["RH_x", "RH_y", "RH_z"]].values
    pm1 = (demo_df.iloc[1:-3]).loc(axis=1)[["RH_x", "RH_y", "RH_z"]].values
    p = (demo_df.iloc[2:-2]).loc(axis=1)[["RH_x", "RH_y", "RH_z"]].values
    pp1 = (demo_df.iloc[3:-1]).loc(axis=1)[["RH_x", "RH_y", "RH_z"]].values
    pp2 = (demo_df.iloc[4:]).loc(axis=1)[["RH_x", "RH_y", "RH_z"]].values

    curvature, torsion, dcurvature, dtorsion = sig3d([pm2, pm1, p, pp1, pp2])
    demo_df_proc["RH_kappa"] = curvature
    demo_df_proc["RH_tau"] = torsion
    demo_df_proc["RH_dkappa"] = dcurvature
    demo_df_proc["RH_dtau"] = dtorsion

    ann_df_proc = ann_df.copy()
    ann_df_proc.iloc[:,0:2] -= 3
    ann_df_proc.iloc[-1,1] -= 3
    if ann_df_proc.iloc[0,0] < 0:
        ann_df_proc.iloc[0, 0] = 0
    return demo_df_proc,ann_df_proc

def create_jigsaws_extra_features_pkl():
    kin_df, ann_df = load_jigsaws_data_from_pkl()
    demo_inds = np.unique(kin_df.index.droplevel(level=2))
    kin_df_proc = pd.DataFrame()
    ann_df_proc = pd.DataFrame()
    for ind in demo_inds:
        demo_df_proc,this_ann_df_proc = invariant_euclidean_signature(kin_df.xs(ind,drop_level=False), ann_df.xs(ind,drop_level=False))
        kin_df_proc = kin_df_proc.append(demo_df_proc)
        ann_df_proc = ann_df_proc.append(this_ann_df_proc)

    kin_df_proc = end_effector_dist(kin_df_proc)
    kin_df_proc.to_pickle(os.path.join(constants.pkl_path,constants.kin_quat_extra_feat_pkl_filename))
    ann_df_proc.to_pickle(os.path.join(constants.pkl_path,constants.annotations_extra_feat_pkl_filename))

    return kin_df_proc,ann_df_proc

def load_jigsaws_data_from_pkl(feature_type='quat',pkl_path=constants.pkl_path):
    annotations_df = pd.read_pickle(os.path.join(pkl_path,constants.annotations_pkl_filename ))
    kin_df = pd.read_pickle(os.path.join(pkl_path, constants.feature_pkl_filenames[feature_type]))
    return kin_df,annotations_df

def convert_R_features_to_quat(kin_df):
    new_df = kin_df.apply(convert_Rcols_to_quat, args=[constants.slave_lh_R_cols, constants.slave_lh_R_cols], axis=1)

    # flip quaternions that are too far

    new_df.loc[new_df.index[1:new_df.shape[0] - 1],constants.slave_lh_R_cols[:4]] = avoid_quat_jumps(
        new_df.loc[new_df.index[1:new_df.shape[0] - 1],constants.slave_lh_R_cols[:4]],
        new_df.loc[new_df.index[0:new_df.shape[0] - 2], constants.slave_lh_R_cols[:4]])
    new_df.loc[new_df.index[1:new_df.shape[0] - 1],constants.slave_rh_R_cols[:4]] = avoid_quat_jumps(
        new_df.loc[new_df.index[1:new_df.shape[0] - 1],constants.slave_rh_R_cols[:4]],
        new_df.loc[new_df.index[0:new_df.shape[0] - 2],constants.slave_rh_R_cols[:4]])
    lh_rm_dict = dict(zip(constants.slave_lh_R_cols[0:4], ['LH_qx', 'LH_qy', 'LH_qz', 'LH_qw']))
    rh_rm_dict = dict(zip(constants.slave_rh_R_cols[0:4], ['RH_qx', 'RH_qy', 'RH_qz', 'RH_qw']))
    new_df = new_df.rename(columns=lh_rm_dict)
    new_df = new_df.rename(columns=rh_rm_dict)
    new_df = new_df.drop(columns=constants.slave_lh_R_cols[4:])
    new_df = new_df.drop(columns=constants.slave_rh_R_cols[4:])
    return new_df

def convert_Rcols_to_quat(kin_df,lh_R_cols,rh_R_cols):

    lh_R = kin_df[lh_R_cols].values.reshape((3, 3))
    r = R.from_matrix(lh_R)
    lh_quat = r.as_quat()
    kin_df.loc['LH_R00'] = lh_quat[0]
    kin_df.loc[ 'LH_R01'] = lh_quat[1]
    kin_df.loc[ 'LH_R02'] = lh_quat[2]
    kin_df.loc[ 'LH_R10'] = lh_quat[3]

    rh_R = kin_df[rh_R_cols].values.reshape((3, 3))
    r = R.from_matrix(rh_R)
    rh_quat = r.as_quat()
    kin_df.loc[ 'RH_R00'] = rh_quat[0]
    kin_df.loc[ 'RH_R01'] = rh_quat[1]
    kin_df.loc['RH_R02'] = rh_quat[2]
    kin_df.loc['RH_R10'] = rh_quat[3]

    return kin_df

def avoid_quat_jumps(quat_cols_next, quat_cols_prev):
    quat_cols_prev_np = quat_cols_prev.to_numpy()
    quat_cols_next_np = quat_cols_next.to_numpy()
    neg_mask = np.linalg.norm(quat_cols_prev_np - quat_cols_next_np, axis=1) > np.linalg.norm(
        quat_cols_prev_np + quat_cols_next_np, axis=1)
    neg_inds = np.where(neg_mask)[0] + 1
    for i in range(0, len(neg_inds), 2):
        if i + 1 >= len(neg_inds):
            quat_cols_next.iloc[neg_inds[i]:] = -quat_cols_next.iloc[neg_inds[i]:]
        else:
            quat_cols_next.iloc[neg_inds[i]:neg_inds[i + 1] - 1] = -quat_cols_next.iloc[neg_inds[i]:neg_inds[i + 1] - 1]

    return quat_cols_next

def load_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs=['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None}):
    kin_filename, ann_filename = get_preprocessed_filenames(feature_type,pp_funcs,pp_func_params)
    if os.path.exists(os.path.join(pkl_path,kin_filename)) == False:
        save_preprocessed_pkl(feature_type,pkl_path,pp_funcs,pp_func_params)

    kin_df = pd.read_pickle(os.path.join(pkl_path,kin_filename))
    ann_df = pd.read_pickle(os.path.join(pkl_path,ann_filename))
    return kin_df, ann_df



def save_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs=['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None}):
    kin_df,ann_df = load_jigsaws_data_from_pkl(feature_type,pkl_path)
    kin_df_proc = preprocess_demos_df(kin_df,pp_funcs,pp_func_params)

    kin_filename, ann_filename = get_preprocessed_filenames(feature_type,pp_funcs,pp_func_params)
    inds = np.unique(kin_df.index.droplevel(level=2))
    if kin_df_proc.shape[0] != kin_df.shape[0]:
        for ind in inds:
            this_ann_df = ann_df.loc[ind].copy()
            this_ann_df.iloc[:, 1] = (this_ann_df.iloc[:, 1] // 3).astype(int)
            this_ann_df.iloc[0, 0] = (this_ann_df.iloc[0, 0] // 3).astype(int)
            this_ann_df.iloc[1:, 0] = (this_ann_df.iloc[0:-1, 1].values + 1).astype(int)
            this_ann_df.iloc[-1, 1] = kin_df_proc.loc[ind, :].shape[0]
            ann_df.loc[ind] = this_ann_df.values
    kin_df_proc.to_pickle(os.path.join(pkl_path,kin_filename))
    ann_df.to_pickle(os.path.join(pkl_path,ann_filename))
    return kin_df_proc,ann_df


def get_preprocessed_filenames(feature_type,pp_funcs=['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None}):
    ann_filename = constants.annotation_pkl_filenames[feature_type][:-4]
    kin_filename = constants.feature_pkl_filenames[feature_type][:-4]
    suffix = ""
    for func_name in pp_funcs:
        suffix += "_" + func_name
        if pp_func_params[func_name] is not None:
            suffix += str(pp_func_params[func_name])
    suffix += ".pkl"
    ann_filename += suffix
    kin_filename += suffix
    return kin_filename, ann_filename

def get_pp_suffix(pp_funcs,pp_func_params):
    suffix = ""
    for func_name in pp_funcs:
        suffix += "_" + func_name
        if pp_func_params[func_name] is not None:
            suffix += str(pp_func_params[func_name])
    return suffix

def load_results_index(substr=""):
    mypath = "./results/data/segmentations/"
    df = pd.read_csv("./results/results1.csv").values
    files = df[:,1] + ["/"]*df.shape[0] + df[:,0]
    files = [f for f in files if ((substr in f)or(substr==""))]
    results_index_df = pd.DataFrame()
    for f in files:
        results_index_df = results_index_df.append(get_results_index_entry(f.split("/")[-1]))
    return results_index_df


def load_features_index(substr=""):
    mypath = "./results/data/features/"
    files = [f for f in listdir(mypath) if (isfile(join(mypath, f))and ((substr in f)or(substr=="")))]
    features_index_df = pd.DataFrame()
    for f in files:
        features_index_df = features_index_df.append(get_features_index_entry(f))
    return features_index_df


def get_features_index_entry(filename):
    filename_split = filename.split("_")
    task = filename_split[1]
    i =2


    skill = filename_split[i]
    lp = -1
    win = 1

    i += 1
    while i < len(filename_split):
        if "lp" in filename_split[i]:
            lp = float(filename_split[i][2:])
            i= i+3
        elif "win" in filename_split[i]:
            win = int(filename_split[i][3:])
            i=i+1
        else:
            i=i+1

    feature_name = "_"
    feature_name = feature_name.join(filename_split[1:])[:-4]

    feature_index_df = pd.DataFrame(np.array([[task,skill,lp,win,feature_name]]),columns=["task","skill","lp","win","feature_name"])
    return feature_index_df

def get_results_index_entry(filename):
    filename_split = filename.split("_")
    seg_func = ""
    cv_fold = 0
    i = 4

    while i < len(filename_split):
        if "kpca" in filename_split[i]:
            seg_func = filename_split[i+1] + "_" +filename_split[i+2]
            i=i+3
        elif "cv" in filename_split[i]:
            cv_fold = int(filename_split[i][2:-4])
            i=i+1
        else:
            i=i+1
    exp_name = "_"
    exp_index_df = get_features_index_entry(exp_name.join(filename_split[0:-3]))
    exp_index_df = exp_index_df.drop(["feature_name"],axis=1)
    exp_name = exp_name.join(filename_split[1:])[:-4]

    exp_index_df = pd.concat((exp_index_df,pd.DataFrame(np.array([[seg_func,cv_fold,exp_name]]),columns=["seg_func","cv_fold","exp_name"])),axis=1)
    return exp_index_df

def load_features_from_exp_name(exp_name):
    filename_split = exp_name.split("_")
    i =3


    while i < len(filename_split):
        if "kpca" in filename_split[i]:
            break
        else:
            i=i+1
    feature_name = "_"
    feature_name = feature_name.join(filename_split[0:i+1])

    features_df = load_features_from_pkl(feature_name)
    return features_df


def load_annotations_from_exp_name(exp_name):
    filename_split = exp_name.split("_")
    if filename_split[0][0] == 'S':
        i = 3
    else:
        i = 4

    feature_name = "_"
    feature_name = feature_name.join(filename_split[0:i])

    annotations_df = load_annotations_from_pkl(feature_name)
    return annotations_df

def load_segmentation_from_pkl(substr):
    mypath = "./results/data/segmentations/"
    for f in listdir(mypath):
        if (isfile(join(mypath, f))and ((substr in f))):
            return pd.read_pickle(os.path.join(mypath,f))

def load_features_from_pkl(substr):
    mypath = "./results/data/features/"
    for f in listdir(mypath):
        if (isfile(join(mypath, f))and ((substr in f))):
            return pd.read_pickle(os.path.join(mypath,f))

def load_annotations_from_pkl(substr):
    mypath = "./results/data/annotations/"
    for f in listdir(mypath):
        if (isfile(join(mypath, f))and ((substr in f))):
            return pd.read_pickle(os.path.join(mypath,f))


if __name__ == "__main__":
    # save_jigsaws_data_to_pkl()
    # # create_jigsaws_kin_quat_pkl()
    # load_jigsaws_data_from_pkl()
    # save_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs=['lp','dec','std'],pp_func_params={'lp':5,'dec':3,'std':None})
    #
    # save_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs=['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None})
    save_preprocessed_pkl(feature_type='quat_eedist_3dsig',pkl_path=constants.pkl_path,pp_funcs=['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None})
    #results_index_df = load_results_index()
    #create_jigsaws_extra_features_pkl()
