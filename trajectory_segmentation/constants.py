import numpy as np
import preprocessing as pp
import segmentation as seg
from matplotlib.pyplot import cm



def get_colors(num_colors,cmap_name):
    colors = cm.get_cmap(cmap_name)(np.linspace(0, 1,num_colors))
    return colors

def get_segment_colors_dict_from_list(labels,cmap_name=['gist_ncar','twilight_shifted']):
    # np.random.seed(0)
    # np.random.shuffle(labels)
    colors = []
    if isinstance(cmap_name, list):
        colors_per_cmap = labels.size//len(cmap_name)
        colors += get_colors(labels.size - (colors_per_cmap*(len(cmap_name)-1)) +1 ,cmap_name[0]).tolist()[:-1]
        for i in range(1,len(cmap_name)):
            colors += get_colors(colors_per_cmap+1 ,cmap_name[i]).tolist()[:-1]

    else:
        colors = get_colors(labels.size,cmap_name)
    segment_colors_dict = dict(zip(labels,colors))
    return segment_colors_dict




val_seeds = range(0,100,10)
source_filepath = "../JIGSAWS/"
pkl_path = "./data/jigsaws/"
results_path = "./results/"
kin_path = "kinematics/AllGestures/"
annotations_path = "transcriptions/"
kin_pkl_filename = "kin_raw.pkl"
annotations_pkl_filename = "annotations.pkl"

kin_quat_pkl_filename = "kin_quat.pkl"
kin_quat_extra_feat_pkl_filename = "kin_quat_eedist_3dsig.pkl"
annotations_extra_feat_pkl_filename = "annotations_quat_eedist_3dsig.pkl"

feature_pkl_filenames = {'raw':kin_pkl_filename,
                         'quat':kin_quat_pkl_filename,
                         'quat_eedist_3dsig':kin_quat_extra_feat_pkl_filename}
annotation_pkl_filenames = {'raw':annotations_pkl_filename,
                         'quat':annotations_pkl_filename,
                         'quat_eedist_3dsig':annotations_extra_feat_pkl_filename}

paper_savefolder = "./results/paper/"

gesture_label_descr_map = {'G1': 'Reaching for needle with right hand',
                           'G2': 'Positioning needle',
                           'G3': "Pushing needle through tissue",
                           'G4': "Transferring needle from left to right",
                           'G5': "Moving to center with needle in grip" ,
                           'G6': "Pulling suture with left hand" ,
                           'G7': "Pulling suture with right hand" ,
                           'G8': "Orienting needle" ,
                           'G9': "Using right hand to help tighten suture" ,
                           'G10': "Loosening more suture" ,
                           'G11': "Dropping suture at end and moving to end points" ,
                           'G12': "Reaching for needle with left hand",
                           'G13': "Making C loop around right hand",
                           'G14': "Reaching for suture with right hand",
                           'G15': "Pulling suture with both hands",
                           'G16': "Unlabled"}

tasks = ["Knot_Tying", "Needle_Passing", "Suturing"]

kin_cols = ['MLH_x', 'MLH_y', 'MLH_z', 'MLH_R00', 'MLH_R01', 'MLH_R02', 'MLH_R10',
            'MLH_R11', 'MLH_R12', 'MLH_R20', 'MLH_R21', 'MLH_R22', 'MLH_vx',
            'MLH_vy', 'MLH_vz', 'MLH_wx', 'MLH_wy', 'MLH_wz', 'MLH_theta',
            'MRH_x', 'MRH_y', 'MRH_z', 'MRH_R00', 'MRH_R01', 'MRH_R02', 'MRH_R10',
            'MRH_R11', 'MRH_R12', 'MRH_R20', 'MRH_R21', 'MRH_R22', 'MRH_vx',
            'MRH_vy', 'MRH_vz', 'MRH_wx', 'MRH_wy', 'MRH_wz', 'MRH_theta',
            'LH_x', 'LH_y', 'LH_z', 'LH_R00', 'LH_R01', 'LH_R02', 'LH_R10',
            'LH_R11', 'LH_R12', 'LH_R20', 'LH_R21', 'LH_R22', 'LH_vx',
            'LH_vy', 'LH_vz', 'LH_wx', 'LH_wy', 'LH_wz', 'LH_theta',
            'RH_x', 'RH_y', 'RH_z', 'RH_R00', 'RH_R01', 'RH_R02', 'RH_R10',
            'RH_R11', 'RH_R12', 'RH_R20', 'RH_R21', 'RH_R22', 'RH_vx',
            'RH_vy', 'RH_vz', 'RH_wx', 'RH_wy', 'RH_wz', 'RH_theta'
            ]

segment_colors_dict = get_segment_colors_dict_from_list(np.array(list(gesture_label_descr_map.keys())[:-1]))
segment_colors_dict["G16"] = [0,0,0,1]

kin_types = [np.float32]*len(kin_cols)

annotation_cols = ["i0", "i1", "label"]
annotation_types = [np.int32, np.int32, np.str]

master_cols = kin_cols[0:38]
master_lh_R_cols = master_cols[3:12]
master_rh_R_cols = master_cols[22:31]
master_pos_cols = master_cols[0:12]+[master_cols[18]]
master_vel_cols = master_cols[12:19]+ master_cols[31:38]

slave_cols = kin_cols[38:]
slave_lh_R_cols = slave_cols[3:12]
slave_rh_R_cols = slave_cols[22:31]
slave_pos_cols = slave_cols[0:12]+[slave_cols[18]] + slave_cols[19:31]+[slave_cols[37]]
slave_vel_cols = slave_cols[12:19] + slave_cols[31:38]


slave_cols_quat = [ 'LH_x', 'LH_y',
       'LH_z', 'LH_qx', 'LH_qy', 'LH_qz', 'LH_qw', 'LH_vx', 'LH_vy', 'LH_vz',
       'LH_wx', 'LH_wy', 'LH_wz', 'LH_theta', 'RH_x', 'RH_y', 'RH_z', 'RH_qx',
       'RH_qy', 'RH_qz', 'RH_qw', 'RH_vx', 'RH_vy', 'RH_vz', 'RH_wx', 'RH_wy',
       'RH_wz', 'RH_theta']
slave_pos_cols_quat = [ 'LH_x', 'LH_y',
       'LH_z', 'LH_qx', 'LH_qy', 'LH_qz', 'LH_qw', 'LH_vx', 'LH_theta', 'RH_x', 'RH_y', 'RH_z', 'RH_qx',
       'RH_qy', 'RH_qz', 'RH_qw', 'RH_theta']
slave_vel_cols_quat = [  'LH_vx', 'LH_vy', 'LH_vz',
       'LH_wx', 'LH_wy', 'LH_wz', 'LH_theta', 'RH_vx', 'RH_vy', 'RH_vz', 'RH_wx', 'RH_wy',
       'RH_wz', 'RH_theta']

skill_map = {"B001":"N",
             "B002":"N",
             "B003":"N",
             "B004":"N",
             "B005":"N",
             "C001":"I",
             "C002":"I",
             "C003":"I",
             "C004":"I",
             "C005":"I",
             "D001":"E",
             "D002":"E",
             "D003":"E",
             "D004":"E",
             "D005":"E",
             "E001":"E",
             "E002":"E",
             "E003":"E",
             "E004":"E",
             "E005":"E",
             "F001":"I",
             "F002":"I",
             "F003":"I",
             "F004":"I",
             "F005":"I",
             "G001":"N",
             "G002":"N",
             "G003":"N",
             "G004":"N",
             "G005":"N",
             "H001":"N",
             "H002":"N",
             "H003":"N",
             "H004": "N",
             "H005": "N",
             "I001": "N",
             "I002": "N",
             "I003": "N",
             "I004": "N",
             "I005": "N"}
ee_dist_cols = ['EE_d','EE_dx','EE_dy','EE_dz']
sig3d_cols =  ['LH_kappa', 'LH_dkappa', 'LH_tau', 'LH_dtau','RH_kappa', 'RH_dkappa', 'RH_tau', 'RH_dtau']
feat_columns_map = {'pos': slave_pos_cols_quat,
                    'pos_vel':slave_cols_quat,
                    'pos_eedist': slave_pos_cols_quat + ee_dist_cols,
                    # 'pos_eedist_3dsig': slave_pos_cols_quat + ee_dist_cols + sig3d_cols,
                    'pos_vel_eedist':slave_cols_quat + ee_dist_cols,
                    # 'pos_vel_eedist_3dsig': slave_cols_quat + ee_dist_cols + sig3d_cols,
                    }


seg_func_label_map = {'gmm_dp': seg.gmm_seg_k_dp,
                      'gmm_bic': seg.gmm_seg_k_bic,
                      'gmm_gmeans':seg.gmm_seg_k_gmeans,
                      'gmm_fixed':seg.gmm_seg_k_fixed,
                      'hdp_hsmm': seg.hdp_hsmm_seg}


pp_func_label_map = {'lp':pp.low_pass,
                     'dec':pp.decimate,
                     'win':pp.window_slds,
                     'kpca':pp.kpca,
                     'pca':pp.pca,
                     'std':pp.standardize}

pp_func_param_map = {}
for key in pp_func_label_map.keys():
    pp_func_param_map[key] = pp_func_label_map[key].__kwdefaults__

