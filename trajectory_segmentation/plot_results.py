from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from sklearn.preprocessing import MinMaxScaler
import constants
from load_data import load_preprocessed_pkl
import matplotlib.patheffects as PathEffects
import copy
import metrics
from matplotlib import rc



def plot_pixel_map_v_time(ax,kin_df,ann_df):
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(kin_df.values)
    features_scaled = np.pad(features_scaled,[[int(ann_df.iloc[0,0])+1,0],[0,0]],constant_values=1)
    ax.imshow(features_scaled.T,aspect='auto')
    hide_axes(ax)
    # plot_seg_lines(ax,ann_df,color = 'w', linestyle='-',ymin=0,ymax=1)
    plt.tight_layout(w_pad=.5,h_pad=.5)
    # plt.show()


def plot_trajectory_3d(kin_df,ann_df,index_list,title_list,seg_cmap,show=False,save=True,save_filename= './plots/jigsaws/'):
    rc('font', **{'family':'serif','serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)

    i = 1
    ax_list = []
    fig_traj_3d = plt.figure(figsize=(14, 10))
    for index in index_list:
        pos_cols = []
        pos_cols+= constants.slave_pos_cols_quat[0:3]
        pos_cols += constants.slave_pos_cols_quat[(len(constants.slave_pos_cols_quat)//2)+1:len(constants.slave_pos_cols_quat)//2+4]
        traj_data_3d = kin_df.loc(axis=0)[index[0],index[1]][pos_cols].values


        ax_list.append(fig_traj_3d.add_subplot(2,len(index_list),i,projection='3d'))
        ax_list.append( fig_traj_3d.add_subplot(2,len(index_list),len(index_list)+i,projection='3d'))
        _plot_trajectory_3d(ax_list[(i-1)*2],ax_list[(i-1)*2+1],traj_data_3d,ann_df.loc[index],seg_cmap)
        ax_list[(i-1)*2].annotate(title_list[i-1], xy=(0.5, 1.05),xycoords='axes fraction',weight="bold",fontsize=16,va="center", ha="center")
        i+=1
    configure_trajectory_3d_legend(fig_traj_3d,ax_list[1])
    plt.subplots_adjust(top=0.95,bottom=0.05,right=0.98,left=0.01,wspace=0.05,hspace=0.05)
    # plt.tight_layout()

    if save:
        fig_traj_3d.savefig(save_filename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_demo(kin_df,ann_df,index,seg_cmap,show=False,save=True,savedir= './plots/jigsaws/'):

    plot_trajectory_3d(kin_df,ann_df,index,seg_cmap,show,save,savedir+index[0]+"_"+index[1] +"_trajectory_3D.pdf")
    plt.rc('font', size=12)

    seg_fname = index[0]+"_"+index[1] +"_trajectory_segmentation.pdf"
    fig_seg = plt.figure(seg_fname[-4],figsize=(10,6))
    plot_traj_seg_v_time(fig_seg, kin_df, ann_df, index, seg_cmap)
    plt.tight_layout(w_pad=.5,h_pad=.5)

    if save:
        fig_seg.savefig(os.path.join(savedir, seg_fname))
        plt.close()

    if show:
        plt.show()



def plot_all_and_save(kin_df,ann_df,seg_cmap,savedir = './plots/jigsaws/'):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    kin_df_lp = pd.DataFrame()
    unique_indices = pd.unique(ann_df.index.droplevel(2))
    plt.rc('font', size=12)
    for i in range(0,unique_indices.shape[0]):

        plot_demo(kin_df,ann_df,unique_indices[i],seg_cmap,show=False,save=True,savedir=savedir)


def plot_kpca_3d(fig,traj_data_3d,ann_df,seg_cmap):
    ax_3d_lh = fig.add_subplot(1,2,1,projection='3d')

    p_i1 = ann_df.iloc[0]["i0"]
    for row in ann_df.itertuples():
        i0 = getattr(row,'i0')
        i1 = getattr(row,'i1')

        # if i0 - p_i1 > 1:
        #     ax_3d_lh.plot3D(traj_data_3d[p_i1:i0 + 1, 0], traj_data_3d[p_i1:i0 + 1, 1], traj_data_3d[p_i1:i0 + 1, 2],color='k',label="unlabeled")
        #     ax_3d_lh.plot3D(traj_data_3d[i0:i1+1,0],traj_data_3d[i0:i1+1,1],traj_data_3d[i0:i1+1,2],color=seg_cmap[row.label],label=row.label)
        # else:
        ax_3d_lh.plot3D(traj_data_3d[p_i1:i1+1,0],traj_data_3d[p_i1:i1+1,1],traj_data_3d[p_i1:i1+1,2],color=seg_cmap[row.label],label=row.label)

        p_i1 = i1
    plt.show()



def _plot_trajectory_3d(ax_3d_lh,ax_3d_rh,traj_data_3d,ann_df, segment_colors_dict):


    p_i1 = ann_df.iloc[0]["i0"]
    for row in ann_df.itertuples():
        i0 = getattr(row,'i0')
        i1 = getattr(row,'i1')

        # if i0 - p_i1 > 1:
        #     ax_3d_lh.plot3D(traj_data_3d[p_i1:i0 + 1, 0], traj_data_3d[p_i1:i0 + 1, 1], traj_data_3d[p_i1:i0 + 1, 2],color='k',label="unlabeled")
        #     ax_3d_rh.plot3D(traj_data_3d[p_i1:i0 + 1, 3], traj_data_3d[p_i1:i0 + 1, 4], traj_data_3d[p_i1:i0 + 1, 5],color='k',label="unlabeled")
        #     ax_3d_lh.plot3D(traj_data_3d[i0:i1+1,0],traj_data_3d[i0:i1+1,1],traj_data_3d[i0:i1+1,2],color=segment_colors_dict[row.label],label=row.label)
        #     ax_3d_rh.plot3D(traj_data_3d[i0:i1+1, 3], traj_data_3d[i0:i1+1, 4], traj_data_3d[i0:i1+1, 5],color=segment_colors_dict[row.label], label=row.label)
        # else:
        ax_3d_lh.plot3D(traj_data_3d[p_i1:i1+1,0],traj_data_3d[p_i1:i1+1,1],traj_data_3d[p_i1:i1+1,2],color=segment_colors_dict[row.label],label=row.label)
        ax_3d_rh.plot3D(traj_data_3d[p_i1:i1+1, 3], traj_data_3d[p_i1:i1+1, 4], traj_data_3d[p_i1:i1+1, 5],color=segment_colors_dict[row.label], label=row.label)

        p_i1 = i1
    ax_3d_lh.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d_lh.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d_lh.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d_rh.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d_rh.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d_rh.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    configure_trajectory_3d_axis(ax_3d_lh,"Left Tool Manipulator")
    configure_trajectory_3d_axis(ax_3d_rh,"Right Tool Manipulator")





def plot_gt_px_map_seg_v_time(kin_df,ann_df,index,savedir='./plots/jigsaws/init_seg/',seg_fname=""):
    if seg_fname is "":
        seg_fname = index[0] + "_" + index[1]+"_gt_segmentation.pdf"

    plt.rc('font', size=12)
    fig = plt.figure(seg_fname[-4],figsize=(10,3))
    fig.subplots_adjust(bottom=0.2)
    height_ratios = [2,1]

    gs = gridspec.GridSpec(2 , 1,height_ratios=height_ratios)

    ax_gt_seg = fig.add_subplot(gs[1, 0])
    plot_gt_segmentation_v_time(ax_gt_seg, ann_df,label=False)
    ax_px_map= fig.add_subplot(gs[0, 0],sharex=ax_gt_seg)

    plot_pixel_map_v_time(ax_px_map,kin_df,ann_df)

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    fig.savefig(savedir+ seg_fname)
    plt.close()

def plot_traj_seg_v_time(fig,kin_df,ann_df,index,segment_colors_dict):


    traj_ax_cols = [["LH_x","LH_y","LH_z","RH_x","RH_y","RH_z"],
                    ["LH_qx","LH_qy","LH_qz","LH_qw","RH_qx","RH_qy","RH_qz","RH_qw"],
                    ["LH_vx","LH_vy","LH_vz","RH_vx","RH_vy","RH_vz"],
                    ['LH_theta','RH_theta'],
                    ['LH_wx','LH_wy','LH_wz','RH_wx','RH_wy','RH_wz']]
    traj_col_labels = [[r'$x_{LH}$',r'$y_{LH}$',r'$z_{LH}$',r'$x_{RH}$',r'$y_{RH}$',r'$z_{RH}$'],
                        [r'$qx_{LH}$',r'$qy_{LH}$',r'$qz_{LH}$',r'$qw_{LH}$',r'$qx_{RH}$',r'$qy_{RH}$',r'$qz_{RH}$',r'$qw_{RH}$'],
                        [r'$vx_{LH}$',r'$vy_{LH}$',r'$vz_{LH}$',r'$vx_{RH}$',r'$vy_{RH}$',r'$vz_{RH}$'],
                       [r'$\theta_{LH}$',r'$\theta_{RH}$'],
                       [r'$wx_{LH}$',r'$wy_{LH}$',r'$wz_{LH}$',r'$wx_{RH}$',r'$wy_{RH}$',r'$wz_{RH}$']]
    lh_rh_colors = [cm.get_cmap("GnBu")(np.linspace(0.5, 1,4)),cm.get_cmap("PuRd")(np.linspace(0.5, 1,4))]
    gs = gridspec.GridSpec(len(traj_col_labels)+1, 1,height_ratios=[6,6,6,6,6,1])
    ax_list = []
    ax_labels = ["position (m)","quaternion","linear \n velocity (m/s)","gripper \n angle (rad)","angular\n velocity (rad/s)","Ground\nTruth"]
    for ax_i in range(0,len(traj_col_labels)):

        if ax_i > 0:
            ax = fig.add_subplot(gs[ax_i,0],sharex=ax_list[-1])
        else:
            ax = fig.add_subplot(gs[ax_i,0])
        for i in range(0,len(traj_ax_cols[ax_i])):
            plot_trajectory_v_time(ax,kin_df.loc[index],traj_ax_cols[ax_i][i],traj_col_labels[ax_i][i],
                                   lh_rh_colors[i//int(0.5*len(traj_ax_cols[ax_i]))][int(0.5*len(traj_ax_cols[ax_i])) - i%int(0.5*len(traj_ax_cols[ax_i]))-1]
                                   ,'-',ax_labels[ax_i])
        ax.get_xaxis().set_visible(False)
        plot_seg_lines(ax,ann_df.loc[index])
        ax_list.append(ax)

    ax_gt_seg = fig.add_subplot(gs[len(traj_col_labels),0],sharex=ax_list[-1])
    plot_gt_segmentation_v_time(ax_gt_seg,ann_df.loc[index],ax_labels[-1],segment_colors_dict)

def plot_trajectories_v_time(ax,kin_df,cols,labels,colors,linestyles,ax_label,show=True):
    for i in range(0,len(cols)):
        plot_trajectory_v_time(ax,kin_df,cols[i],labels[i],colors[i],linestyles[i],ax_label)
    if show:
        plt.show()

def plot_trajectory_v_time(ax, kin_df,col, label,color,linestyle,ax_label):
    ax.plot(kin_df[col].values,label=label,linewidth=1,color=color,linestyle=linestyle)
    configure_trajectory_v_time_legend(ax)
    ax.set_ylabel(ax_label)
    # ax.xaxis.set_major_locator(MultipleLocator(300))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(30))


def plot_gt_segmentation_v_time(ax,ann_df,ax_label="",segment_colors_dict=constants.segment_colors_dict,label=False):
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticks([0.5])
    ax.set_yticklabels([ax_label],fontsize=12)
    for tick_label in ax.get_yticklabels():
        tick_label.set_verticalalignment('center')
    x_min = ann_df.iloc[0]["i0"]
    rect_height = 1

    p_i1 = x_min
    for row in ann_df.itertuples():
        i0 = getattr(row,'i0')
        i1 = getattr(row,'i1')

        # if i0 - p_i1 > 1:
        #     rect = patches.Rectangle((p_i1, 0), i0-p_i1+1, rect_height, linewidth=1, edgecolor='k', facecolor='k',label="unlabeled")
        #     ax.add_patch(rect)
        #     rect = patches.Rectangle((i0, 0), i1-i0+1, rect_height, linewidth=1, edgecolor='k', facecolor=segment_colors_dict[row.label],label=row.label)
        #     ax.add_patch(rect)
        #     rx, ry = rect.get_xy()
        #     cx = rx + rect.get_width() / 2.0
        #     cy = ry + rect.get_height() / 2.0
        #     if label:
        #         t = ax.text(cx,cy,row.label,color='w',fontsize="8",ha="center",va="center")
        #         t.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
        #
        # else:
        rect = patches.Rectangle((p_i1, 0), i1-p_i1+1, rect_height, linewidth=1, edgecolor='k', facecolor=segment_colors_dict[row.label],label=row.label)
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        # if label:
        #     t = ax.text(cx,cy,row.label,color='w',fontsize="8",ha="center",va="center")
        #     t.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])


        p_i1 = i1

    x_max = p_i1
    ax.set_xlabel("Frame no.")
    ax.set_xlim(x_min,x_max)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_ylim(0,1)

    if not label:
        configure_gt_seg_legend(ax)


def plot_seg_lines(ax,seg_df,color = 'k', linestyle='--',ymin=0,ymax=1):
    p_i1 = seg_df.iloc[0]["i0"]
    count = 0
    for row in seg_df.itertuples():
        i0 = getattr(row,'i0')
        i1 = getattr(row,'i1')

        # if i0 - p_i1 > 1:
        #     ax.axvline(i0, color=color, linestyle=linestyle,linewidth=1,ymin=ymin,ymax=ymax,clip_on=False)
        if count != seg_df.shape[0]-1:
            ax.axvline(i1, color=color, linestyle=linestyle,linewidth=1,ymin=ymin,ymax=ymax,clip_on=False)

        p_i1 = i1
        count += 1

def get_colors(num_colors,cmap_name):
    colors = cm.get_cmap(cmap_name)(np.linspace(0, 1,num_colors))
    return colors

def get_segment_colors_dict_from_ann(ann_df,cmap_name='gist_rainbow'):
    np.random.seed(0)
    segment_labels_unique = ann_df.label.unique()
    np.random.shuffle(segment_labels_unique)
    colors = get_colors(len(segment_labels_unique),cmap_name)
    segment_colors_dict = dict(zip(segment_labels_unique,colors))
    return segment_colors_dict


def configure_trajectory_3d_legend(fig,ax):
    # remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center',bbox_to_anchor=[0,-0.02,1,0.05],ncol = len(by_label),mode="expand",borderaxespad=0.,bbox_transform=plt.gcf().transFigure)


def configure_trajectory_3d_axis(ax,title):

    # set plot and axes titles
    ax.set_title(title,{'verticalalignment':'bottom'})
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

def plot_gesture_color_legend(gesture_color_dict,gesture_descr_dict,save_filename):
    rc('font', **{'family':'serif','serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)

    legend_elements = []
    for gesture_key in gesture_color_dict.keys():
        legend_elements += [Patch(facecolor=gesture_color_dict[gesture_key], edgecolor='k',label=gesture_key + ": " + gesture_descr_dict[gesture_key])]

    # Create the figure
    fig, ax = plt.subplots()
    lgd = ax.legend(handles=legend_elements, loc='center',frameon=False,ncol=2)
    ax.axis('off')
    ax.autoscale(enable=True)
    hide_axes(ax)
    fig.tight_layout(pad=0)
    plt.savefig(save_filename, bbox_extra_artists=[lgd], bbox_inches='tight')
    plt.close()


def hide_axes(ax):
    hide_xaxis(ax)
    hide_yaxis(ax)

def hide_xaxis(ax):
    ax.get_xaxis().set_visible(False)

def hide_yaxis(ax):
    ax.get_yaxis().set_visible(False)

def configure_gt_seg_legend(ax):
    # remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower center',bbox_to_anchor=[0,-0.01,1,1],ncol = len(by_label),mode="expand",borderaxespad=0,bbox_transform=plt.gcf().transFigure)

def configure_trajectory_v_time_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', prop={'size':4}, bbox_to_anchor=[1.01, 0.5,0.5,0.5] ,borderaxespad=0.)


def get_feature_pca_labels(metrics_df):

    x_labels_idx = []

    feat_labels = []
    for (i,feat_key) in enumerate(constants.feat_columns_map.keys()):
        no_pca = False
        feat_exists = False
        x_vals = []
        for idx in metrics_df.index:
            if feat_key+"_kpca" in idx:
                idx_split = idx.split("_")
                x_vals += [int(idx_split[-3][4:])]
                feat_exists = True
            elif feat_key+"_g" in idx:
                no_pca = True
                feat_exists = True
        if feat_exists:
            feat_labels.append(feat_key)

        x_vals.sort()
        this_x_labels = ["NO_PCA"] if no_pca else []
        this_x_labels += x_vals
        if i > 0:
            x_labels_idx = x_labels_idx.intersection(pd.Index(this_x_labels))
        else:
            x_labels_idx = pd.Index(this_x_labels)
    x_labels = x_labels_idx.tolist()


    return x_labels,feat_labels

def get_feature_pca_metrics_dict(metrics_df,feat_labels,x_labels,metric_key):
    metrics_dict = {}
    for feat in feat_labels:
        feat_metrics_df = metrics_df[metrics_df.index.str.startswith(feat+"_k") | metrics_df.index.str.startswith(feat+"_g")]
        x_vals = []
        for x in x_labels:
            if x == "NO_PCA":
               this_val = feat_metrics_df[~feat_metrics_df.index.str.contains('kpca')][metric_key].values[0]
            else:
               this_val = feat_metrics_df[feat_metrics_df.index.str.contains(str(x))][metric_key].values[0]
            x_vals += [this_val]

        metrics_dict[feat] = np.array(x_vals)
    return metrics_dict

def plot_feature_pca_eval(metrics_df,save_filepath=""):
    # get xlabels
    rc('font', **{'family':'serif','serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    feat_labels = []
    si_vals = []
    dbi_vals = []

    x_labels,feat_labels = get_feature_pca_labels(metrics_df)
    si_metrics_dict = get_feature_pca_metrics_dict(metrics_df,feat_labels,x_labels, "si")
    dbi_metrics_dict = get_feature_pca_metrics_dict(metrics_df,feat_labels,x_labels, "dbi")


    for i in range(0,len(x_labels)):
        if isinstance(x_labels[i],str):
            x_labels[i] = x_labels[i].replace("_","\_")
        else:
            x_labels[i] = str(x_labels[i])

    feat_colors = cm.get_cmap("tab10")(np.linspace(0,1,10))
    feat_colors = np.vstack((feat_colors[0:5], feat_colors[6:]))
    feat_colors_dict = dict(zip(feat_labels,feat_colors[0:len(feat_labels)]))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(wspace=0.0, hspace=0.05)
    fig = plt.figure()

    ax_si = fig.add_subplot(gs[1, 0])
    for feat_key in si_metrics_dict.keys():
        ax_si.plot(x_labels,si_metrics_dict[feat_key],color=feat_colors_dict[feat_key],label = feat_key.replace("_","\_"),marker='o',fillstyle='none')
    ax_si.set_ylabel("SI")
    ax_si.set_xlabel("\# Principal Components")
    ax_si.spines['top'].set_visible(False)
    ax_si.spines['right'].set_visible(False)


    ax_dbi = fig.add_subplot(gs[0, 0],sharex=ax_si)
    for feat_key in dbi_metrics_dict.keys():
        ax_dbi.plot(x_labels,dbi_metrics_dict[feat_key],color=feat_colors_dict[feat_key],label = feat_key.replace("_","\_"),marker='o',fillstyle='none')
    ax_dbi.set_ylabel("DBI")
    ax_dbi.legend(loc='bottom left',bbox_to_anchor=[-0.1, 1.2,1.2,.05],ncol=len(feat_labels),borderaxespad=0.,mode="expand")
    ax_dbi.spines['top'].set_visible(False)
    ax_dbi.spines['right'].set_visible(False)
    ax_dbi.spines['bottom'].set_visible(False)

    hide_xaxis(ax_dbi)
    fig.subplots_adjust(bottom=0.15, top=0.85)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_filepath)

def get_window_pca_labels(metrics_df):

    x_labels_idx = []
    s = "_"
    win_features = [s.join(f.split(s)[0:3]) for f in metrics_df.index[metrics_df.index.str.contains("pos_eedist_win")]]

    feat_labels = {}
    features = ["pos_eedist"] + np.unique(win_features).tolist()
    for (i,feat_key) in enumerate(features):
        no_pca = False
        feat_exists = False
        x_vals = []
        for idx in metrics_df.index:
            if feat_key +"_kpca" in idx:
                idx_split = idx.split("_")
                x_vals += [int(idx_split[-3][4:])]
                feat_exists = True
            elif feat_key+"_g" in idx:
                no_pca = True
                feat_exists = True
        if feat_exists:
            if "win" in feat_key:
                feat_labels[feat_key] = "W="+feat_key.split("_")[2][3:]
            else:
                feat_labels[feat_key] = "W=1"

        x_vals.sort()
        this_x_labels = ["NO_PCA"] if no_pca else []
        this_x_labels += x_vals
        if i > 0:
            x_labels_idx = x_labels_idx.intersection(pd.Index(this_x_labels))
        else:
            x_labels_idx = pd.Index(this_x_labels)
    x_labels = x_labels_idx.tolist()
    feat_labels = {k:v for k,v in sorted(feat_labels.items(),key=lambda x: int(x[1][2:]))}

    return x_labels,feat_labels

def get_window_pca_metrics_dict(metrics_df,feat_labels,x_labels,metric_key):
    metrics_dict = {}
    for feat in feat_labels.keys():
        feat_metrics_df = metrics_df[metrics_df.index.str.startswith(feat+"_k") | metrics_df.index.str.startswith(feat+"_g")]
        x_vals = []
        for x in x_labels:
            if x == "NO_PCA":
               this_val = feat_metrics_df[~feat_metrics_df.index.str.contains('kpca')][metric_key].values[0]
            else:
               this_val = feat_metrics_df[feat_metrics_df.index.str.contains(str(x))][metric_key].values[0]
            x_vals += [this_val]

        metrics_dict[feat] = np.array(x_vals)
    return metrics_dict


def plot_window_pca_eval(metrics_df,save_filepath=""):
    rc('font', **{'family':'serif','serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)

    si_vals = []
    dbi_vals = []

    x_labels,feat_labels = get_window_pca_labels(metrics_df)
    si_metrics_dict = get_window_pca_metrics_dict(metrics_df,feat_labels,x_labels, "si")
    dbi_metrics_dict = get_window_pca_metrics_dict(metrics_df,feat_labels,x_labels, "dbi")


    for i in range(0,len(x_labels)):
        if isinstance(x_labels[i],str):
            x_labels[i] = x_labels[i].replace("_","\_")
        else:
            x_labels[i] = str(x_labels[i])

    feat_colors = cm.get_cmap("tab10")(np.linspace(0,1,10))
    feat_colors = np.vstack((feat_colors[0:5], feat_colors[6:]))
    feat_colors_dict = dict(zip(feat_labels.keys(),feat_colors[0:len(feat_labels)]))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(wspace=0.0, hspace=0.05)
    fig = plt.figure()

    ax_si = fig.add_subplot(gs[1, 0])
    for feat_key in si_metrics_dict.keys():
        ax_si.plot(x_labels,si_metrics_dict[feat_key],color=feat_colors_dict[feat_key],label = feat_labels[feat_key],marker='o',fillstyle='none')
    ax_si.set_ylabel("SI")
    ax_si.set_xlabel("\# Principal Components")
    ax_si.spines['top'].set_visible(False)
    ax_si.spines['right'].set_visible(False)


    ax_dbi = fig.add_subplot(gs[0, 0],sharex=ax_si)
    for feat_key in dbi_metrics_dict.keys():
        ax_dbi.plot(x_labels,dbi_metrics_dict[feat_key],color=feat_colors_dict[feat_key],label = feat_labels[feat_key],marker='o',fillstyle='none')
    ax_dbi.set_ylabel("DBI")
    ax_dbi.legend(loc='bottom left',bbox_to_anchor=[-.1, 1.2,1.2,.05],ncol=len(feat_labels),borderaxespad=0.,mode="expand")
    ax_dbi.spines['top'].set_visible(False)
    ax_dbi.spines['right'].set_visible(False)
    ax_dbi.spines['bottom'].set_visible(False)

    hide_xaxis(ax_dbi)
    fig.subplots_adjust(bottom=0.15, top=0.85)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_filepath)

def plot_demo_seg_results(kin_df,ann_df,seg_df_list,seg_alg_list,save_filepath):

    fig = plt.figure(figsize=(9,6))

    height_ratios = [2]
    for i in range(0,len(seg_df_list)+1):
        height_ratios.append(1)

    gs = gridspec.GridSpec(2 + len(seg_df_list) , 1,height_ratios=height_ratios)

    ax_gt_seg = fig.add_subplot(gs[len(seg_df_list)+1, 0])
    plot_gt_segmentation_v_time(ax_gt_seg, ann_df, "GT")
    ax_px_map= fig.add_subplot(gs[0, 0],sharex=ax_gt_seg)

    plot_pixel_map_v_time(ax_px_map,kin_df,ann_df)
    ax_list = [ax_gt_seg]
    for ax_i in range(0,len(seg_df_list) ):
        ax_seg_alg = fig.add_subplot(gs[ax_i+1, 0], sharex=ax_list[-1])
        segment_labels_unique = ann_df.label.unique()

        if ann_df.iloc[-1,1] > seg_df_list[ax_i].iloc[-1,1]:
            seg_df_list[ax_i].iloc[-1, 1] = ann_df.iloc[-1,1]

        pred_to_gt_label_map = metrics.map_pred_labels_to_gt(ann_df,seg_df_list[ax_i],len(constants.segment_colors_dict.keys()))
        colors = cm.get_cmap("Set3")(np.linspace(0,1,12))
        colors = colors[0:8].tolist() + colors[9:].tolist()
        colors_dict = constants.segment_colors_dict
        count = 0
        for k,v in pred_to_gt_label_map.items():
            for r in range(0,seg_df_list[ax_i].shape[0]):
                if seg_df_list[ax_i].iloc[r,2]  == k:
                    seg_df_list[ax_i].iloc[r,2] = v
            if v not in constants.segment_colors_dict:
                colors_dict[v] = colors[count]
                count+=1


        plot_gt_segmentation_v_time(ax_seg_alg, seg_df_list[ax_i],  seg_alg_list[ax_i],colors_dict,label=True)
        # plot_seg_lines(ax_seg_alg,seg_df_list[ax_i],'r','-')
        ax_list.append(ax_seg_alg)
        ax_seg_alg.get_xaxis().set_visible(False)
    fig.subplots_adjust(left=0.13)
    fig.savefig(save_filepath)
    plt.close()


def plot_seg_results(feat_dfs,gt_seg_df,pred_seg_dfs,best_trials,fig_paths):
    feat = "pos_eedist_win3_kpca12"
    rc('font', **{'family':'serif','serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    for (i,task) in enumerate(constants.tasks):
        feat_df = feat_dfs[task+"_"+feat]
        unique, counts = np.unique([trial[0]+"_"+trial[1] for key,trial in best_trials.items() if (task+"_"+feat) in key],return_counts=True)
        index = unique[np.argmax(counts)]
        index = index.split("_")
        index = ("_".join(index[0:-1]),index[-1])
        pred_seg_df_list = [df.loc(axis=0)[index[0],index[1]] for key,df in pred_seg_dfs.items() if (task in key) and (feat in key)]

        axis_labels = [l.upper().replace("_","\_") for l in constants.seg_func_label_map.keys() if "gmeans" not in l]
        plot_demo_seg_results(feat_df.loc(axis=0)[index[0],index[1]],gt_seg_df.loc(axis=0)[index[0],index[1]],pred_seg_df_list,axis_labels,fig_paths[i])


# def plot_pca_metrics(metrics_df)
if __name__ == "__main__":
    kin_df, ann_df = load_preprocessed_pkl(feature_type='quat', pkl_path=constants.pkl_path)
    ann_df_test = ann_df.xs(('Suturing','D004'),axis=0,drop_level=False).copy()

    kin_df_test = kin_df.xs(('Suturing','D004'),axis=0,drop_level=False).copy()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plot_pixel_map_v_time(ax,kin_df_test,ann_df_test)