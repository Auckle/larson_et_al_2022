##########################################################
# Imports

import argparse
from bokeh.palettes import Colorblind, Category20c, Category20
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import pandas as pd
import pylab as plot
import seaborn as sns


##########################################################
# Constants

ALL_MODELS = [
    "train_a_grpd_all_test_a_grpd_all_10_100_1000",
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c",
    "train_b_indi_all_test_c_indi_biome_ots",
    "train_c_grpd_10_test_a_grpd_all_10_100_1000",
    "train_d_grpd_100_test_a_grpd_all_10_100_1000",
    "train_e_grpd_1000_test_a_grpd_all_10_100_1000",
    "train_f_indi_10_test_b_indi_all_10_100_1000_biome_c",
    "train_g_indi_100_test_b_indi_all_10_100_1000_biome_c",
    "train_h_indi_1000_test_b_indi_all_10_100_1000_biome_c",
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf",
    "train_i_lg_prj_test_e_lg_w_nvl_sps",
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf",
    "train_j_sm_prj_test_g_sm_w_nvl_sps",
]

# https://docs.bokeh.org/en/latest/docs/reference/palettes.html
model_colors_double = Category20[20]
model_colors = Category20c[20]
model_colors_color_blind = Colorblind[8]

MODEL_COLORS = {
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c": model_colors_color_blind[2],
    "train_b_indi_all_test_c_indi_biome_ots": model_colors_color_blind[2],
    "train_a_grpd_all_test_a_grpd_all_10_100_1000": model_colors_color_blind[3],
    "train_f_indi_10_test_b_indi_all_10_100_1000_biome_c": model_colors[3],
    "train_g_indi_100_test_b_indi_all_10_100_1000_biome_c": model_colors[1],
    "train_h_indi_1000_test_b_indi_all_10_100_1000_biome_c": model_colors[0],
    "train_c_grpd_10_test_a_grpd_all_10_100_1000": model_colors[7],
    "train_d_grpd_100_test_a_grpd_all_10_100_1000": model_colors[5],
    "train_e_grpd_1000_test_a_grpd_all_10_100_1000": model_colors[4],
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf": model_colors_double[13],
    "train_i_lg_prj_test_e_lg_w_nvl_sps": model_colors_double[13],
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf": model_colors_double[18],
    "train_j_sm_prj_test_g_sm_w_nvl_sps": model_colors_double[18],
}

MODELS_W_MATCHING_TRAIN_AND_TEST_LABELS = [
    "train_a_grpd_all_test_a_grpd_all_10_100_1000",
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c",
    "train_b_indi_all_test_c_indi_biome_ots",
    "train_c_grpd_10_test_a_grpd_all_10_100_1000",
    "train_d_grpd_100_test_a_grpd_all_10_100_1000",
    "train_e_grpd_1000_test_a_grpd_all_10_100_1000",
    "train_f_indi_10_test_b_indi_all_10_100_1000_biome_c",
    "train_g_indi_100_test_b_indi_all_10_100_1000_biome_c",
    "train_h_indi_1000_test_b_indi_all_10_100_1000_biome_c",
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf",
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf",
]

MODELS_Q1 = [
    "train_a_grpd_all_test_a_grpd_all_10_100_1000",
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c",
]
MODELS_Q1_CLASS_HATCHES = ["/", "///"]
MODEL_NAMES_LEG_Q1 = {
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c": "Individual",
    "train_a_grpd_all_test_a_grpd_all_10_100_1000": "Grouped",
}

MODELS_Q2 = [
    "train_c_grpd_10_test_a_grpd_all_10_100_1000",
    "train_d_grpd_100_test_a_grpd_all_10_100_1000",
    "train_e_grpd_1000_test_a_grpd_all_10_100_1000",
    "train_f_indi_10_test_b_indi_all_10_100_1000_biome_c",
    "train_g_indi_100_test_b_indi_all_10_100_1000_biome_c",
    "train_h_indi_1000_test_b_indi_all_10_100_1000_biome_c",
]
MODELS_Q2_HATCHES = [
    "/",
    "/",
    "/",
    "///",
    "///",
    "///",
]

MODELS_Q3 = [
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf",
    "train_j_sm_prj_test_g_sm_w_nvl_sps",
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf",
    "train_i_lg_prj_test_e_lg_w_nvl_sps",
]
MODELS_Q3_HATCHES = [
    None,
    "x",
    None,
    "x",
]
MODEL_NAMES_LEG_Q2_and_Q3 = {
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c": "13 Individual",
    "train_b_indi_all_test_c_indi_biome_ots": "13 Individual Out-of-Sample",
    "train_a_grpd_all_test_a_grpd_all_10_100_1000": "3 Grouped",
    "train_f_indi_10_test_b_indi_all_10_100_1000_biome_c": "10 Image",
    "train_g_indi_100_test_b_indi_all_10_100_1000_biome_c": "100 Image",
    "train_h_indi_1000_test_b_indi_all_10_100_1000_biome_c": "1,000 Image",
    "train_c_grpd_10_test_a_grpd_all_10_100_1000": "10 Image",
    "train_d_grpd_100_test_a_grpd_all_10_100_1000": "100 Image",
    "train_e_grpd_1000_test_a_grpd_all_10_100_1000": "1,000 Image",
    "train_i_lg_prj_test_e_lg_w_nvl_sps": "Large Project",
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf": "Large Project",
    "train_j_sm_prj_test_g_sm_w_nvl_sps": "Small Project",
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf": "Small Project",
}

MODELS_Q4 = [
    "train_b_indi_all_test_c_indi_biome_ots",
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c",
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf",
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf",
]
MODELS_Q4_HATCHES = [
    "o",
    "///",
    None,
    None,
]
MODEL_NAMES_LEG_Q4 = {
    "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c": "Custom",
    "train_b_indi_all_test_c_indi_biome_ots": "Off-the-shelf",
    "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf": "Large",
    "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf": "Small",
}


def get_model_hatches(model_list):
    if model_list == MODELS_Q1:
        return MODELS_Q1_CLASS_HATCHES
    elif model_list == MODELS_Q2:
        return MODELS_Q2_HATCHES
    elif model_list == MODELS_Q3:
        return MODELS_Q3_HATCHES
    elif model_list == MODELS_Q4:
        return MODELS_Q4_HATCHES


FIG_SIZE_MULTIPLIER = 7

PLOT_BACKGROUND_COLOR = "whitesmoke"
PLOT_GRID_STYLE = "dotted"

LINE_WIDTH = 0.5

AXES_LABEL_Y = 1.03

LEG_LOC = "upper right"
LEG_BORDER_PAD = 1
LEG_FANCY_BOX = True
LEG_SHADOW = True
LEG_BORDER_AX_PAD = 1.0

LEG_BORDER_AX_PAD_B = 1.3
LEG_BBOX_TO_ANCHOR = (1.1, 1.1)
LEG_LABEL_SPACING = 1
LEG_HANDLE_LENGTH = 2

LEG_LABEL_SPACING_B = 1.5
LEG_HANDLE_LENGTH_B = 4
LEG_PATCH_HEIGHT = 22
LEG_Y_OFFSET = -6

MARKER_SIZE = 150
MARKER_SIZE_3x1_PLT = 400
MARKER_SIZE_2x1_PLT = MARKER_SIZE_3x1_PLT
MARKER_ALPHA = 0
MARKER_EDGECOLOR = "black"

# font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

TWO_SUBPLOTS_SIZE_MOD = 1.4

PLOT_B_ANNOTATION_OFFSET = 3  # 3 points vertical offset
LEGEND_PATCH_EDGE_COLOR = "white"

##########################################################
# Functions

# helpers

def get_fig_path(fig_name=str, output_dir=str, prefix=None):
    if prefix:
        fig_name = prefix + "_" + fig_name
    fig_name = fig_name + ".png"
    return os.path.join(output_dir, fig_name)


def configure_matplt(sub_plt_cnt=int):
    if sub_plt_cnt == 2:
        # for 2 figure figures
        small_size = SMALL_SIZE * TWO_SUBPLOTS_SIZE_MOD
        medium_size = MEDIUM_SIZE * TWO_SUBPLOTS_SIZE_MOD  # default bar size
        bigger_size = BIGGER_SIZE * TWO_SUBPLOTS_SIZE_MOD

    else:
        # for 3 figure figures
        small_size = SMALL_SIZE
        medium_size = MEDIUM_SIZE
        bigger_size = BIGGER_SIZE

    params = {
        "legend.fontsize": small_size,
        "legend.handlelength": 2,
        "legend.title_fontsize": small_size,
    }
    plot.rcParams.update(params)
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=bigger_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=bigger_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("figure", titlesize=medium_size)  # fontsize of the figure title


def stat_by_training_images(
    df=pd.DataFrame,
    ax=plt.axes,
    x_stat=str,
    y_stat=str,
    x_label=None,
    y_label=None,
    face_color=None,
    edge_color=MARKER_EDGECOLOR,
    marker=None,
    title=None,
    limits=None,
    details=True,
    x_ax_percent=False,
    y_ax_percent=False,
    hide_y_lable=False,
    show_legend=True,
    legend_label=None,
    legend_title=None,
    marker_size=MARKER_SIZE,
    linewidth=LINE_WIDTH,
):

    if not x_label:
        x_label = x_stat
    if not y_label:
        y_label = y_stat

    if not edge_color:
        edge_color = face_color

    if not marker:
        marker = jl.INDIVIDUAL_MARKER

    ax.set_title(title, y=AXES_LABEL_Y)

    legend_label_attached = False

    for model_id in df.model_id.unique():

        x_df = df[df["model_id"] == model_id]
        x_df.sort_values(by=[x_stat])

        x = x_df[x_stat]
        y = x_df[y_stat]

        if y_ax_percent:
            y = [num * 100 for num in y]
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        if x_ax_percent:
            x = [num * 100 for num in x]
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        # only add legend label for first model
        if legend_label_attached:
            legend_label = None

        ax.scatter(
            x,
            y,
            label=legend_label,
            marker=marker,
            facecolors=face_color,
            edgecolors=edge_color,
            s=marker_size,
            alpha=0.7,
            linewidth=1,
        )

        legend_label_attached = True

    if show_legend:
        ax.legend(
            loc="lower right",
            ncol=1,
            fancybox=True,
            shadow=True,
            borderaxespad=1.0,
            title=legend_title,
            labelspacing=0.5,
        )  # , prop={'size': 120})

    ax.set_xlabel(
        x_label, labelpad=plt.rcParams["font.size"]
    )  # print(plt.rcParams["font.size"])
    ax.set_ylabel(y_label)
    if hide_y_lable:
        ax.set_ylabel("")
    # ax.set_ylabel(y_label, labelpad=SMALL_SIZE) # for figure 4

    ax.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax.grid(linestyle=PLOT_GRID_STYLE)

    if limits is not None:
        if limits[0] is not None:
            ax.set_xlim(left=limits[0])
        if limits[1] is not None:
            ax.set_xlim(right=limits[1])
        if limits[2] is not None:
            ax.set_ylim(top=limits[2])
        if limits[3] is not None:
            ax.set_ylim(bottom=limits[3])


def adjust_legend(legend=plt.legend, height=LEG_PATCH_HEIGHT, set_y=LEG_Y_OFFSET):
    legend._legend_box.align = "left"

    # https://github.com/mwaskom/seaborn/issues/1440
    legend_items = legend.get_patches()

    for k in range(len(legend_items)):
        patch = legend_items[k]

        patch.set_height(height)
        patch.set_y(set_y)

    for vpack in legend._legend_handle_box.get_children():
        for hpack in vpack.get_children():
            draw_area, text_area = hpack.get_children()
            for collection in draw_area.get_children():
                alpha = collection.get_alpha()
                if alpha == 0:
                    draw_area.set_visible(False)


def autolabel(
    rects,
    ax=plt.axes,
    center=False,
    stacked_on=None,
    sum_label=False,
    text_props=None,
    text_add="",
    cnts=False,
    font_size=plt.rcParams["font.size"],
):
    """Attach a text label above each bar in *rects*, displaying its height."""

    if len(rects) > 8:
        font_size = 8

    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()

        xytext_offset = (0, PLOT_B_ANNOTATION_OFFSET)
        if height < 0:
            xytext_offset = (0, -PLOT_B_ANNOTATION_OFFSET - font_size)

        format_string = "{:.1f}%"
        if cnts:
            format_string = "{}"
        if not sum_label:
            height_text = format_string.format(height) + text_add
        else:
            height_text = (
                format_string.format(height + stacked_on[i].get_height()) + text_add
            )

        if not stacked_on:
            stack_height = 0
        else:
            stack_height = stacked_on[i].get_height()

        if center:
            height_divide = 2
        else:
            height_divide = 1

        ax.annotate(
            height_text,
            xy=(
                rect.get_x() + rect.get_width() / 2,
                (height / height_divide) + stack_height,
            ),
            xytext=xytext_offset,
            fontsize=font_size,
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=text_props,
        )


def draw_legend_group(patches, model_names, legend_patches, leg_items, label, indexes):
    patch = Patch(
        label=label,
        edgecolor=LEGEND_PATCH_EDGE_COLOR,
        visible=False,
        alpha=MARKER_ALPHA,
    )
    legend_patches.append(patch)
    for i in indexes:
        patch = Patch(
            label=leg_items[model_names[i]],
            facecolor=patches[i].get_facecolor(),
            edgecolor=patches[i].get_edgecolor(),
            hatch=patches[i].get_hatch(),
        )
        legend_patches.append(patch)
    return legend_patches

def draw_legend(
    df=pd.DataFrame, 
    ax=plt.axes, 
    model_names=list, 
    model_leg_names=dict,
    group_names=list, 
    group_indexes=list, 
    patch_height=LEG_PATCH_HEIGHT
):
    patches = ax.patches
    legend_patches = []

    for group_name, group_index in zip(group_names, group_indexes):
        legend_patches = draw_legend_group(
            patches,
            model_names,
            legend_patches,
            model_leg_names,
            group_name,
            group_index,
        )

    # add legend to subplot
    leg = ax.legend(
        handles=legend_patches,
        loc=LEG_LOC,
        borderaxespad=LEG_BORDER_AX_PAD,
        borderpad=LEG_BORDER_PAD,
        fancybox=LEG_FANCY_BOX,
        shadow=LEG_SHADOW,
        labelspacing=LEG_LABEL_SPACING_B,
        handlelength=LEG_HANDLE_LENGTH_B,
    )

    # adjust legend
    adjust_legend(leg, height=patch_height)


# figures


def make_figure_stats_v_invs_natv_balance_in_train_df(df=pd.DataFrame, png_path=str):
    configure_matplt(3)

    model_names = MODELS_Q2
    group_by = "ALL"  # by model

    sb_plt_show_leg = False

    x_label = "Invasive to Non-Invasive\nTraining Image Ratio"
    x_col = "balance_inv_nat_train"

    y_labels = ["Top-1 Accuracy", "False Alarm Rate", "Missed Invasive Rate"]
    y_cols = ["top_1_correct_acc", "false_alarm_acc", "missed_invasive_acc"]
    y_ax_percent = True

    marker_grp = "o"
    marker_ind = "^"

    leg_label_grp = "Grouped Models"
    leg_label_ind = "Individual Models"

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, sharex=True, figsize=(3 * FIG_SIZE_MULTIPLIER, 1 * FIG_SIZE_MULTIPLIER)
    )
    fig.tight_layout(pad=5.0)
    axs = [ax1, ax2, ax3]

    df = df[df.model_name.isin(model_names)]
    df = df[df.result_type == group_by]

    for i in range(len(y_cols)):

        axs[i].scatter([], [], label=leg_label_grp, alpha=MARKER_ALPHA)

        for model_name in MODELS_Q2[0:3]:
            model_df = df[df.model_name == model_name]
            stat_by_training_images(
                model_df,
                axs[i],
                x_col,
                y_cols[i],
                x_label,
                y_labels[i],
                face_color=MODEL_COLORS[model_name],
                marker=marker_grp,
                marker_size=MARKER_SIZE_3x1_PLT,
                y_ax_percent=y_ax_percent,
                show_legend=sb_plt_show_leg,
                legend_label=MODEL_NAMES_LEG_Q2_and_Q3[model_name],
            )

        axs[i].scatter([], [], label=leg_label_ind, alpha=MARKER_ALPHA)

        for model_name in MODELS_Q2[3:]:
            model_df = df[df.model_name == model_name]
            stat_by_training_images(
                model_df,
                axs[i],
                x_col,
                y_cols[i],
                x_label,
                y_labels[i],
                face_color=MODEL_COLORS[model_name],
                marker=marker_ind,
                marker_size=MARKER_SIZE_3x1_PLT,
                y_ax_percent=y_ax_percent,
                show_legend=sb_plt_show_leg,
                legend_label=MODEL_NAMES_LEG_Q2_and_Q3[model_name],
            )

    legend = ax3.legend(
        loc=LEG_LOC,
        borderaxespad=LEG_BORDER_AX_PAD,
        borderpad=LEG_BORDER_PAD,
        bbox_to_anchor=LEG_BBOX_TO_ANCHOR,
        fancybox=LEG_FANCY_BOX,
        shadow=LEG_SHADOW,
        labelspacing=LEG_LABEL_SPACING,
        handlelength=LEG_HANDLE_LENGTH,
    )
    adjust_legend(legend)

    if png_path:
        plt.savefig(png_path)


def make_figure_false_alarm_and_missed_invasive_v_top_1_acc(df=pd.DataFrame, png_path=str):
    configure_matplt(2)

    model_names = MODELS_W_MATCHING_TRAIN_AND_TEST_LABELS
    group_by = "ALL"

    sb_plt_show_leg = False

    x_labels = ["Top-1 Accuracy", "Top-1 Accuracy"]
    x_cols = ["top_1_correct_acc", "top_1_correct_acc"]
    x_ax_percent = True

    y_labels = ["False Alarm Rate", "Missed Invasive Rate"]
    y_cols = ["false_alarm_acc", "missed_invasive_acc"]
    y_ax_percent = True

    ax_limts = [None, None, 37, -2]

    face_color = "black"
    marker = "x"
    linewidth = 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=True, figsize=(3 * FIG_SIZE_MULTIPLIER, 1.5 * FIG_SIZE_MULTIPLIER)
    )
    axs = [ax1, ax2]

    df = df[df.model_name.isin(model_names)]
    df = df[df.result_type == group_by]

    for i in range(len(x_cols)):

        axs[i].scatter([], [], label="13 Individual Classes", alpha=MARKER_ALPHA)

        stat_by_training_images(
            df,
            axs[i],
            x_cols[i],
            y_cols[i],
            x_labels[i],
            y_labels[i],
            face_color=face_color,
            marker=marker,
            marker_size=MARKER_SIZE_2x1_PLT,
            linewidth=linewidth,
            x_ax_percent=x_ax_percent,
            y_ax_percent=y_ax_percent,
            limits=ax_limts,
            show_legend=sb_plt_show_leg,
        )

    if png_path:
        plt.savefig(png_path)


def make_figure_top_k_train_df_cnt(df=pd.DataFrame, png_path=str):
    configure_matplt(2)

    model_names = ALL_MODELS
    group_by = "CLASS"

    linewidth = 1

    limits = [-1, 150000, None, None]

    x_label = "Training Images"
    x_col = "train_cnt"

    y_label = "Accuracy"
    y_col_top_1 = "top_1_correct_acc"
    y_col_top_5 = "top_5_correct_acc"

    sub_plt_title_top_1 = "Top-1"
    sub_plt_title_top_5 = "Top-5"

    PLOT_A_0_LEGEND_TITLE = "class type"
    leg_title = "Output Class Type"
    leg_label_native = "Native or Empty"
    leg_label_invasive = "Invasive"
    leg_loc = "lower right"

    invasive_color = model_colors_double[6]  # class_catgory_colors[0]
    invasive_marker = "+"

    native_color = model_colors_double[18]  # class_catgory_colors[1]
    native_marker = "o"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=True, figsize=(3 * FIG_SIZE_MULTIPLIER, 1.5 * FIG_SIZE_MULTIPLIER)
    )

    df = df[df.model_name.isin(model_names)]
    df = df[df.result_type == group_by]

    df_ntv = df[df.is_invasive == False]
    df_inv = df[df.is_invasive == True]

    # Top 1 recall

    # Native
    stat_by_training_images(
        df_ntv,
        ax1,
        x_col,
        y_col_top_1,
        x_label,
        y_label,
        "None",
        edge_color=native_color,
        marker=native_marker,
        linewidth=linewidth,
        title=sub_plt_title_top_1,
        limits=limits,
        y_ax_percent=True,
        show_legend=False,
        legend_label=leg_label_native,
        legend_title=PLOT_A_0_LEGEND_TITLE,
        marker_size=MARKER_SIZE_2x1_PLT,
    )

    # Invasive
    stat_by_training_images(
        df_inv,
        ax1,
        x_col,
        y_col_top_1,
        x_label,
        y_label,
        invasive_color,
        edge_color=invasive_color,
        marker=invasive_marker,
        marker_size=MARKER_SIZE_2x1_PLT,
        linewidth=linewidth,
        title=sub_plt_title_top_1,
        limits=limits,
        y_ax_percent=True,
        show_legend=False,
        legend_label=leg_label_invasive,
    )

    # Top 5 recall

    # Native
    stat_by_training_images(
        df_ntv,
        ax2,
        x_col,
        y_col_top_5,
        x_label,
        y_label,
        "None",
        edge_color=native_color,
        marker=native_marker,
        marker_size=MARKER_SIZE_2x1_PLT,
        linewidth=linewidth,
        title=sub_plt_title_top_5,
        limits=limits,
        y_ax_percent=True,
        hide_y_lable=True,
        show_legend=True,
        legend_label=leg_label_native,
    )

    # Invasive
    stat_by_training_images(
        df_inv,
        ax2,
        x_col,
        y_col_top_5,
        x_label,
        y_label,
        invasive_color,
        edge_color=invasive_color,
        marker=invasive_marker,
        marker_size=MARKER_SIZE_2x1_PLT,
        linewidth=linewidth,
        title=sub_plt_title_top_5,
        limits=limits,
        y_ax_percent=True,
        hide_y_lable=True,
        show_legend=True,
        legend_label=leg_label_invasive,
    )

    legend = ax2.legend(
        loc=leg_loc,
        borderaxespad=LEG_BORDER_AX_PAD_B,
        borderpad=LEG_BORDER_PAD,
        fancybox=LEG_FANCY_BOX,
        shadow=LEG_SHADOW,
        labelspacing=LEG_LABEL_SPACING,
        handlelength=LEG_HANDLE_LENGTH,
        title=leg_title,
    )
    adjust_legend(legend)

    plt.xlim(left=1)
    plt.xscale("log")
    plt.savefig(png_path)


def make_figure_all_stats_of_models(
    df=pd.DataFrame,
    png_path=str,
    model_names=list,
    y_cols=list,
    y_labels=list,
    y_label=None,
    legend_subplot=2,
    legend_type="Q1",
):
    configure_matplt(3)

    group_by = "ALL"  # by model

    hatches = get_model_hatches(model_names)

    fig, (axs) = plt.subplots(
        1,
        len(y_cols),
        sharex=True,
        sharey=False,
        figsize=(3 * FIG_SIZE_MULTIPLIER, 1.2 * FIG_SIZE_MULTIPLIER),
    )
    plt.subplots_adjust(wspace=0.25)

    df = df[df.model_name.isin(model_names)]
    df = df[df.result_type == group_by]

    for idx, (y_col, y_label, ax) in enumerate(zip(y_cols, y_labels, axs)):

        # convert values to %
        df[y_col] = df[y_col] * 100

        # draw bars
        bar_plot = sns.barplot(
            x="model_name",
            y=y_col,
            data=df,
            ax=ax,
            order=model_names,
            palette=MODEL_COLORS,
            edgecolor="black",
        )

        # set subplot background styles
        ax.set_facecolor(PLOT_BACKGROUND_COLOR)
        ax.grid(linestyle=PLOT_GRID_STYLE)

        # hide subplot x axis tick labels
        ax.get_xaxis().set_visible(False)

        # set subplot y axis parameters
        ax.set_title(y_label, y=AXES_LABEL_Y)
        ax.set_ylabel("")
        ax.tick_params(axis="y")
        ax.set_ylim([0, 100])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        # add values atop the bars
        autolabel(ax.patches, ax)

        # Loop over the bars
        patch_index = 0
        for i, thisbar in enumerate(ax.patches):
            # Set a different hatch for each bar
            thisbar.set_hatch(hatches[patch_index])
            patch_index += 1

        # legend
        if idx == legend_subplot:
            if legend_type == "Q1":
                group_names = ["Models"]
                group_indexes = [range(0, len(ax.patches))]
                draw_legend(df, ax, model_names, MODEL_NAMES_LEG_Q1, group_names, group_indexes)

            elif legend_type == "Q2":
                group_names = ["Grouped Models", "Individual Models"]
                group_indexes = [range(0, 3), range(3, 6)]
                MODEL_NAMES_LEG_Q2_and_Q3
                draw_legend(df, ax, model_names, MODEL_NAMES_LEG_Q2_and_Q3, group_names, group_indexes)

            elif legend_type == "Q3":
                group_names = ["Models Without Novel Species", "Models With Novel Species"]
                group_indexes = [range(0, 4, 2), range(1, 4, 2)]
                draw_legend(df, ax, model_names, MODEL_NAMES_LEG_Q2_and_Q3, group_names, group_indexes)
            
            elif legend_type == "Q4":
                group_names = ["Biome Models", "Project-specifc Models"]
                group_indexes = (range(0, 2), range(2, 4))
                draw_legend(df, ax, model_names, MODEL_NAMES_LEG_Q4, group_names, group_indexes, 20)

    plt.savefig(png_path, bbox_inches="tight")


##########################################################
# Main

if __name__ == "__main__":

    ##########################################################
    # Configuration

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures/",
        help="Directory containing cropped images",
    )
    parser.add_argument(
        "--figure_prefix",
        type=str,
        default=None,
        help="Prefix to add to saved figure file names.",
    )
    args = parser.parse_args()

    # Check arguments
    output_dir = args.output_dir
    assert os.path.exists(output_dir), output_dir + " does not exist"

    figure_prefix = args.figure_prefix

    df = pd.read_csv("./model_performance_results.csv")

    save_path = get_fig_path("stats_v_invs_natv_balance_in_train_df", output_dir, figure_prefix)
    make_figure_stats_v_invs_natv_balance_in_train_df(df, save_path)

    save_path = get_fig_path(
        "false_alarm_and_missed_invasive_v_top_1_acc", output_dir, figure_prefix
    )
    make_figure_false_alarm_and_missed_invasive_v_top_1_acc(df, save_path)

    save_path = get_fig_path("top_k_train_df_cnt", output_dir, figure_prefix)
    make_figure_top_k_train_df_cnt(df, save_path)

    stats = ["top_1_correct_acc", "false_alarm_acc", "missed_invasive_acc"]
    labels = ["Top-1 Accuracy", "False Alarm Rate", "Missed Invasive Rate"]

    save_path = get_fig_path("all_stats_Q1_models_grp_v_ind", output_dir, figure_prefix)
    make_figure_all_stats_of_models(
        df, save_path, MODELS_Q1, stats, labels, legend_subplot=1, legend_type="Q1"
    )

    save_path = get_fig_path("all_stats_Q2_models_10_100_1000", output_dir, figure_prefix)
    make_figure_all_stats_of_models(
        df, save_path, MODELS_Q2, stats, labels, legend_subplot=1, legend_type="Q2"
    )

    save_path = get_fig_path("all_stats_Q3_models_sm_v_lg", output_dir, figure_prefix)
    make_figure_all_stats_of_models(
        df, save_path, MODELS_Q3, stats, labels, legend_subplot=1, legend_type="Q3"
    )

    save_path = get_fig_path("all_stats_Q4_models_model_selection", output_dir, figure_prefix)
    make_figure_all_stats_of_models(
        df, save_path, MODELS_Q4, stats, labels, legend_subplot=1, legend_type="Q4"
    )
