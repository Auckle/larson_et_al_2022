##########################################################
# Imports

import argparse
import model_constants as mc
import numpy as np
import os
import pandas as pd


SET_NAMES_TRAIN_INDI = [
    "test_b_indi_all_10_100_1000_biome_c",
    "train_f_indi_10",
    "train_g_indi_100",
    "train_h_indi_1000",
]

SET_NAMES_TRAIN_GRPD = [
    "test_a_grpd_all_10_100_1000",
    "train_c_grpd_10",
    "train_d_grpd_100",
    "train_e_grpd_1000",
]

SET_NAMES_TEST_INDI = [
    "test_b_indi_all_10_100_1000_biome_c",
    "train_h_indi_1000",
]

SET_NAMES_TEST_GRPD = ["test_a_grpd_all_10_100_1000", "train_e_grpd_1000"]


##########################################################
# Functions


def print_set_info(set_name=str, label_name=str, db=pd.DataFrame, counts=False):
    a = db.groupby([set_name, label_name]).size()
    if counts:
        print(a)
    else:
        b = a.groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))
        print(b)
    print("")


def print_all_sets(db=pd.DataFrame, counts=False):
    for dataset in mc.CUSTOM_DATASETS_INFO:
        print_set_info(dataset["db_col"], dataset["label_col"], db, counts)


def define_training_sets_full(db=pd.DataFrame):
    for site in db.site.unique():

        # select all entries for site
        site_db = db.loc[db["site"] == site]
        # sort entries by date
        site_db = site_db.sort_values(by=["date"])

        # count entries
        site_cnt = len(site_db)

        # Set train images for Q1 models

        # calculate the location of the 80-20 time split for the entries
        # splitting training and testing by time for each location
        # NOT each label, makes a more realistic dataset
        if site_cnt <= 2:
            split_cnt = 0
        elif site_cnt <= 5:
            split_cnt = site_cnt - 2
        elif site_cnt <= 10:
            split_cnt = site_cnt - 3
        else:
            split_cnt = int(np.ceil(site_cnt * 0.8))

        split_entry = site_db.iloc[split_cnt]
        split_date = split_entry["date"]

        # set imgs taken before 80-20 time split to training  for Q1 models
        db.loc[
            (db["date"] < split_date) & (db["site"] == site),
            ["train_a_grpd_all", "train_b_indi_all"],
        ] = True
        # make sure to set all imgs. within a set to the same test or train label
        db.loc[
            (db["org_file"] == split_entry["org_file"]) & (db["site"] == site),
            ["train_a_grpd_all", "train_b_indi_all"],
        ] = True

        # mark the last of each label type as test this means that sometimes
        # a animal label is in the test set but not the training data
        for label in db.label_indi.unique():
            label_db = site_db[site_db[mc.LABEL_INDI] == label]
            if len(label_db) == 0:
                continue
            label_db = label_db.sort_values(by=["date"])
            last_row = label_db.iloc[-1]
            db.loc[
                db["org_file"] == last_row["org_file"],
                ["train_a_grpd_all", "train_b_indi_all"],
            ] = False


def define_training_sets_10_100_1000(set_names=list, label_name=str, db=pd.DataFrame):
    for site in db.site.unique():
        site_db = db[db["site"] == site]

        # sort entries by date
        site_db = site_db.sort_values(by=["date"])

        for label in db[label_name].unique():
            label_db = site_db[site_db[label_name] == label]
            label_db = label_db.sort_values(by=["date"])

            is_10_count = 0
            is_100_count = 0
            is_1000_count = 0
            is_test_count = 0

            train_10 = set([])
            train_100 = set([])
            train_1000 = set([])
            test_always = set([])
            test_10 = set([])
            test_100 = set([])
            test_1000 = set([])

            for index, row in label_db.iterrows():

                is_10 = False
                is_100 = False
                is_1000 = False
                is_test = False
                if len(test_10) == 0:
                    # save this one for testing
                    is_test_count += 1
                    is_test = True
                    test_always.add(row["org_file"])
                    test_10.add(row["org_file"])
                    test_100.add(row["org_file"])
                    test_1000.add(row["org_file"])
                else:
                    if row["org_file"] not in test_10:
                        if len(train_10) < 10:
                            is_10_count += 1
                            is_10 = True
                            train_10.add(row["org_file"])
                        if len(train_10) > 11:
                            test_10.add(row["org_file"])

                        if len(train_100) < 100:
                            is_100_count += 1
                            is_100 = True
                            train_100.add(row["org_file"])
                        if len(train_100) > 101:
                            test_100.add(row["org_file"])

                        if len(train_1000) < 1000:
                            is_1000_count += 1
                            is_1000 = True
                            train_1000.add(row["org_file"])
                        if len(train_1000) > 1001:
                            test_1000.add(row["org_file"])

                        # test images that are EMPTY, UNCLASSIFIABLE, or NATIVE
                        if (
                            not is_10
                            and not is_100
                            and not is_1000
                            and row[mc.LABEL_GRPD] != "INVASIVE"
                        ):
                            is_test_count += 1
                            is_test = True

            db.loc[(db["org_file"].isin(test_always)), [set_names[0]]] = True
            db.loc[(db["org_file"].isin(train_10)), [set_names[1]]] = True
            db.loc[(db["org_file"].isin(train_100)), [set_names[2]]] = True
            db.loc[(db["org_file"].isin(train_1000)), [set_names[3]]] = True


def define_testing_set_10_100_1000(set_names=list, label_name=str, db=pd.DataFrame):
    db.loc[
        (db[set_names[1]] == False) & (db[mc.LABEL_GRPD] != "INVASIVE"), [set_names[0]]
    ] = True

    for site in db.site.unique():
        site_db = db[db["site"] == site]
        # sort entries by date
        site_db = site_db.sort_values(by=["date"])

        for label in db[label_name].unique():
            label_db = site_db[site_db[label_name] == label]

            is_invasive = (
                label_db.shape[0] != 0 and label_db.iloc[0][mc.LABEL_GRPD] == "INVASIVE"
            )

            # 1% of INVASIVE IMAGES
            db_1000 = label_db[label_db[set_names[1]] == False]
            if db_1000.shape[0] > 1:
                image_sets = db_1000["org_file"].unique()
                poss_test_count = len(image_sets)
                test_count = np.maximum(int(np.floor(poss_test_count * 0.01)), 1)
                selected_1_percent = np.random.randint(0, poss_test_count, test_count)
                invasive_imgs_to_test = [
                    image_sets[index] for index in selected_1_percent
                ]
                db.loc[
                    (db["org_file"].isin(invasive_imgs_to_test)), [set_names[0]]
                ] = True


def filter_set_lg_and_sm_prj_by_classes(
    db=pd.DataFrame,
    hedgehog_sites=mc.HEDGEHOG_SITES_FROM_PAPER,
    lg_set_name=str,
    sm_set_name=str,
    starting_set_name=str,
    classes_to_keep=list,
):
    db[lg_set_name] = db[starting_set_name]
    db.loc[~db.label_indi.isin(classes_to_keep), [lg_set_name]] = False
    db.loc[db.site.isin(hedgehog_sites) & db[lg_set_name] == True, [sm_set_name]] = True


##########################################################
# Main


def main():

    ##########################################################
    # Configuration

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detections_csv",
        type=str,
        default=mc.DEFAULT_DETECTION_CSV_FILE,
        help="CSV file generated from the crop_detections.py script.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=mc.DEFAULT_OUTPUT_DIR_TRAIN_TEST_DF_CSV,
        help="Output directory for processed datasets csv",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=mc.DEFAULT_TRAIN_TEST_CSV_FILE_NAME,
        help="Output CSV file for processed datasets",
    )

    args = parser.parse_args()

    # Check arguments
    detections_csv = args.detections_csv
    assert os.path.exists(detections_csv), detections_csv + " does not exist"

    output_dir = args.output_dir
    assert os.path.exists(output_dir), output_dir + " does not exist"
    output_csv = os.path.join(output_dir, args.output_csv)

    print("\n##########################################################")
    print("##### START #####\n")

    print("detections csv:", detections_csv)
    print("   output_csv:", output_csv)

    db = pd.read_csv(detections_csv)

    db = db.set_index(["file"])
    db["date"] = pd.to_datetime(db["date"])
    db.columns

    db[mc.LABEL_INDI] = db["label"]
    db.loc[db[mc.LABEL_INDI] == "Ship rat", mc.LABEL_INDI] = "RAT"
    db.loc[db[mc.LABEL_INDI] == "Norway rat", mc.LABEL_INDI] = "RAT"
    db.loc[db[mc.LABEL_INDI] == "RABBIT", mc.LABEL_INDI] = "HARE"
    db.loc[db[mc.LABEL_INDI] == "NOTHINGHERE", mc.LABEL_INDI] = "EMPTY"

    db[mc.LABEL_GRPD] = "INVASIVE"
    db.loc[db[mc.LABEL_INDI] == "EMPTY", mc.LABEL_GRPD] = "EMPTY"
    db.loc[db[mc.LABEL_INDI] == "BIRD", mc.LABEL_GRPD] = "NATIVE"

    print(db[mc.LABEL_GRPD].value_counts())
    print(db[mc.LABEL_INDI].value_counts())
    print(db["label"].value_counts())

    for dataset in mc.CUSTOM_DATASETS_INFO:
        db[dataset["db_col"]] = False

    # Merge duplicate sites
    for site in db.site.unique():
        try:
            int_site = int(site)
            str_site = str(int_site)
            db.loc[db.site == site, ["site"]] = str_site
        except ValueError:
            pass

    # Prepare training sets
    # Set train_a_grpd_all, train_b_indi_all
    define_training_sets_full(db)

    # Set train sets: train_c_grpd_10, train_d_grpd_100, train_e_grpd_1000
    define_training_sets_10_100_1000(SET_NAMES_TRAIN_GRPD, mc.LABEL_GRPD, db)

    # Set train sets: train_f_indi_10, train_g_indi_100, train_h_indi_1000
    define_training_sets_10_100_1000(SET_NAMES_TRAIN_INDI, mc.LABEL_INDI, db)

    # set train set: train_i_lg_prj, train_j_sm_prj
    db.train_i_lg_prj = db.train_c_grpd_10
    db.loc[~db.label_indi.isin(mc.CLASSES_PRJ_SPCF), ["train_i_lg_prj"]] = False

    temp_db = db[(db.train_i_lg_prj == True) & (db.label_indi == "HEDGEHOG")]
    site_label_grouped = temp_db.groupby(["site", mc.LABEL_INDI]).size().nlargest(10)
    hedgehog_sites = [index[0] for index, item in site_label_grouped.items()]
    # Note the sites used in the paper were:
    # hedgehog_sites = ['040b', '022b', '011c', '011a', '022a', '040c', '952', '891', '018c', '038a']
    print("######## INFO: 10 Sites with the most hedgehogs", hedgehog_sites)
    db.loc[
        db.site.isin(hedgehog_sites) & (db.train_i_lg_prj) == True, ["train_j_sm_prj"]
    ] = True

    # Prepare testing sets
    # Set test set: test_a_grpd_all_10_100_1000
    define_testing_set_10_100_1000(SET_NAMES_TEST_GRPD, mc.LABEL_GRPD, db)

    # Set test set: test_b_indi_all_10_100_1000_biome_c
    define_testing_set_10_100_1000(SET_NAMES_TEST_INDI, mc.LABEL_INDI, db)

    # set test_c_indi_biome_ots
    db.loc[
        (db.test_b_indi_all_10_100_1000_biome_c == True)
        & (db.site.isin(mc.OUT_OF_SAMPLE_TEST_SITES_ALL)),
        ["test_c_indi_biome_ots"],
    ] = True

    # Set test_d_lg_wo_nvl_sps_lg_prj_spcf, test_f_sm_wo_nvl_sps_sm_prj_spcf
    filter_set_lg_and_sm_prj_by_classes(
        db,
        hedgehog_sites,
        "test_d_lg_wo_nvl_sps_lg_prj_spcf",
        "test_f_sm_wo_nvl_sps_sm_prj_spcf",
        "test_b_indi_all_10_100_1000_biome_c",
        mc.CLASSES_PRJ_SPCF,
    )

    # Set test_e_lg_w_nvl_sps, test_g_sm_w_nvl_sps
    filter_set_lg_and_sm_prj_by_classes(
        db,
        hedgehog_sites,
        "test_e_lg_w_nvl_sps",
        "test_g_sm_w_nvl_sps",
        "test_b_indi_all_10_100_1000_biome_c",
        mc.CLASSES_W_NVL_SPS,
    )

    # remove out of sample sites from every set except test_c_indi_biome_ots
    db.loc[db.site.isin(mc.OUT_OF_SAMPLE_TEST_SITES_ALL), mc.DATASETS_IN_SAMPLE] = False

    print_all_sets(db, True)
    print_all_sets(db)

    print("\n\n######## INFO: Saving processed datasets to", output_csv)
    db.to_csv(output_csv)

    print("##### END #####\n")
    print("\n##########################################################")


if __name__ == "__main__":
    main()
