import os

# Training dataset names
TRAINING_DATASETS = [
    "train_a_grpd_all",  # first  80% of images for each site by date time
    "train_b_indi_all",  # first  80% of images for each site by date time
    "train_c_grpd_10",  # first   10 imgs. by time for each class
    "train_d_grpd_100",  # first  100 imgs. by time for each class
    "train_e_grpd_1000",  # first 1000 imgs. by time for each class
    "train_f_indi_10",  # first   10 imgs. by time for each class
    "train_g_indi_100",  # first  100 imgs. by time for each class
    "train_h_indi_1000",  # first 1000 imgs. by time for each class
    "train_i_lg_prj",  # train_c_grpd_10 but only EMPTY, BIRD, HEDGEHOG classes
    "train_j_sm_prj",  # train_i_lg_prj but only the 10 sites with the most hedgehogs
]

# Testing dataset names
TESTING_DATASETS_IN_SAMPLE = [
    "test_a_grpd_all_10_100_1000",  # after first 1000 imgs. by time for each class and (native or empty or (1% of invasive))
    "test_b_indi_all_10_100_1000_biome_c",  # after first 1000 imgs. by time for each class and (native or empty or (1% of invasive))
    "test_c_indi_biome_ots",  # imgs. excluded from test_b_indi_all_10_100_1000_biome_c when removing out of sample sites
    "test_d_lg_wo_nvl_sps_lg_prj_spcf",  # test_b_indi_all_10_100_1000_biome_c but only EMPTY, BIRD, HEDGEHOG classes
    "test_e_lg_w_nvl_sps",  # test_b_indi_all_10_100_1000_biome_c but only EMPTY, BIRD, HEDGEHOG, RAT, UNCLASSIFIABLE classes
    "test_f_sm_wo_nvl_sps_sm_prj_spcf",  # test_d_lg_wo_nvl_sps_lg_prj_spcf but only the 10 sites with the most hedgehogs
    "test_g_sm_w_nvl_sps",  # test_e_lg_w_nvl_sps but only the 10 sites with the most hedgehogs
]

DATASETS_IN_SAMPLE = (
    TRAINING_DATASETS + TESTING_DATASETS_IN_SAMPLE[:2] + TESTING_DATASETS_IN_SAMPLE[3:]
)

TRAIN_DATASET_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TEST_DATASET_INDEXES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

CLASS_MAP_INDIVIDUAL = {
    "EMPTY": 0,
    "BIRD": 1,
    "UNCLASSIFIABLE": 2,
    "CAT": 3,
    "POSSUM": 4,
    "DOG": 5,
    "HEDGEHOG": 6,
    "MOUSE": 7,
    "RAT": 8,
    "HARE": 9,
    "DEER": 10,
    "PIG": 11,
    "MUSTELID": 12,
    "GOAT": 13,
}
CLASS_MAP_GROUPED = {"EMPTY": 0, "NATIVE": 1, "INVASIVE": 2}
CLASSES_MAP_PRJ_SPCF = {"EMPTY": 0, "BIRD": 1, "HEDGEHOG": 2}
CLASSES_MAP_W_NVL_SPS = {
    "EMPTY": 0,
    "BIRD": 1,
    "HEDGEHOG": 2,
    "RAT": 3,
    "UNCLASSIFIABLE": 4,
}

CLASSES_INDIVIDUAL = [
    "EMPTY",
    "BIRD",
    "UNCLASSIFIABLE",
    "CAT",
    "POSSUM",
    "DOG",
    "HEDGEHOG",
    "MOUSE",
    "RAT",
    "HARE",
    "DEER",
    "PIG",
    "MUSTELID",
    "GOAT",
]
CLASSES_GROUPED = ["EMPTY", "NATIVE", "INVASIVE"]
CLASSES_PRJ_SPCF = ["EMPTY", "BIRD", "HEDGEHOG"]
CLASSES_W_NVL_SPS = ["EMPTY", "BIRD", "HEDGEHOG", "RAT", "UNCLASSIFIABLE"]

LABEL_INDI = "label_indi"
LABEL_GRPD = "label_grpd"

CUSTOM_DATASETS_INFO = [
    {
        "name": "train_a_grpd_all",
        "id": 0,
        "db_col": "train_a_grpd_all",
        "train_id": 0,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
    },
    {
        "name": "train_b_indi_all",
        "id": 1,
        "db_col": "train_b_indi_all",
        "train_id": 1,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
    },
    {
        "name": "train_c_grpd_10",
        "id": 2,
        "db_col": "train_c_grpd_10",
        "train_id": 2,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
    },
    {
        "name": "train_d_grpd_100",
        "id": 3,
        "db_col": "train_d_grpd_100",
        "train_id": 3,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
    },
    {
        "name": "train_e_grpd_1000",
        "id": 4,
        "db_col": "train_e_grpd_1000",
        "train_id": 4,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
    },
    {
        "name": "train_f_indi_10",
        "id": 5,
        "db_col": "train_f_indi_10",
        "train_id": 5,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
    },
    {
        "name": "train_g_indi_100",
        "id": 6,
        "db_col": "train_g_indi_100",
        "train_id": 6,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
    },
    {
        "name": "train_h_indi_1000",
        "id": 7,
        "db_col": "train_h_indi_1000",
        "train_id": 7,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
    },
    {
        "name": "train_i_lg_prj",
        "id": 8,
        "db_col": "train_i_lg_prj",
        "train_id": 8,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_PRJ_SPCF,
        "class_map": CLASSES_MAP_PRJ_SPCF,
    },
    {
        "name": "train_j_sm_prj",
        "id": 9,
        "db_col": "train_j_sm_prj",
        "train_id": 9,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_PRJ_SPCF,
        "class_map": CLASSES_MAP_PRJ_SPCF,
    },
    {
        "name": "train_a_grpd_all_test_a_grpd_all_10_100_1000",
        "id": 10,
        "db_col": "test_a_grpd_all_10_100_1000",
        "train_id": 0,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_GROUPED,
        "test_id": 0,
    },
    {
        "name": "train_b_indi_all_test_b_indi_all_10_100_1000_biome_c",
        "id": 11,
        "db_col": "test_b_indi_all_10_100_1000_biome_c",
        "train_id": 1,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
        "test_id": 1,
    },
    {
        "name": "train_b_indi_all_test_c_indi_biome_ots",
        "id": 12,
        "db_col": "test_c_indi_biome_ots",
        "train_id": 1,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
        "test_id": 2,
    },
    {
        "name": "train_c_grpd_10_test_a_grpd_all_10_100_1000",
        "id": 13,
        "db_col": "test_a_grpd_all_10_100_1000",
        "train_id": 2,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
        "test_id": 0,
    },
    {
        "name": "train_d_grpd_100_test_a_grpd_all_10_100_1000",
        "id": 14,
        "db_col": "test_a_grpd_all_10_100_1000",
        "train_id": 3,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
        "test_id": 0,
    },
    {
        "name": "train_e_grpd_1000_test_a_grpd_all_10_100_1000",
        "id": 15,
        "db_col": "test_a_grpd_all_10_100_1000",
        "train_id": 4,
        "label_col": LABEL_GRPD,
        "class_list": CLASSES_GROUPED,
        "class_map": CLASS_MAP_GROUPED,
        "test_id": 0,
    },
    {
        "name": "train_f_indi_10_test_b_indi_all_10_100_1000_biome_c",
        "id": 16,
        "db_col": "test_b_indi_all_10_100_1000_biome_c",
        "train_id": 5,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
        "test_id": 1,
    },
    {
        "name": "train_g_indi_100_test_b_indi_all_10_100_1000_biome_c",
        "id": 17,
        "db_col": "test_b_indi_all_10_100_1000_biome_c",
        "train_id": 6,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
        "test_id": 1,
    },
    {
        "name": "train_h_indi_1000_test_b_indi_all_10_100_1000_biome_c",
        "id": 18,
        "db_col": "test_b_indi_all_10_100_1000_biome_c",
        "train_id": 7,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_INDIVIDUAL,
        "class_map": CLASS_MAP_INDIVIDUAL,
        "test_id": 1,
    },
    {
        "name": "train_i_lg_prj_test_d_lg_wo_nvl_sps_lg_prj_spcf",
        "id": 19,
        "db_col": "test_d_lg_wo_nvl_sps_lg_prj_spcf",
        "train_id": 8,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_PRJ_SPCF,
        "class_map": CLASSES_MAP_PRJ_SPCF,
        "test_id": 3,
    },
    {
        "name": "train_i_lg_prj_test_e_lg_w_nvl_sps",
        "id": 20,
        "db_col": "test_e_lg_w_nvl_sps",
        "train_id": 8,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_W_NVL_SPS,
        "class_map": CLASSES_MAP_W_NVL_SPS,
        "test_id": 4,
    },
    {
        "name": "train_j_sm_prj_test_f_sm_wo_nvl_sps_sm_prj_spcf",
        "id": 21,
        "db_col": "test_f_sm_wo_nvl_sps_sm_prj_spcf",
        "train_id": 9,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_PRJ_SPCF,
        "class_map": CLASSES_MAP_PRJ_SPCF,
        "test_id": 5,
    },
    {
        "name": "train_j_sm_prj_test_g_sm_w_nvl_sps",
        "id": 22,
        "db_col": "test_g_sm_w_nvl_sps",
        "train_id": 9,
        "label_col": LABEL_INDI,
        "class_list": CLASSES_W_NVL_SPS,
        "class_map": CLASSES_MAP_W_NVL_SPS,
        "test_id": 6,
    },
]


OUT_OF_SAMPLE_TEST_SITES_ALL = [
    "040a",
    "244",
    "030c",
    "050c",
    "005c",
    "021c",
    "005a",
    "038a",
    "039b",
    "006a",
]

# The 10 sites with most hedgehogs used in publication
HEDGEHOG_SITES_FROM_PAPER = [
    "040b",
    "022b",
    "011c",
    "011a",
    "022a",
    "040c",
    "952",
    "891",
    "018c",
    "038a",
]


CROP_SIZE = 224
CROP_SIZE_W_PADDING = CROP_SIZE + 20
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.01
NUM_EPOCHS = 55

MEANS_STDS_DF_COLS = [
    "train_index",
    "mean_a",
    "mean_b",
    "mean_c",
    "std_a",
    "std_b",
    "std_c",
]

PERFORMANCE_DF_COLS = [
    "model_id",
    "model_name",
    "result_type",
    "result_id",
    "is_invasive",
    "top_5_correct_cnt",
    "top_5_correct_acc",
    "top_1_correct_cnt",
    "top_1_correct_acc",
    "incorrect_invasive_cnt",
    "incorrect_invasive_acc",
    "false_alarm_cnt",
    "false_alarm_acc",
    "missed_invasive_cnt",
    "missed_invasive_acc",
    "train_cnt",
    "native_train_cnt",
    "invasive_train_cnt",
    "balance_inv_nat_train",
]


def get_checkpoint_path(dataset_name=str):
    return "./training/checkpoints/current_checkpoint_model_" + dataset_name + ".pt"


def get_best_model_path(dataset_name=str):
    return "./training/best_model/best_checkpoint_model_" + dataset_name + ".pt"


def get_evaluation_path(dataset_name=str):
    return "./training/predictions/predictions_" + dataset_name + ".csv"


TRAINING_LOSS_FIG_FILE = "./training/training_loss.png"
DEFAULT_DETECTION_CSV_FILE = "./data/detections/wellington_camera_traps_detections.csv"
DEFAULT_OUTPUT_DIR_TRAIN_TEST_DF_CSV = "./training/"
DEFAULT_TRAIN_TEST_CSV_FILE_NAME = "training_testing_datasets.csv"
DEFAULT_TRAIN_TEST_CSV_FILE = os.path.join(
    DEFAULT_OUTPUT_DIR_TRAIN_TEST_DF_CSV, DEFAULT_TRAIN_TEST_CSV_FILE_NAME
)
