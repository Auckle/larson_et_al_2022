##########################################################
# Imports
import argparse
import glob
import json
import model_constants as mc
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys
import traceback



##########################################################
# Functions


def save_progress(output_db=pd.DataFrame, output_csv=str):
    print("\n\n######## INFO: Saving cropping progress")
    output_db.to_csv(output_csv)


def img_crop_detection(output_dir=str, img_file=str, output_file=str, bbox=list, padding_factor=float):

    # Load image
    img = Image.open(img_file)
    img_width, img_height = img.size
    imsize = [img_width, img_height]
    image = np.array(img)

    # Get these boxes and convert normalized coordinates to pixel coordinates
    cords_box = bbox * np.tile(imsize, 2)

    # Pad the detected animal to a square box and additionally by padding factor, the
    # result will be in crop_boxes. However, we need to make sure that it box
    # coordinates are still within the image
    bbox_size = cords_box[-2:]
    offsets = (padding_factor * np.max([bbox_size], axis=1) - bbox_size) / 2
    crop_box = cords_box + np.hstack([-offsets, offsets])
    crop_box = np.maximum(0, crop_box).astype(int)

    # Crop image
    cropped_img = image[
        crop_box[1] : crop_box[1] + crop_box[3], crop_box[0] : crop_box[0] + crop_box[2]
    ]

    output_img_file = os.path.join(output_dir, output_file)
    save_img = Image.fromarray(cropped_img)
    save_img = save_img.resize((mc.CROP_SIZE_W_PADDING, mc.CROP_SIZE_W_PADDING))
    save_img.save(output_img_file)


##########################################################
# Main

if __name__ == "__main__":

    print("\n##########################################################")
    print("##### START #####\n")

    ##########################################################
    # Configuration

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "detections_json",
        type=str,
        default=None,
        help="JSON file generated from the Megadetector run_tf_detector_batch.py script.",
    )

    parser.add_argument(
        "img_metadata_csv", type=str, default=None, help="CSV file with img metadata"
    )

    parser.add_argument(
        "input_dir", type=str, default=None, help="Input directory to raw images"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        default=None,
        help="Output directory for cropped images and img metadata csv",
    )

    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.5,
        help="Threshold for detections to use. Default is 0.5.",
    )
    parser.add_argument(
        "--padding_factor",
        type=float,
        default=1.3 * 1.3,
        help="We will crop a tight square box around the animal enlarged by this factor. "
        + "Default is 1.3*1.3 = 1.69, which accounts for the cropping at test time and for"
        + " a reason",
    )
    args = parser.parse_args()


    # Check arguments
    detections_json = args.detections_json
    assert os.path.exists(detections_json), detections_json + " does not exist"

    image_metadata_csv = args.img_metadata_csv
    assert os.path.exists(image_metadata_csv), image_metadata_csv + " does not exist"

    input_dir = args.input_dir
    assert os.path.exists(input_dir), input_dir + " does not exist"

    output_dir = args.output_dir
    assert os.path.exists(output_dir), output_dir + " does not exist"

    output_csv = os.path.join(output_dir, "wellington_camera_traps_detections.csv")

    # Detection threshold should be in [0,1]
    detection_threshold = args.detection_threshold
    assert (
        detection_threshold >= 0 and detection_threshold <= 1
    ), "Detection threshold should be in [0,1]"

    # Padding around the detected objects when cropping
    # 1.3 for the cropping during test time and 1.3 for
    # the context that the CNN requires in the left-over
    # image
    padding_factor = args.padding_factor
    assert padding_factor >= 1, "Padding factor should be equal or larger 1"


    input_db = pd.read_csv(image_metadata_csv)
    input_db = input_db.set_index(["file"])

    if os.path.exists(output_csv):
        print("\n##### Loading existing output csv file.")
        output_db = pd.read_csv(output_csv)
        output_db = output_db.set_index(["file"])
    else:
        print("\n##### Creating new output database.")
        output_cols = list(input_db.columns.values)
        output_cols.append("file")
        output_cols.append("org_file")
        output_cols.append("detection")
        output_db = pd.DataFrame(columns=output_cols)
        output_db = output_db.set_index(["file"])

    detections_json = json.load(open(detections_json))
    detections_db = pd.DataFrame(detections_json["images"])
    detections_db = detections_db.set_index(["file"])

    image_list = glob.glob(input_dir + "/*.JPG")

    print(
        "\detections json:",
        detections_json,
        "\image metadata csv:",
        image_metadata_csv,
        "\input dir:",
        input_dir,
        "\output dir:",
        output_dir,
        "\output csv:",
        output_csv,
        "\detection threshold:",
        detection_threshold,
        "\padding factor:",
        padding_factor,
        "\n",
    )
    print("     input database:", input_db.columns.values)
    print("detections database:", detections_db.columns.values)
    print("    output database:", output_db.columns.values, "\n")

    raw_img_cnt = len(image_list)
    raw_img_processed_cnt = 0
    raw_empty_imgs = 0
    detections_total = 0
    detections_thresh_too_low = 0
    detections_processed = 0
    detections_previously_processed = 0
    imgs_w_no_detection_count = 0
    imgs_w_no_meta_data = 0
    last_detection_cords = [
        0.505458,
        0.5305875,
        0.61862177,
        0.6465393,
    ]  # random bbox detection

    try:

        # For each raw image
        for img_file in image_list:
            raw_img_processed_cnt += 1

            org_file_name = img_file.split("/")[-1]
            input_file_name = org_file_name.split(".")[0] + ".jpg"

            if input_file_name in input_db.index:

                input_row = input_db.loc[input_file_name]

                print(
                    "##### Processing img:",
                    img_file.split("/")[-1],
                    raw_img_processed_cnt,
                    "of",
                    raw_img_cnt,
                    raw_img_processed_cnt / raw_img_cnt * 100,
                    "% complete.",
                    end="\r",
                )

                if input_row["label"] == "NOTHINGHERE":
                    raw_empty_imgs += 1
                    # print("    # CROPPING IMG W/ NO ANIMAL - ", org_file_name, input_row['label'])
                    # use the bounding boxes from the last non-empty image to generate a cropped empty image
                    img_crop_detection(output_dir, img_file, new_file_name, last_detection_cords, padding_factor)
                    row = [
                        input_row["sequence"],
                        input_row["image_sequence"],
                        input_row["label"],
                        input_row["site"],
                        input_row["date"],
                        input_row["camera"],
                        org_file_name,
                        detection,
                    ]
                    output_db.loc[new_file_name] = row
                else:

                    detection_row = detections_db.loc[img_file]
                    detections = detection_row["detections"]
                    detection_index = 0

                    if len(detections) == 0:
                        imgs_w_no_detection_count += 1
                    else:
                        for detection in detections:
                            detections_total += 1

                            new_file_name = (
                                org_file_name.split(".")[0]
                                + "_"
                                + str(detection_index)
                                + ".JPG"
                            )

                            if not detection["conf"] > detection_threshold:
                                detections_thresh_too_low += 1
                                # print("   # SKIP - Detection threshold", detection['conf'], " is not greater than selected threshold ", detection_threshold)
                                continue

                            if new_file_name in output_db.index:
                                # print("    # SKIP - Detection already processed.")
                                detections_previously_processed += 1
                                detection_index += 1
                                continue

                            # print("    # CROPPING IMG W/ ANIMAL - ", new_file_name, input_row['label'])
                            img_crop_detection(
                                output_dir, img_file, new_file_name, detection["bbox"], padding_factor
                            )
                            last_detection_cords = detection["bbox"]
                            row = [
                                input_row["sequence"],
                                input_row["image_sequence"],
                                input_row["label"],
                                input_row["site"],
                                input_row["date"],
                                input_row["camera"],
                                org_file_name,
                                detection,
                            ]
                            output_db.loc[new_file_name] = row

                            detections_processed += 1

                            detection_index += 1

            else:
                imgs_w_no_meta_data += 1
                # print("##### ERROR No metadata found for img", org_file_name)

    except KeyboardInterrupt:
        save_progress(output_db, output_csv)
        raise
    except:
        print("##### ERROR in cropping process", sys.exc_info())
        print(traceback.format_exc())
        save_progress(output_db, output_csv)
        raise

    save_progress(output_db, output_csv)

    print(
        "\n##### Summary #####\n",
        raw_img_cnt,
        "raw images detected in data.",
        "\n",
        raw_img_processed_cnt,
        "raw images examined today.",
        "\n",
        raw_empty_imgs,
        "raw images labeled as NOTHINGHERE/EMPTY.",
        "\n",
        detections_total,
        "total detected animals.",
        "\n",
        detections_thresh_too_low,
        "detections did not meet the confidence threshold of ",
        detection_threshold,
        ".\n",
        detections_processed,
        "detections processed this run.",
        "\n",
        detections_previously_processed,
        "detections previously processed.",
        "\n",
        imgs_w_no_detection_count,
        "images with no detected animals",
        "\n",
        imgs_w_no_meta_data,
        "images with no metadate",
        "\n",
    )

    print("##### Output csv saved to", output_csv)
    print(output_db.head(10), "\n")
    print("##### output_db\n", output_db["label"].value_counts())
    print("\n###### input_db\n", input_db["label"].value_counts())

    print(" ")
    print("\n##### END #####")
    print("##########################################################")
