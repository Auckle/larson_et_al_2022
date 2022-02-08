##########################################################
# Imports

import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, full_db=pd.DataFrame, dataset_dir=str, dataset_info=dict, transform=None
    ):

        self.name = dataset_info["name"]
        self.class_map = dataset_info["class_map"]
        self.label_col = dataset_info["label_col"]
        self.db = full_db[full_db[dataset_info["db_col"]] == True]
        self.dataset_dir = dataset_dir
        print(
            "#####",
            "Create datset info:{} ".format(str(dataset_info)),
            "using data directory:{}, ".format(self.dataset_dir),
        )
        self.transform = transform
        file_list, label_map = self.get_file_list()
        self.file_list = file_list
        self.label_map = label_map
        print(
            "     {} images in dataset and found {} images.".format(
                len(self.db), len(file_list)
            )
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx=int):

        img = Image.open(
            os.path.join(self.dataset_dir, self.file_list[idx].split("/")[-1])
        )

        try:
            if self.transform:
                img = self.transform(img)
            # img = img.numpy()
            img = np.array(img)
            # image, actual_label, filename, site,
        except:
            print(
                "getting item",
                idx,
                self.dataset_dir,
                self.file_list[idx].split("/")[-1],
            )
            print(img, img.getextrema(), img.getextrema())
            raise

        try:
            img_value = img.astype("float32")
            label_value = self.label_map[self.file_list[idx]]["label"]
            idx_value = self.file_list[idx]
            site_value = self.label_map[self.file_list[idx]]["site"]
        except:
            print("{} label map does not contain {}".format(idx, self.file_list[idx]))
            raise

        return img_value, label_value, idx_value, site_value

    def get_file_list(self):

        file_list = []
        label_map = {}

        all_files = os.listdir(self.dataset_dir)
        fail_count = 0
        for img_file in self.db.index:
            if img_file in all_files:
                row = self.db.loc[img_file]
                file_list.append(img_file)
                img_info = {
                    "label": self.class_map[row[self.label_col]],
                    "site": row["site"],
                }
                label_map[img_file] = img_info
            else:
                fail_count += 1

        return file_list, label_map
