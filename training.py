##########################################################
# Imports
import argparse
from collections import OrderedDict
from custom_dataset import CustomDataset
import matplotlib.pyplot as plt
import model_constants as mc
import numpy as np
import os
import pandas as pd
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

##########################################################
# Citations
"""
Norouzzadeh, M. S., Nguyen, A., Kosmala, M., Swanson, A., Palmer, M. S., Packer, C., &
Clune, J. (2018). Automatically identifying, counting, and describing wild animals in
camera-trap images with deep learning. Proceedings of the National Academy of Sciences
- PNAS,115(25), Article E5716–E5725. https://doi.org/10.1073/pnas.1719367115

Tabak, M. A., Norouzzadeh, M. S., Wolfson, D. W., Sweeney, S. J., Vercauteren, K. C.,
Snow, N. P., Halseth, J. M., Di Salvo, P. A., Lewis, J. S., White, M. D., Teton, B.,
Beasley, J. C., Schlichting, P. E., Boughton, R. K.,Wight, B., Newkirk, E. S., Ivan,
J. S., Odell, E. A., Brook, R. K., ... Miller, R. S. (2019). Machine learning to
classify animal species in camera trap images: Applications in ecology. Methods in
Ecology and Evolution,10(4), 585–590. https://doi.org/10.1111/2041-210X.13120
"""


##########################################################
# Functions


# Norouzzadeh et al., 2018, Tabak et al., 2019
def update_optimizer(epoch=int, optimizer=optim.SGD):
    """
    Epoch Number Learning Rate Weight Decay
    1-18         0.01          0.0005
    19-29        0.005         0.0005
    30-43        0.001         0
    44-52        0.0005        0
    53-55        0.0001        0
    """
    lr = 0
    weight_decay = 0

    if epoch <= 18:
        lr = 0.01
        weight_decay = 0.0005
    elif epoch <= 29:
        lr = 0.005
        weight_decay = 0.0005
    elif epoch <= 43:
        lr = 0.001
        weight_decay = 0.0
    elif epoch <= 52:
        lr = 0.000
        weight_decay = 0.0
    else:
        lr = 0.000
        weight_decay = 0.0

    for g in optimizer.param_groups:
        g["lr"] = lr
        g["weight_decay"] = weight_decay

    return optimizer


def get_model(class_count=int):

    # Download the pretrained model
    model = models.resnet18(pretrained=True)

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Classifier architecture to put on top of resnet18
    fc = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(512, 100)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(100, class_count)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.fc = fc

    return model


def save_model(
    epoch=int,
    model=models.resnet.ResNet,
    optimizer=optim.SGD,
    epoch_loss=torch.tensor,
    path=str,
    verbose=True,
    add_time_stamp=False,
):
    save_path = path
    if add_time_stamp:
        save_path += time.strftime("%Y%m%d-%H%M%S")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        },
        path,
    )
    if not verbose:
        print(
            "SAVING MODEL | epoch {} | loss {:.6f} | {}".format(epoch, epoch_loss, path)
        )


def get_means_stds(
    df=pd.DataFrame,
    dataset_dir=str,
    dataset_info=dict,
    mean_stds_df=pd.DataFrame,
    save_file=str,
):

    train_index = dataset_info["train_id"]
    try:
        row = mean_stds_df.loc[train_index]
        means_stds = [
            [row.mean_a, row.mean_b, row.mean_c],
            [row.std_a, row.std_b, row.std_c],
        ]
        print(
            "##### Loaded means and stds for training dataset", train_index, means_stds
        )

    except KeyError:
        dataset = CustomDataset(df, dataset_dir, dataset_info)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        mean = 0.0
        std = 0.0
        nb_samples = 0.0

        for images, labels, filenames, sites in dataloader:

            try:  # cuda
                # torch.set_default_tensor_type("torch.cuda.FloatTensor")
                data = images.view(len(images), 3, -1).cuda()
                mean += data.mean(2).sum(0).cuda()
                std += data.std(2).sum(0).cuda()
            except TypeError:  # Use cpu
                data = images.view(len(images), 3, -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)

            nb_samples += len(images)
            mean /= nb_samples
            std /= nb_samples

            mean /= 255
            std /= 255

            means = mean.tolist()
            stds = std.tolist()
            means_stds = [[means[0], means[1], means[2]], [stds[0], stds[1], stds[2]]]
            mean_stds_df.at[train_index] = [
                means[0],
                means[1],
                means[2],
                stds[0],
                stds[1],
                stds[2],
            ]

        mean_stds_df.to_csv(save_file)
        print(
            "##### Generated means and stds for training dataset",
            train_index,
            means_stds,
        )

    return means_stds


def train(
    device=torch.device,
    model=models.resnet.ResNet,
    dataloader=torch.utils.data.dataloader.DataLoader,
    criterion=torch.nn.modules.loss.CrossEntropyLoss,
    optimizer=optim.SGD,
    dataset_info=dict,
    epochs=int,
    current_epoch=int,
):
    model.train()

    since_start = time.time()

    min_val_loss = np.Inf
    epochs_no_improve = 0

    train_loss = []

    for e in range(current_epoch, epochs):
        since_epoch_start = time.time()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for images, labels, filenames, sites in dataloader:
            inputs = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_total += len(outputs)

            loss = criterion(outputs, labels)
            running_loss += loss

            loss.backward()
            optimizer.step()
            optimizer = update_optimizer(e, optimizer)

        epoch_loss = running_loss / len(dataloader)

        epoch_acc = running_corrects.double() / running_total

        time_elapsed = time.time() - since_epoch_start

        print(
            "Epoch : {}/{}..".format(e + 1, epochs),
            "Running Loss: {:.6f}".format(running_loss),
            "Epoch Loss: {:.6f}".format(epoch_loss),
            "Accuracy: {:.6f}%".format(epoch_acc * 100),
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            ),
        )
        train_loss.append(running_loss.item())

        # always save the latest checkpoint
        save_model(
            e,
            model,
            optimizer,
            epoch_loss,
            mc.get_checkpoint_path(dataset_info["name"]),
        )

        # if this epoch is the best seen save it
        if epoch_loss < min_val_loss:
            min_val_loss = epoch_loss
            epochs_no_improve = 0
            save_model(
                e,
                model,
                optimizer,
                epoch_loss,
                mc.get_best_model_path(dataset_info["name"]),
                verbose=False,
            )
        else:
            epochs_no_improve += 1

    time_elapsed = time.time() - since_start
    print(
        "#" * 5,
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        ),
    )

    plt.plot(train_loss, label=dataset_info["name"])
    plt.xlabel("Epoch")
    plt.xticks(np.arange(current_epoch, epochs + 1, 5))
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig(mc.TRAINING_LOSS_FIG_FILE)


def calculate_performance_by_result_type(
    df=pd.DataFrame,
    dataset_info=dict,
    output_df=pd.DataFrame,
    result_type=str,
    result_id=str,
    is_invasive=bool,
    train_cnt=int,
    native_train_cnt=int,
    invasive_train_cnt=int,
    test_invasive_cnt=None,
):

    balance_inv_nat_train = (
        invasive_train_cnt / native_train_cnt if invasive_train_cnt else 0
    )
    submission = {
        "model_id": dataset_info["id"],
        "model_name": dataset_info["name"],
        "result_type": result_type,
        "result_id": result_id,
        "is_invasive": is_invasive,
        "train_cnt": train_cnt,
        "native_train_cnt": native_train_cnt,
        "invasive_train_cnt": invasive_train_cnt,
        "balance_inv_nat_train": balance_inv_nat_train,
        "test_invasive_cnt": test_invasive_cnt,
    }

    test_image_cnt = len(df)
    stats = [
        "top_5_correct",
        "top_1_correct",
        "incorrect_invasive",
        "false_alarm",
        "missed_invasive",
    ]
    for stat in stats:
        stat_sum = df[stat].sum()
        submission[stat + "_acc"] = stat_sum / test_image_cnt
        submission[stat + "_cnt"] = stat_sum

    if test_invasive_cnt:
        submission["missed_invasive_rate"] = (
            df["missed_invasive"].sum()
        ) / test_invasive_cnt  # todo
    else:
        submission["missed_invasive_rate"] = None
    print(submission["missed_invasive_rate"], submission["test_invasive_cnt"])

    output_df.loc[len(output_df.index)] = submission


def add_model_prediction_results_to_performance_stats_output(
    dataset_info=dict,
    model_prediction_df=pd.DataFrame,
    performance_df=pd.DataFrame,
    all_img_df=pd.DataFrame,
    evaluate_output_csv=str,
):

    print(
        "##### Caluclating model performance and adding stats to performance dataframe"
    )
    # open model's training dataframe
    train_col = mc.CUSTOM_DATASETS_INFO[dataset_info["train_id"]]["db_col"]
    train_label_col = mc.CUSTOM_DATASETS_INFO[dataset_info["train_id"]]["label_col"]
    train_df = all_img_df[all_img_df[train_col] == True]

    invasive_train_df = train_df[train_df.label_grpd == "INVASIVE"]
    train_invasive_cnt = len(invasive_train_df)

    test_df = all_img_df[all_img_df[dataset_info["db_col"]] == True]  # todo
    print(test_df.columns, test_df.label_grpd.unique())
    invasive_test_df = test_df[test_df.label_grpd == "INVASIVE"]
    test_invasive_cnt = len(invasive_test_df)  # todo
    print("test_invasive_cnt", test_invasive_cnt)

    native_train_cnt = 0
    invasive_train_cnt = 0
    # calculate performance for each class label
    for class_label in model_prediction_df["real_label_text"].unique():
        class_test_df = model_prediction_df[
            model_prediction_df["real_label_text"] == class_label
        ]
        is_invasive = class_test_df["is_invasive"].unique().item()
        class_train_df_cnt = len(train_df[train_df[train_label_col] == class_label])
        if is_invasive:
            invasive_train_cnt += class_train_df_cnt
        else:
            native_train_cnt += class_train_df_cnt
        calculate_performance_by_result_type(
            class_test_df,
            dataset_info,
            performance_df,
            "CLASS",
            class_label,
            is_invasive,
            class_train_df_cnt,
            0,
            0,
        )

    # calculate performance for each camera site
    for site in model_prediction_df["site"].unique():
        site_test_df = model_prediction_df[model_prediction_df["site"] == site]
        site_train_df = train_df[train_df["site"] == site]
        calculate_performance_by_result_type(
            site_test_df,
            dataset_info,
            performance_df,
            "SITE",
            site,
            None,
            len(site_train_df),
            0,
            0,
        )

    # calculate performance across all test images
    calculate_performance_by_result_type(
        model_prediction_df,
        dataset_info,
        performance_df,
        "ALL",
        "ALL",
        None,
        len(train_df),
        native_train_cnt,
        train_invasive_cnt,
        test_invasive_cnt,
    )

    performance_df.to_csv(evaluate_output_csv, index=False)


def evaluate(
    device=torch.device,
    model=models.resnet.ResNet,
    dataloader=DataLoader,
    class_cnt=int,
    dataset_info=dict,
):
    model.eval()

    model_id_list = []
    model_name_list = []
    file_list = []
    site_list = []
    real_label_list = []
    real_label_text_list = []
    prediction_list = []
    is_invasive_list = []
    top_5_correct_list = []
    top_1_correct_list = []
    pred_label_list = []
    pred_label_text_list = []
    incorrect_invasive_list = []
    false_alarm_list = []
    missed_invasive_list = []

    for images, real_labels, filenames, sites in dataloader:
        with torch.no_grad():
            images = images.to(device)
            real_labels = real_labels.to(device)
            output = model(images)
            output = torch.exp(output)

            # Generate top 5 metric for analysis

            # top_5_correct = The real label is in the top 5 predicted labels
            if class_cnt > 5:
                _c, top_5_indices = torch.topk(output, 5)
                temp = real_labels.repeat_interleave(5, dim=0)
                temp = torch.reshape(temp, (int(len(temp) / 5), 5))
                top_5_corrects = torch.sum(
                    torch.eq(temp, top_5_indices), dim=1, dtype=bool
                )
            else:
                top_5_corrects = torch.ones(len(real_labels), dtype=torch.bool)

            # Generate top 1 metrics for analysis

            # top_1_correct = The real label is in the top 1 predicted labels
            _c, top_1_indices = torch.topk(output, 1)
            temp = real_labels.repeat_interleave(1, dim=0)
            temp = torch.reshape(temp, (int(len(temp) / 1), 1))
            # print("0.", temp.is_cuda, top_1_indices.is_cuda, real_labels.is_cuda)
            temp = torch.eq(temp, top_1_indices)
            top_1_corrects = torch.sum(temp, dim=1, dtype=bool)

            print("1. ", len(top_1_indices), end="\r")
            predicted_labels = torch.squeeze(top_1_indices)
            invasive_labels_imgs = torch.gt(real_labels, 1)
            predicted_invasive_labels_imgs = torch.gt(predicted_labels, 1)

            # incorrect_invasive = The predicted label is incorrect and
            # the real label is invasive
            incorrect_invasives = torch.logical_and(
                invasive_labels_imgs, torch.logical_not(top_1_corrects)
            )

            # false_alarm = The real label is not invasive but
            # at least one predicted label is invasive
            false_alarms = torch.logical_and(
                predicted_invasive_labels_imgs, torch.logical_not(invasive_labels_imgs)
            )

            # Images with invasive real labels,
            # where the predicted label is invasive but NOT the correct invasive label,
            # do NOT qualify as a missed invasive
            # missed_invasives = is_invasive and predicted_class < 2
            missed_invasives = torch.logical_and(
                invasive_labels_imgs, torch.logical_not(predicted_invasive_labels_imgs)
            )

            model_id_list += [dataset_info["id"] for label in real_labels]
            model_name_list += [dataset_info["name"] for label in real_labels]
            file_list += [f for f in filenames]
            site_list += [site for site in sites]
            real_label_list += [label.item() for label in real_labels]
            real_label_text_list += [
                dataset_info["class_list"][int(label)] for label in real_labels
            ]
            is_invasive_list += [label.item() >= 2 for label in real_labels]
            prediction_list += [pred.tolist() for pred in output]
            top_5_correct_list += [top_5.item() for top_5 in top_5_corrects]
            pred_label_list += [label.item() for label in predicted_labels]
            pred_label_text_list += [
                dataset_info["class_list"][int(label)] for label in predicted_labels
            ]
            top_1_correct_list += [top_1.item() for top_1 in top_1_corrects]
            incorrect_invasive_list += [r.item() for r in incorrect_invasives]
            false_alarm_list += [r.item() for r in false_alarms]
            missed_invasive_list += [r.item() for r in missed_invasives]

    sub_item = {
        "model_id": model_id_list,
        "model_name": model_name_list,
        "file": file_list,
        "site": site_list,
        "preds": prediction_list,
        "real_label": real_label_list,
        "real_label_text": real_label_text_list,
        "pred_label_text_list": pred_label_text_list,
        "is_invasive": is_invasive_list,
        "top_5_correct": top_5_correct_list,
        "top_1_correct": top_1_correct_list,
        "predicted_label": pred_label_list,
        "predicted_label_text": pred_label_text_list,
        "incorrect_invasive": incorrect_invasive_list,
        "false_alarm": false_alarm_list,
        "missed_invasive": missed_invasive_list,
    }

    df = pd.DataFrame(sub_item)
    return df


##########################################################
# Main


def main():

    ##########################################################
    # Configuration

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_testing_csv",
        type=str,
        default=mc.DEFAULT_TRAIN_TEST_CSV_FILE,
        help="CSV file generated from the prepare_classification_datasets.py script.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/detections",
        help="Directory containing cropped images",
    )

    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume training from saved checkpoint",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate models with testing datasets"
    )

    args = parser.parse_args()

    # Check arguments
    training_testing_csv = args.training_testing_csv
    assert os.path.exists(training_testing_csv), (
        training_testing_csv + " does not exist"
    )

    dataset_dir = args.dataset_dir
    assert os.path.exists(dataset_dir), dataset_dir + " does not exist"

    resume_training = args.resume_training
    evaluate_models = args.evaluate

    means_stds_csv = "./training/means_stds.csv"
    if evaluate_models:
        assert os.path.exists(
            means_stds_csv
        ), "No means_stds.csv file found first run training.py without --evaluate flag."

    evaluate_output_csv = "./model_performance_results.csv"

    print("##########################################################")
    print("##### START  #####\n")
    print("dataset dir          :", dataset_dir)
    print("training_testing csv :", training_testing_csv)
    print("resume training      :", resume_training)

    if os.path.exists(means_stds_csv):
        print("\n##### Loading existing means stds csv file.")
        mean_stds_df = pd.read_csv(means_stds_csv)
        mean_stds_df = mean_stds_df.set_index(["train_index"])
    else:
        print("\n##### Creating new means stds csv.")
        mean_stds_df = pd.DataFrame(columns=mc.MEANS_STDS_DF_COLS)
        mean_stds_df = mean_stds_df.set_index(["train_index"])

    df = pd.read_csv(training_testing_csv)
    df = df.set_index(["file"])

    # select device use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets_indexes = (
        mc.TEST_DATASET_INDEXES if evaluate_models else mc.TRAIN_DATASET_INDEXES
    )
    if evaluate_models:
        evaluation_output_df = pd.DataFrame(columns=mc.PERFORMANCE_DF_COLS)

    for dataset_index in datasets_indexes:  # [datasets_indexes[-1]]:
        print("")
        dataset_info = mc.CUSTOM_DATASETS_INFO[dataset_index]
        train_index = dataset_info["train_id"]

        """means_stds = get_means_stds(
            df, dataset_dir, dataset_info, mean_stds_df, means_stds_csv
        )

        data_transform = transforms.Compose(
            [
                transforms.ColorJitter(),
                transforms.RandomCrop(mc.CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means_stds[0], means_stds[1]),
            ]
        )

        dataset = CustomDataset(df, dataset_dir, dataset_info, transform=data_transform)"""
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # Note: Getting the number of classes is ugly because several
        # train/test combinations have different # of classes as a point of research.
        class_cnt = len(mc.CUSTOM_DATASETS_INFO[train_index]["class_map"])
        model = get_model(class_cnt)
        model.to(device)  # shifting model to correct device

        # Norouzzadeh et al., 2018, Tabak et al., 2019
        optimizer = optim.SGD(
            model.parameters(),
            lr=mc.LEARNING_RATE,
            weight_decay=mc.WEIGHT_DECAY,
            momentum=mc.MOMENTUM,
        )
        # Norouzzadeh et al., 2018, Tabak et al., 2019
        criterion = nn.CrossEntropyLoss()

        if evaluate_models:  # evaluate models with test datasets

            model_path = mc.get_best_model_path(
                mc.CUSTOM_DATASETS_INFO[train_index]["name"]
            )
            if os.path.exists(model_path):
                print("##### EVALUATING using model from checkpoint", model_path)

                # load model information from checkpoint
                checkpoint = torch.load(model_path)
                current_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            else:
                print("##### ERROR Could not find model checkpoint", model_path)

            means_stds = get_means_stds(
                df, dataset_dir, dataset_info, mean_stds_df, means_stds_csv
            )

            data_transform = transforms.Compose(
                [
                    transforms.ColorJitter(),
                    transforms.RandomCrop(mc.CROP_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(means_stds[0], means_stds[1]),
                ]
            )

            dataset = CustomDataset(
                df, dataset_dir, dataset_info, transform=data_transform
            )
            dataloader = DataLoader(
                dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4
            )

            # dataset.train_data.to(torch.device("cuda:0"))  # put data into GPU entirely
            # dataset.train_labels.to(torch.device("cuda:0"))

            model_prediction_df = evaluate(
                device, model, dataloader, class_cnt, dataset_info
            )
            model_prediction_df.to_csv(
                mc.get_evaluation_path(dataset_info["name"]), index=False
            )

            add_model_prediction_results_to_performance_stats_output(
                dataset_info,
                model_prediction_df,
                evaluation_output_df,
                df,
                evaluate_output_csv,
            )

        else:  # train models

            print("##### STARTING Training model", dataset_info["name"])
            model_path = mc.get_checkpoint_path(dataset_info["name"])
            if resume_training and os.path.exists(model_path):
                print("##### RESUMING training with", model_path)

                # load model information from checkpoint
                checkpoint = torch.load(model_path)
                current_epoch = checkpoint["epoch"]
                if current_epoch == mc.NUM_EPOCHS - 1:
                    print("already trained for ", current_epoch + 1, "epochs")
                    continue
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            else:
                print("##### INITIATING training fresh")
                current_epoch = 0

            means_stds = get_means_stds(
                df, dataset_dir, dataset_info, mean_stds_df, means_stds_csv
            )

            data_transform = transforms.Compose(
                [
                    transforms.ColorJitter(),
                    transforms.RandomCrop(mc.CROP_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(means_stds[0], means_stds[1]),
                ]
            )

            dataset = CustomDataset(
                df, dataset_dir, dataset_info, transform=data_transform
            )
            dataloader = DataLoader(
                dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4
            )
            train(
                device,
                model,
                dataloader,
                criterion,
                optimizer,
                dataset_info,
                mc.NUM_EPOCHS,
                current_epoch,
            )
            print("##### FINISHED Training model", dataset.name)

        print("#" * 80)

    print("##### END #####\n")
    print("\n##########################################################")


if __name__ == "__main__":
    main()
