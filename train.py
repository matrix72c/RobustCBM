import argparse
import torch
import torch.nn as nn
import os
from aim import Run
import yaml
import pandas as pd

from utils import AverageMeter, set_seed
from attack import attack


def train(conf):
    # set seed
    set_seed(conf["seed"])

    # load data
    train_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=224 if conf["model"] == "resnet50" else 299,
        is_train=True,
    )
    test_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=224 if conf["model"] == "resnet50" else 299,
        is_train=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
    )

    # load model
    model = getattr(
        __import__("models." + conf["model"], fromlist=[""]), conf["model"]
    )(**conf["model_args"])

    # load loss function
    loss_fn = getattr(nn, conf["loss_fn"])()
    if conf["model_args"]["base"] == "inceptionv3":
        if conf["use_imbalance_ratio"]:
            attr_loss_fn = [
                getattr(nn, conf["attr_loss_fn"])(
                    weight=torch.FloatTensor([ratio]).cuda()
                )
                for ratio in train_dataset.imbalance_ratio
            ]
        else:
            attr_loss_fn = [
                getattr(nn, conf["attr_loss_fn"])()
                for _ in train_dataset.imbalance_ratio
            ]
    else:
        attr_loss_fn = getattr(nn, conf["attr_loss_fn"])()

    # train
    model.cuda()

    run = Run(experiment=conf["experiment"], repo=os.getenv("AIM_REPO"))
    run[...] = conf
    if conf["model_args"]["use_pretrained"] is True:
        pretrained_mode = "pretrained_"
    elif conf["model_args"]["use_pretrained"] is False:
        pretrained_mode = "nopretrained_"
    else:
        pretrained_mode = "advpretrained_"
    adv_mode = conf["use_adv"] if len(conf["use_adv"]) > 0 else "noadv"
    noise_mode = conf["use_noise"] if len(conf["use_noise"]) > 0 else "nonoise"
    run.description = (
        conf["trainer"] + "_" + pretrained_mode + adv_mode + "_" + noise_mode + "_"
    )
    run_hash = run.hash
    f = open("results/{}.yaml".format(run_hash), "w", encoding="utf-8")
    yaml.dump(conf, f, allow_unicode=True)
    train_log = []
    if conf["trainer"] == "Seperate":
        for epoch in range(conf["epochs"]):
            model.train()
            label_loss_meter = AverageMeter()
            label_acc_meter = AverageMeter()
            attr_loss_meter = AverageMeter()
            attr_acc_meter = AverageMeter()
            for img, label, attr in train_loader:
                kwargs = {
                    "img": img,
                    "label": label,
                    "attr": attr,
                    "model": model,
                    "model_base": conf["model_args"]["base"],
                    "label_loss_meter": label_loss_meter,
                    "label_acc_meter": label_acc_meter,
                    "attr_loss_meter": attr_loss_meter,
                    "attr_acc_meter": attr_acc_meter,
                    "optimizer_type": conf["optimizer"],
                    "optimizer_args": conf["optimizer_args"],
                    "scheduler_type": conf["scheduler"],
                    "scheduler_args": conf["scheduler_args"],
                    "loss_fn": loss_fn,
                    "attr_loss_fn": attr_loss_fn,
                    "attr_loss_weight": conf["attr_loss_weight"],
                    "use_adv": conf["use_adv"],
                    "use_noise": conf["use_noise"],
                    "adv_v2v_eps": conf["adv_v2v_eps"],
                    "noise_eps": conf["noise_eps"],
                }
                getattr(
                    __import__("trainers.Seperate", fromlist=[""]),
                    "image2concept",
                )(**kwargs)
            run.track(name="attr_loss", value=attr_loss_meter.avg, epoch=epoch)
            run.track(name="attr_acc", value=attr_acc_meter.avg, epoch=epoch)
            train_log.append([attr_loss_meter.avg, attr_acc_meter.avg, 0, 0])
            if attr_acc_meter.avg > 0.995:
                break
        for epoch in range(conf["epochs"]):
            model.train()
            label_loss_meter = AverageMeter()
            label_acc_meter = AverageMeter()
            attr_loss_meter = AverageMeter()
            attr_acc_meter = AverageMeter()
            for img, label, attr in train_loader:
                kwargs = {
                    "img": img,
                    "label": label,
                    "attr": attr,
                    "model": model,
                    "model_base": conf["model_args"]["base"],
                    "label_loss_meter": label_loss_meter,
                    "label_acc_meter": label_acc_meter,
                    "attr_loss_meter": attr_loss_meter,
                    "attr_acc_meter": attr_acc_meter,
                    "optimizer_type": conf["optimizer"],
                    "optimizer_args": conf["optimizer_args"],
                    "scheduler_type": conf["scheduler"],
                    "scheduler_args": conf["scheduler_args"],
                    "loss_fn": loss_fn,
                    "attr_loss_fn": attr_loss_fn,
                    "attr_loss_weight": conf["attr_loss_weight"],
                    "use_adv": conf["use_adv"],
                    "use_noise": conf["use_noise"],
                    "adv_v2v_eps": conf["adv_v2v_eps"],
                    "noise_eps": conf["noise_eps"],
                    "seperate_mode": conf["seperate_mode"],
                }
                getattr(
                    __import__("trainers.Seperate", fromlist=[""]),
                    "concept2label",
                )(**kwargs)
            run.track(name="label_loss", value=label_loss_meter.avg, epoch=epoch)
            run.track(name="label_acc", value=label_acc_meter.avg, epoch=epoch)
            train_log[epoch][2] = label_loss_meter.avg
            train_log[epoch][3] = label_acc_meter.avg
            if label_acc_meter.avg > 0.80:
                torch.save(
                    model.state_dict(),
                    "checkpoints/"
                    + run.description
                    + str("{:.2f}".format(label_acc_meter.avg * 100))
                    + "_"
                    + run_hash
                    + ".pth",
                )
            if label_acc_meter.avg > 0.95:
                models = [f for f in os.listdir("checkpoints/") if run_hash in f]
                min_diff = float("inf")
                file_to_keep = None
                for file in models:
                    parts = file.split("_")
                    if len(parts) > 2 and parts[-2].replace(".", "", 1).isdigit():
                        acc = float(parts[-2])
                        diff = abs(acc - 90)
                        if diff < min_diff:
                            min_diff = diff
                            file_to_keep = file
                for file in models:
                    if file != file_to_keep and ".pth" in file:
                        os.remove(os.path.join("checkpoints", file))
                df = pd.DataFrame(
                    train_log,
                    columns=["attr_loss", "attr_acc", "label_loss", "label_acc"],
                )
                df.to_csv("results/train_" + run_hash + ".csv", index=False)
                attack(run_hash, run)
                return
    for epoch in range(conf["epochs"]):
        model.train()
        label_loss_meter = AverageMeter()
        label_acc_meter = AverageMeter()
        attr_loss_meter = AverageMeter()
        attr_acc_meter = AverageMeter()
        for img, label, attr in train_loader:
            kwargs = {
                "img": img,
                "label": label,
                "attr": attr,
                "model": model,
                "model_base": conf["model_args"]["base"],
                "label_loss_meter": label_loss_meter,
                "label_acc_meter": label_acc_meter,
                "attr_loss_meter": attr_loss_meter,
                "attr_acc_meter": attr_acc_meter,
                "optimizer_type": conf["optimizer"],
                "optimizer_args": conf["optimizer_args"],
                "scheduler_type": conf["scheduler"],
                "scheduler_args": conf["scheduler_args"],
                "loss_fn": loss_fn,
                "attr_loss_fn": attr_loss_fn,
                "attr_loss_weight": conf["attr_loss_weight"],
                "use_adv": conf["use_adv"],
                "use_noise": conf["use_noise"],
                "adv_v2v_eps": conf["adv_v2v_eps"],
                "noise_eps": conf["noise_eps"],
            }
            getattr(
                __import__("trainers." + conf["trainer"], fromlist=[""]),
                conf["trainer"],
            )(**kwargs)
        run.track(name="label_loss", value=label_loss_meter.avg, epoch=epoch)
        run.track(name="label_acc", value=label_acc_meter.avg, epoch=epoch)
        run.track(name="attr_loss", value=attr_loss_meter.avg, epoch=epoch)
        run.track(name="attr_acc", value=attr_acc_meter.avg, epoch=epoch)
        train_log.append(
            [
                attr_loss_meter.avg,
                attr_acc_meter.avg,
                label_loss_meter.avg,
                label_acc_meter.avg,
            ]
        )
        if label_acc_meter.avg > 0.80:
            torch.save(
                model.state_dict(),
                "checkpoints/"
                + run.description
                + str("{:.2f}".format(label_acc_meter.avg * 100))
                + "_"
                + run_hash
                + ".pth",
            )
        if label_acc_meter.avg > 0.95:
            models = [f for f in os.listdir("checkpoints/") if run_hash in f]
            min_diff = float("inf")
            file_to_keep = None
            for file in models:
                parts = file.split("_")
                if len(parts) > 2 and parts[-2].replace(".", "", 1).isdigit():
                    acc = float(parts[-2])
                    diff = abs(acc - 90)
                    if diff < min_diff:
                        min_diff = diff
                        file_to_keep = file
            for file in models:
                if file != file_to_keep and ".pth" in file:
                    os.remove(os.path.join("checkpoints", file))
            df = pd.DataFrame(
                train_log, columns=["attr_loss", "attr_acc", "label_loss", "label_acc"]
            )
            df.to_csv("results/train_" + run_hash + ".csv", index=False)
            attack(run_hash, run)
            return
        # if (epoch + 1) % 100 == 0:
        #     acc, attr_acc = test_acc(model, test_loader)
        #     run.track(name="test_acc", value=acc, epoch=epoch)
        #     run.track(name="test_attr_acc", value=attr_acc, epoch=epoch)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_model = model.state_dict()
        # if (epoch + 1) % 500 == 0:
        #     # delete previous checkpoints
        #     for file in os.listdir("checkpoints"):
        #         if file.endswith(run_hash + ".pth"):
        #             os.remove(os.path.join("checkpoints", file))
        #     torch.save(
        #         best_model,
        #         "checkpoints/"
        #         + pretrained_mode
        #         + conf["model_args"]["base"]
        #         + "_"
        #         + ("adv_" if conf["use_adv"] else "")
        #         + ("noise_" if conf["add_noise"] else "")
        #         + str("{:.2f}".format(best_acc * 100))
        #         + "_"
        #         + run_hash
        #         + ".pth",
        #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    args = parser.parse_args()
    f = open(args.config, "r", encoding="utf-8")
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    train(conf)
