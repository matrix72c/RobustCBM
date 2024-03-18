import argparse
import torch
import torch.nn as nn
import os
from aim import Run
import yaml
from torchattacks import PGD

from utils import AverageMeter, set_seed, test_acc


def main(config):
    f = open(config, "r", encoding="utf-8")
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)

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
    if conf["use_imbalance_ratio"]:
        attr_loss_fn = [
            getattr(nn, conf["attr_loss_fn"])(weight=torch.FloatTensor([ratio]).cuda())
            for ratio in train_dataset.imbalance_ratio
        ]
    else:
        attr_loss_fn = [
            getattr(nn, conf["attr_loss_fn"])()
            for _ in train_dataset.imbalance_ratio
        ]

    # train
    model.cuda()
    best_model = None
    best_acc = 0

    run = Run(experiment=conf["experiment"], repo=os.getenv("AIM_REPO"))
    run["hparams"] = conf
    run_hash = run.hash
    for epoch in range(conf["epochs"]):
        model.train()
        label_loss_meter = AverageMeter()
        label_acc_meter = AverageMeter()
        attr_loss_meter = AverageMeter()
        attr_acc_meter = AverageMeter()
        for img, label, attr in train_loader:
            if conf["use_adv"]:
                model.use_adv = True
                atk = PGD(model, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True)
                atk.set_normalization_used(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                adv_img = atk(img, label).cpu()
                adv_label = label.clone().detach().cpu()
                adv_attr = attr.clone().detach().cpu()
                img = torch.cat([img, adv_img], dim=0)
                label = torch.cat([label, adv_label], dim=0)
                attr = torch.cat([attr, adv_attr], dim=0)
                model.use_adv = False
            if conf["add_noise"]:
                noise = torch.randn_like(img) * 0.1
                img += noise
            getattr(
                __import__("trainers." + conf["trainer"], fromlist=[""]),
                conf["trainer"],
            )(
                img,
                label,
                attr,
                model,
                conf["model_args"]["base"],
                label_loss_meter,
                label_acc_meter,
                attr_loss_meter,
                attr_acc_meter,
                conf["optimizer"],
                conf["optimizer_args"],
                conf["scheduler"],
                conf["scheduler_args"],
                loss_fn,
                attr_loss_fn,
                conf["attr_loss_weight"],
            )
        run.track(name="label_loss", value=label_loss_meter.avg, epoch=epoch)
        run.track(name="label_acc", value=label_acc_meter.avg, epoch=epoch)
        run.track(name="attr_loss", value=attr_loss_meter.avg, epoch=epoch)
        run.track(name="attr_acc", value=attr_acc_meter.avg, epoch=epoch)
        if conf["model_args"]["use_pretrained"] is True:
            pretrained_mode = "pretrained_"
        elif conf["model_args"]["use_pretrained"] is False:
            pretrained_mode = ""
        else:
            pretrained_mode = "selfpretrained_"
        if label_acc_meter.avg > 85:
            torch.save(
                model.state_dict(),
                "checkpoints/"
                + pretrained_mode
                + conf["model_args"]["base"]
                + "_"
                + ("adv_" if conf["use_adv"] else "")
                + ("noise_" if conf["add_noise"] else "")
                + str("{:.2f}".format(label_acc_meter.avg * 100))
                + "_"
                + run_hash
                + ".pth",
            )
        if label_acc_meter.avg > 95:
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
    main(args.config)
