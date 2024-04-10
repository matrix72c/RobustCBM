import argparse
import os
import numpy as np
import torch
import yaml
from torchattacks import PGD
from aim import Run
import pandas as pd

from utils import AverageMeter, set_seed, cal_top_k


def attack_train(run_hash, run=None):
    f = open("results/{}.yaml".format(run_hash), "r", encoding="utf-8")
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)

    # set seed
    set_seed(conf["seed"])

    # load data
    test_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=224,
        is_train=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
    )

    # create model
    model = getattr(
        __import__("models." + conf["model"], fromlist=[""]), conf["model"]
    )(**conf["model_args"])

    # Get model path under the same experiment settings
    for file in os.listdir("checkpoints"):
        if run_hash in file:
            model_path = os.path.join("checkpoints", file)

    # load model
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    # attack
    attack_log = [[0 for _ in range(10)] for _ in range(11)]
    for i in range(11):
        atk = PGD(model, eps=i / 255, alpha=2 / 225, steps=10, random_start=True)
        atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        label_acc = [AverageMeter() for _ in range(10)]
        acc = np.zeros(10)
        data_len = 0
        for img, label, attr in test_loader:
            img, label, attr = img.cuda(), label.cuda(), attr.cuda()
            batch_len = img.size(0)
            model.use_adv = "image2label"
            adv_img = atk(img, label) if i > 0 else img
            with torch.no_grad():
                label_pred = model(img)
                adv_label_pred = model(adv_img)
            label_pred = label_pred.cpu()
            adv_label_pred = adv_label_pred.cpu()
            label_pred = label_pred.max(1, keepdim=True)[1]
            for j in range(10):
                acc[j] += (batch_len - cal_top_k(adv_label_pred,label_pred,k=j+1).item())
                label_acc[j].update(
                    1
                    - cal_top_k(adv_label_pred, label_pred, k=j + 1).item()
                    / label.size(0),
                    label.size(0),
                )
            data_len += batch_len
        print(i, " ", acc, " / ", data_len)
        for j in range(10):
            attack_log[i][j] = label_acc[j].avg
            print("eps: {}, pgd_train_top_{}: {}".format(i, j + 1, label_acc[j].avg))
            if run is not None:
                run.track(
                    name="pgd_label_acc_top_{}".format(j + 1),
                    value=label_acc[j].avg,
                    epoch=i,
                )
                # run.track(name="pgd_attr_acc_top_{}".format(j + 1), value=attr_acc[j].avg, epoch=i)
    df = pd.DataFrame(attack_log, columns=["pgd_train_top_{}".format(i + 1) for i in range(10)])
    df.to_csv("results/{}_{}_train.csv".format(run.description, run_hash), index=False)

def attack_eval(run_hash, run=None):
    f = open("results/{}.yaml".format(run_hash), "r", encoding="utf-8")
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)

    # set seed
    set_seed(conf["seed"])

    # load data
    test_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=224,
        is_train=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
    )

    # create model
    model = getattr(
        __import__("models." + conf["model"], fromlist=[""]), conf["model"]
    )(conf)

    # Get model path under the same experiment settings
    for file in os.listdir("checkpoints"):
        if run_hash in file:
            model_path = os.path.join("checkpoints", file)
    print(model_path)

    # load model
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # attack
    attack_log = [[0 for _ in range(10)] for _ in range(11)]
    for i in range(11):
        atk = PGD(model, eps=i / 255, alpha=2 / 225, steps=10, random_start=True)
        atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        label_acc = [AverageMeter() for _ in range(10)]
        acc = np.zeros(10)
        data_len = 0
        for img, label, attr in test_loader:
            img, label, attr = img.cuda(), label.cuda(), attr.cuda()
            batch_len = img.size(0)
            model.atk_mode = True
            adv_img = atk(img, label) if i > 0 else img
            with torch.no_grad():
                label_pred = model(img)
                adv_label_pred = model(adv_img)
            label_pred = label_pred.cpu()
            adv_label_pred = adv_label_pred.cpu()
            label_pred = label_pred.max(1, keepdim=True)[1]
            for j in range(10):
                acc[j] += (batch_len - cal_top_k(adv_label_pred,label_pred,k=j+1).item())
                label_acc[j].update(
                    1
                    - cal_top_k(adv_label_pred, label_pred, k=j + 1).item()
                    / label.size(0),
                    label.size(0),
                )
            data_len += batch_len
        print(i, " ", acc, " / ", data_len)
        for j in range(10):
            attack_log[i][j] = label_acc[j].avg
            print("eps: {}, pgd_eval_top_{}: {}".format(i, j + 1, label_acc[j].avg))
            if run is not None:
                run.track(
                    name="pgd_label_acc_top_{}".format(j + 1),
                    value=label_acc[j].avg,
                    epoch=i,
                )
                # run.track(name="pgd_attr_acc_top_{}".format(j + 1), value=attr_acc[j].avg, epoch=i)
    df = pd.DataFrame(attack_log, columns=["pgd_eval_top_{}".format(i + 1) for i in range(10)])
    df.to_csv("results/{}_{}_eval.csv".format(run.description, run_hash), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", type=str, default="")
    args = parser.parse_args()
    try:
        run = Run(run_hash=args.hash, repo=os.getenv("AIM_REPO"))
    except:
        run = None
    # attack_train(args.hash, run)
    attack_eval(args.hash, run)
