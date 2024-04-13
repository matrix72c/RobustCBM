import argparse
import torch
import torch.nn
import os
from aim import Run
import yaml

from utils import set_seed
from attack import attack_eval


def train(conf):
    # set seed
    set_seed(conf["seed"])

    # load data
    train_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=conf["resol"],
        is_train=True,
    )
    test_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=conf["resol"],
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
    )(conf)

    run = Run(experiment=conf["experiment"], repo=os.getenv("AIM_REPO"))
    run[...] = conf
    if conf["use_pretrained"] is True:
        pretrained_mode = "pretrained_"
    elif conf["use_pretrained"] is False:
        pretrained_mode = "nopretrained_"
    else:
        pretrained_mode = "advpretrained_"
    adv_mode = conf["use_adv"] if len(conf["use_adv"]) > 0 else "noadv"
    noise_mode = conf["use_noise"] if len(conf["use_noise"]) > 0 else "nonoise"
    run.description = (
        conf["model"]
        + "_"
        + conf["dataset"]
        + "_"
        + conf["mode"]
        + "_"
        + pretrained_mode
        + adv_mode
        + "_"
        + noise_mode
    )
    run_hash = run.hash
    f = open("results/configs/{}.yaml".format(run_hash), "w", encoding="utf-8")
    yaml.dump(conf, f, allow_unicode=True)

    # train
    model.cuda()
    for epoch in range(conf["epochs"]):
        res = model.run_epoch(train_loader)
        log = "Epoch: {} ".format(epoch)
        for key, value in res.items():
            run.track(name=key, value=value, epoch=epoch)
            log += "{}: {:.4f} ".format(key, value)
        print(log)
        if res["label_acc"] > 0.80:
            torch.save(
                model.state_dict(),
                "checkpoints/"
                + run.description
                + "_"
                + str("{:.2f}".format(res["label_acc"] * 100))
                + "_"
                + run_hash
                + ".pth",
            )
        if res["label_acc"] > 0.9:
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
            print(file_to_keep)
            for file in models:
                if file != file_to_keep and ".pth" in file:
                    os.remove(os.path.join("checkpoints", file))
            attack_eval(run_hash, run)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    args = parser.parse_args()
    f = open(args.config, "r", encoding="utf-8")
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    train(conf)
