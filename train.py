import argparse
import torch
import torch.nn
import os
from aim import Run
import yaml

from utils import cal_class_imbalance_weights, set_seed, gen_mask
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
    if conf["imbalance"]:
        train_dataset.imbalance_ratio = cal_class_imbalance_weights(
            train_loader, conf["num_attributes"]
        )
        test_dataset.imbalance_ratio = train_dataset.imbalance_ratio
    if conf["mask"]:
        train_dataset.mask = gen_mask(train_loader, conf["num_attributes"], conf["n_features"])
        test_dataset.mask = train_dataset.mask

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

    # load model
    model = getattr(
        __import__("models." + conf["model"], fromlist=[""]), conf["model"]
    )(conf, train_loader, test_loader)
    
    # train
    model.cuda()
    best_acc = 0
    best_acc_epoch = 0
    best_model = None
    for epoch in range(conf["epochs"]):
        res = model.run_epoch(train_loader)
        log = "Epoch: {} ".format(epoch)
        for key, value in res.items():
            run.track(name=key, value=value, epoch=epoch)
            log += "{}: {:.4f} ".format(key, value)
        acc = model.test(test_loader)
        log += "test_acc: {:.4f}".format(acc)
        run.track(name="test_acc", value=acc, epoch=epoch)
        if (epoch + 1) % 10 == 0:
            adv_acc = model.test(test_loader, is_adv=True)
            log += " adv_acc: {:.4f}".format(adv_acc)
            run.track(name="adv_acc", value=adv_acc, epoch=epoch)
        print(log)
        if acc > best_acc + 0.01:
            best_acc = acc
            best_acc_epoch = epoch
            best_model = model.state_dict().copy()
        if epoch - best_acc_epoch > 20:
            print("Early stopping at epoch {}".format(epoch))
            break
        
    if best_model is not None:
        torch.save(
                best_model,
                "checkpoints/"
                + run.description
                + "_"
                + str("{:.2f}".format(best_acc * 100))
                + "_"
                + run_hash
                + ".pth",
            )
    adv_acc = model.test(test_loader, is_adv=True)
    print("adv_acc: {:.4f}".format(adv_acc))
    run.track(name="adv_acc", value=adv_acc, epoch=epoch)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    args = parser.parse_args()
    f = open(args.config, "r", encoding="utf-8")
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    train(conf)
