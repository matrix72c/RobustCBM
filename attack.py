import argparse
import os
import torch
from torchattacks import PGD
from aim import Run
import yaml

from utils import AverageMeter, set_seed


def main(hash):
    run = Run(run_hash=hash, repo=os.getenv("AIM_REPO"))
    conf = run["hparams"]

    # set seed
    set_seed(conf["seed"])

    # load data
    test_dataset = getattr(
        __import__("dataprovider." + conf["dataset"], fromlist=[""]), conf["dataset"]
    )(
        conf["data_path"],
        resol=224 if conf["model"] == "resnet50" else 299,
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
        if hash in file:
            model_path = os.path.join("checkpoints", file)

    # load model
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # attack
    for i in range(11):
        atk = PGD(model, eps=i / 255, alpha=2 / 225, steps=2, random_start=True)
        atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        attr_acc = AverageMeter()
        label_acc = AverageMeter()
        for img, label, attr in test_loader:
            img, label, attr = img.cuda(), label.cuda(), attr.cuda()
            model.use_adv = "image2label"
            adv_img = atk(img, label) if i > 0 else img
            model.use_adv = ""
            with torch.no_grad():
                attr_pred, label_pred = model(img)
                adv_attr_pred, adv_label_pred = model(adv_img)
            attr_pred = attr_pred.cpu()
            label_pred = label_pred.cpu()
            adv_attr_pred = adv_attr_pred.cpu()
            adv_label_pred = adv_label_pred.cpu()
            label_pred = label_pred.max(1, keepdim=True)[1]
            label_acc.update(label_pred.eq(label.view_as(label_pred)).sum().item(), label.size(0))
        run.track(name="pgd_label_acc", value=label_acc.avg, epoch=i)
        run.track(name="pgd_attr_acc", value=attr_acc.avg, epoch=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", type=str, default="")
    args = parser.parse_args()
    main(args.hash)
