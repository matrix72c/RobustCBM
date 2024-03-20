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
            img = atk(img, label) if i > 0 else img
            model.use_adv = ""
            with torch.no_grad():
                attr_pred, label_pred = model(img)
                label_pred = torch.argmax(label_pred, dim=1)
                correct = torch.sum(label_pred == label).int().sum().item()
                num = len(label)
                label_acc.update(correct / num, num)
                attr_pred = torch.sigmoid(attr_pred).ge(0.5)
                attr_correct = torch.sum(attr_pred == attr).int().sum().item()
                attr_num = attr.shape[0] * attr.shape[1]
                attr_acc.update(attr_correct / attr_num, attr_num)
        run.track(name="pgd_label_acc", value=label_acc.avg, epoch=i)
        run.track(name="pgd_attr_acc", value=attr_acc.avg, epoch=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", type=str, default="")
    args = parser.parse_args()
    main(args.hash)
