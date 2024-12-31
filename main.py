import os
import tempfile
from lightning.pytorch.plugins.io import AsyncCheckpointIO
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import wandb
import argparse, yaml
from attacks import PGD
from utils import OssCheckpointIO, get_args, get_md5, get_oss
import dataset, model

def exp(model, dm, cfg, ckpt_path):
    md5 = get_md5(cfg)
    print("MD5:", md5)
    wandb.config.update({"md5": md5})

    bucket = get_oss()
    oss_checkpoint_io = OssCheckpointIO(bucket)

    logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(
        monitor="acc",
        dirpath="checkpoints/",
        filename=md5,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
        save_weights_only=True,
        every_n_epochs=20,
    )
    early_stopping = EarlyStopping(
        monitor="acc", patience=cfg["patience"], mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    trainer = Trainer(
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        max_epochs=-1,
        gradient_clip_algorithm="norm",
        plugins=[AsyncCheckpointIO(oss_checkpoint_io)],
    )

    if ckpt_path is not None and bucket.object_exists(ckpt_path):
        mode = model.adv_mode
        key = os.path.relpath(ckpt_path, os.getcwd())
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, os.path.basename(key))
            print(f"Downloading {key} to {fp}")
            bucket.get_object_to_file(key, fp)
            model = model.__class__.load_from_checkpoint(fp)
            model.adv_mode = mode
        print("Load from checkpoint: ", ckpt_path)
    trainer.fit(model, dm)

    if not model.adv_mode:
        eps = [0, 0.001, 0.01, 0.1, 1.0]
    else:
        eps = list(range(5))
    accs, acc5s, acc10s, asrs, asr5s, asr10s = [], [], [], [], [], []
    for i in eps:
        if i > 0:
            model.eval_atk = PGD(model, eps=i / 255.0, alpha=i / 2550.0, steps=10)
            model.adv_mode = True
        else:
            model.adv_mode = False
        ret = trainer.test(model, datamodule=dm, ckpt_path="best")[0]
        acc, acc5, acc10 = ret["acc"], ret["acc5"], ret["acc10"]
        accs.append(acc), acc5s.append(acc5), acc10s.append(acc10)
        if i == 0:
            ca, ca5, ca10 = acc, acc5, acc10
            asr, asr5, asr10 = 0, 0, 0
        else:
            asr = (ca - acc) / ca
            asr5 = (ca5 - acc5) / ca5
            asr10 = (ca10 - acc10) / ca10
        asrs.append(asr), asr5s.append(asr5), asr10s.append(asr10)

    wandb.run.summary["eps"] = eps
    wandb.run.summary["Acc@1"] = accs
    wandb.run.summary["Acc@5"] = acc5s
    wandb.run.summary["Acc@10"] = acc10s
    wandb.run.summary["ASR@1"] = asrs
    wandb.run.summary["ASR@5"] = asr5s
    wandb.run.summary["ASR@10"] = asr10s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model = getattr(config["model"])(**config)
    dm = cli.datamodule
    args = get_args(cli.config.as_dict())
    args["model"] = model.__class__.__name__
    args["dataset"] = dm.__class__.__name__
    wandb.init(project="RobustCBM", config=args, tags=[model.__class__.__name__, dm.__class__.__name__])
    exp(model, dm, args, cli.config["ckpt_path"])
