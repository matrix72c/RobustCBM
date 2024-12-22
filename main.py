from lightning.pytorch.plugins.io import AsyncCheckpointIO
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import wandb
from attacks import PGD
from utils import OssCheckpointIO, get_args, get_md5, get_oss

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--patience", default=100)
        parser.add_argument("--train", default=False)
        parser.link_arguments("data.init_args.num_concepts", "model.init_args.num_concepts")
        parser.add_argument("--sweep_id", default=None)

def exp(model, dm, cfg, train=True):
    md5 = get_md5(cfg)
    print("MD5:", md5)
    wandb.config.update({"md5": md5})

    bucket = get_oss()
    oss_checkpoint_io = OssCheckpointIO(bucket)

    logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename=md5,
        save_top_k=1,
        mode="min",
        enable_version_counter=False,
        save_weights_only=True,
        every_n_epochs=10,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=cfg["patience"], mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    trainer = Trainer(
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        max_epochs=-1,
        gradient_clip_algorithm="norm",
        gradient_clip_val=5.0,
        plugins=[AsyncCheckpointIO(oss_checkpoint_io)],
    )

    ckpt_path = f"checkpoints/{md5}.ckpt"
    if bucket.object_exists(ckpt_path):
        if train:
            bucket.delete_object(ckpt_path)
            trainer.fit(model, dm)
            print("Train from scratch: ", md5)
        else:
            bucket.get_object_to_file(ckpt_path, ckpt_path)
            model = model.__class__.load_from_checkpoint(ckpt_path)
            print("Load from checkpoint: ", md5)
    else:
        # if model.adv_mode:
        #     normal_cfg = copy.deepcopy(cfg)
        #     normal_cfg.pop("adv_mode")
        #     normal_md5 = get_md5(normal_cfg)
        #     normal_ckpt_path = f"checkpoints/{normal_md5}.ckpt"
        #     if bucket.object_exists(normal_ckpt_path):
        #         bucket.get_object_to_file(normal_ckpt_path, normal_ckpt_path)
        #         model = model.__class__.load_from_checkpoint(normal_ckpt_path)
        #         model.adv_mode = True
        #         print("Load from normal checkpoint: ", normal_md5)
        trainer.fit(model, dm)
        train = True

    if not model.adv_mode:
        eps = [0, 0.001, 0.01, 0.1, 1.0]
    else:
        eps = list(range(5))
    accs, acc5s, acc10s, asrs, asr5s, asr10s = [], [], [], [], [], []
    for i in eps:
        if i > 0:
            model.eval_atk = PGD(model, eps=i / 255.0, alpha=1 / 255, steps=10)
            model.adv_mode = True
        else:
            model.adv_mode = False
        ret = trainer.test(model, datamodule=dm, ckpt_path="best" if train else None)[0]
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
    torch.set_float32_matmul_precision("high")
    cli = MyLightningCLI(save_config_callback=None, run=False)
    model = cli.model
    dm = cli.datamodule
    args = get_args(cli.config.as_dict())
    args["model"] = model.__class__.__name__
    args["dataset"] = dm.__class__.__name__
    wandb.init(project="RobustCBM", config=args, tags=[model.__class__.__name__, dm.__class__.__name__])
    exp(model, dm, args, cli.config["train"])
