import sys
from lightning.pytorch.cli import LightningCLI
import pandas as pd
import torch
from torchattacks import PGD, AutoAttack, FGSM, CW


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default="run")
        parser.add_argument("--ckpt", default=None)
        parser.add_argument("--std", action="store_true")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli = MyLightningCLI(save_config_callback=None, run=False)

    if cli.config.std:
        # Normal training
        cli.trainer.fit(cli.model, cli.datamodule)
        sys.exit()

    # Evaluation std model
    cli.model.adv_training = False
    ret = cli.trainer.test(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt)
    std_acc, std_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    cli.model.adv_training = True
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    cli.model.test_atk = PGD(cli.model)
    cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    ret = cli.trainer.test(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt)
    std_pgd_acc, std_pgd_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    cli.model.test_atk = FGSM(cli.model)
    cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    ret = cli.trainer.test(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt)
    std_fgsm_acc, std_fgsm_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    print(
        f"std_acc: {std_acc}, std_concept_acc: {std_concept_acc}, pgd_acc: {std_pgd_acc}, pgd_concept_acc: {std_pgd_concept_acc}, fgsm_acc: {std_fgsm_acc}, fgsm_concept_acc: {std_fgsm_concept_acc}"
    )

    # Adversarial training
    cli.model.adv_training = True
    cli.model.load_state_dict(torch.load(cli.config.ckpt)["state_dict"])
    cli.trainer.fit(cli.model, cli.datamodule)

    # Evaluate robust model
    cli.model.adv_training = False
    ret = cli.trainer.test(cli.model, cli.datamodule)
    adv_acc, adv_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    # Adversarial attacks
    cli.model.adv_training = True

    cli.model.test_atk = PGD(cli.model)
    cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    ret = cli.trainer.test(cli.model, cli.datamodule)
    adv_pgd_acc, adv_pgd_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    # cli.model.test_atk = AutoAttack(cli.model, n_classes=cli.model.hparams.num_classes)
    # cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    # ret = cli.trainer.test(cli.model, cli.datamodule)
    # aa_acc, aa_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    cli.model.test_atk = FGSM(cli.model)
    cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    ret = cli.trainer.test(cli.model, cli.datamodule)
    adv_fgsm_acc, adv_fgsm_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    # cli.model.test_atk = CW(cli.model)
    # cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    # ret = cli.trainer.test(cli.model, cli.datamodule)
    # cw_acc, cw_concept_acc = ret[0]["test_acc_epoch"], ret[0]["test_concept_acc_epoch"]

    df = pd.read_csv("result.csv")
    new_row = {
        "run_name": cli.config.run_name,
        "std_acc": std_acc,
        "std_concept_acc": std_concept_acc,
        "std_pgd_acc": std_pgd_acc,
        "std_pgd_concept_acc": std_pgd_concept_acc,
        "std_fgsm_acc": std_fgsm_acc,
        "std_fgsm_concept_acc": std_fgsm_concept_acc,
        "adv_acc": adv_acc,
        "adv_concept_acc": adv_concept_acc,
        "adv_pgd_acc": adv_pgd_acc,
        "adv_pgd_concept_acc": adv_pgd_concept_acc,
        # "aa_acc": aa_acc,
        # "aa_concept_acc": aa_concept_acc,
        "adv_fgsm_acc": adv_fgsm_acc,
        "adv_fgsm_concept_acc": adv_fgsm_concept_acc,
        # "cw_acc": cw_acc,
        # "cw_concept_acc": cw_concept_acc,
        "base": cli.model.hparams.base,
        "num_classes": cli.model.hparams.num_classes,
        "num_concepts": cli.model.hparams.num_concepts,
        "use_pretrained": cli.model.hparams.use_pretrained,
        "concept_weight": cli.model.hparams.concept_weight,
        "lr": cli.model.hparams.lr,
        "step_size": cli.model.hparams.step_size,
        "gamma": cli.model.hparams.gamma,
        "vib_lambda": (
            cli.model.hparams.vib_lambda
            if hasattr(cli.model.hparams, "vib_lambda")
            else 0
        ),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("result.csv", index=False)
