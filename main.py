from lightning.pytorch.cli import LightningCLI
import pandas as pd
import torch
from torchattacks import PGD, AutoAttack, FGSM, CW


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default="run")
        parser.add_argument("--adv_patience", default=30)


def calc_acc(res):
    total_acc = 0
    total_concept_acc = 0
    for i in res:
        total_acc += i["test_acc"]
        total_concept_acc += i["test_concept_acc"]
    return total_acc / len(res), total_concept_acc / len(res)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli = MyLightningCLI(save_config_callback=None, run=False)

    static_early_stopping = cli.trainer.early_stopping_callback.state_dict()

    # Normal training
    cli.trainer.fit(cli.model, cli.datamodule)
    ret = cli.trainer.test(cli.model, cli.datamodule)
    std_acc, std_concept_acc = calc_acc(ret)

    # Reset the model and adversarial training
    static_early_stopping["patience"] = cli.config.adv_patience
    cli.trainer.early_stopping_callback.load_state_dict(static_early_stopping)
    static_checkpoint = cli.trainer.checkpoint_callback.state_dict()
    static_checkpoint["best_model_score"] = torch.tensor(-torch.inf)
    static_checkpoint["kth_value"] = torch.tensor(-torch.inf)
    cli.trainer.checkpoint_callback.load_state_dict(static_checkpoint)
    
    cli.model.adv_training = True
    cli.trainer.optimizers, cli.trainer.schedulers = cli.model.configure_optimizers()
    cli.trainer.should_stop = False
    cli.trainer.fit(cli.model, cli.datamodule)

    # Evaluation
    cli.model.adv_training = False
    ret = cli.trainer.test(cli.model, cli.datamodule)
    adv_acc, adv_concept_acc = calc_acc(ret)

    # Adversarial attacks
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cli.model.adv_training = True

    cli.model.test_atk = PGD(cli.model)
    cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    ret = cli.trainer.test(cli.model, cli.datamodule)
    pgd_acc, pgd_concept_acc = calc_acc(ret)

    # cli.model.test_atk = AutoAttack(cli.model, n_classes=cli.model.hparams.num_classes)
    # cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    # ret = cli.trainer.test(cli.model, cli.datamodule)
    # aa_acc, aa_concept_acc = calc_acc(ret)

    # cli.model.test_atk = FGSM(cli.model)
    # cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    # ret = cli.trainer.test(cli.model, cli.datamodule)
    # fgsm_acc, fgsm_concept_acc = calc_acc(ret)

    # cli.model.test_atk = CW(cli.model)
    # cli.model.test_atk.set_normalization_used(mean=mean, std=std)
    # ret = cli.trainer.test(cli.model, cli.datamodule)
    # cw_acc, cw_concept_acc = calc_acc(ret)

    df = pd.read_csv("result.csv")
    new_row = {
        "run_name": cli.config.run_name,
        "std_acc": std_acc,
        "std_concept_acc": std_concept_acc,
        "adv_acc": adv_acc,
        "adv_concept_acc": adv_concept_acc,
        "pgd_acc": pgd_acc,
        "pgd_concept_acc": pgd_concept_acc,
        # "aa_acc": aa_acc,
        # "aa_concept_acc": aa_concept_acc,
        # "fgsm_acc": fgsm_acc,
        # "fgsm_concept_acc": fgsm_concept_acc,
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
