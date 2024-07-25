from lightning.pytorch.callbacks import Callback


class AdversarialEarlyStop(Callback):
    def __init__(self, monitor="val_acc", mode="max", patience=3, min_delta=0.001):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)

        if current is None:
            return

        if pl_module.adv_training:
            if self.should_stop(current):
                trainer.should_stop = True
        else:
            if self.should_stop(current):
                pl_module.adv_training = True
                pl_module.trainer.optimizers, pl_module.trainer.schedulers = (
                    pl_module.configure_optimizers()
                )
                pl_module.acc.reset()
                self.best_score = None
                self.wait = 0

    def should_stop(self, current):
        if self.best_score is None:
            self.best_score = current
            self.wait = 0
            return False

        if self.mode == "min":
            improvement = current < self.best_score - self.min_delta
        else:  # mode == "max"
            improvement = current > self.best_score + self.min_delta

        if improvement:
            self.best_score = current
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience
