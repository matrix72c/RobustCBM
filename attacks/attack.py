import torch


class attack(object):
    r"""
    Base class for all attack classes.
    """

    def __init__(self):
        pass

    def __call__(self, model, x, y):
        r"""
        Call method for attack class.
        """
        params_requires_grad = {}
        for name, p in model.named_parameters():
            params_requires_grad[name] = p.requires_grad
            p.requires_grad = False
        training = False
        if model.training:
            model.eval()
            training = True
        x_adv = self.attack(model, x, y)
        if training:
            model.train()
        for name, p in model.named_parameters():
            p.requires_grad = params_requires_grad[name]
        return x_adv

    def attack(self, model, x, y):
        r"""
        Attack method to be overridden by all subclasses.
        """
        raise NotImplementedError
