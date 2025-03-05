import torch
import torch.nn.functional as F
from attacks import Attack


class PGD(Attack):
    def __init__(
        self,
        eps: float = 4.0 / 255,
        alpha: float = 4.0 / 2550,
        steps: int = 10,
        loss_fn: callable = F.cross_entropy,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        **kwargs
    ):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.enable_grad()
    @torch.inference_mode(False)
    def attack(self, model, x, y):
        for param in model.parameters():
            param.requires_grad = False

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta.requires_grad_(True)

        for _ in range(self.steps):
            x_adv = x + delta
            o = model(x_adv)
            loss = self.loss_fn(o, y)
            loss.backward()

            grad_sign = delta.grad.data.sign()
            delta.data += self.alpha * grad_sign
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = (
                torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x
            )

            delta.grad.data.zero_()

        model.zero_grad()
        return (x + delta).data
