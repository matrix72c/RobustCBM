import torch
import torch.nn.functional as F
from attacks import Attack

class PGD(Attack):
    def __init__(
        self,
        eps: float = 8.0 / 255,
        alpha: float = 2.0 / 255,
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
    def attack(self, model, x, y):
        x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(
            -self.eps, self.eps
        )
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        for _ in range(self.steps):
            x_adv.requires_grad = True
            o = model(x_adv)
            loss = self.loss_fn(o, y)
            loss.backward()
            x_adv = x_adv.detach() + self.alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max).detach()

        return x_adv
