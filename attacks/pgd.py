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
    def attack(self, model, x, y):
        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = torch.clamp(delta, -self.eps, self.eps)
        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max).detach()

        for _ in range(self.steps):
            x_adv.requires_grad = True
            o = model(x_adv)
            loss = self.loss_fn(o, y)
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad.detach()
                x_adv = x_adv.detach() + self.alpha * grad.sign()
                delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
                x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)

        return x_adv