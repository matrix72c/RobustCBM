import torch
import torch.nn.functional as F
from attacks import attack

class pgd(attack):
    def __init__(
        self,
        eps: float = 8.0 / 255,
        alpha: float = 2.0 / 255,
        steps: int = 10,
        loss_fn: callable = F.cross_entropy,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
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
            grad = torch.autograd.grad(loss.sum(), [x_adv])[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = torch.min(
                torch.max(x_adv, x - self.eps), x + self.eps
            ).clamp(self.clip_min, self.clip_max)

        return x_adv
