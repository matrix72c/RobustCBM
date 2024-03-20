import torch
import torch.nn as nn

from ..attack import Attack


class PGD_V2V(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 5e-2)
        alpha (float): step size. (Default: 1e-2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, L)` where `N = number of batches`, `L = number of concepts`.
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, L)`.

    Examples::
        >>> attack = torchattacks.PGD_V2V(model, eps=5e-2, alpha=1e-2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=5e-2, alpha=1e-2, steps=10, random_start=True):
        super().__init__("PGD_V2V", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, concepts, labels):
        r"""
        Overridden.
        """

        concepts = concepts.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(concepts, labels)

        loss = nn.CrossEntropyLoss()
        adv_concepts = concepts.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_concepts = adv_concepts + torch.empty_like(adv_concepts).uniform_(
                -self.eps, self.eps
            )
            # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_concepts.requires_grad = True
            outputs = self.get_logits(adv_concepts)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_concepts, retain_graph=False, create_graph=False
            )[0]

            adv_concepts = adv_concepts.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_concepts - concepts, min=-self.eps, max=self.eps)
            adv_concepts = (concepts + delta).detach()
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_concepts
