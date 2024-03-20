import torch
import torch.nn as nn

from ..attack import Attack


class PGD_AE(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack (concept->label).
        eps (float): maximum perturbation. (Default: 1e-2)
        alpha (float): step size. (Default: 1e-3)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - concepts: :math:`(N, L)` where `N = number of batches`, `L = vector length`.
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, L)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, model_type='concept', eps=1e-2,
                 alpha=1e-3, steps=10, lamba=10, random_start=True):
        super().__init__("PGD_AE", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.model_type = model_type
        self.lamba = lamba

    def forward(self, images, labels, explanations):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        explanations = explanations.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # untargeted_version
        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        loss_label = nn.CrossEntropyLoss()
        if args.model_type == 'concept':
            loss_explanation = nn.BCELoss()
        else:
            loss_explanation = nn.MSELoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            explanation_after, output_after = model(adv_image, flag=True)

            # Calculate loss
            # if self.targeted:
            #     cost = -loss_label(outputs, target_labels)
            # else:
            # constrain label loss decrease , concept loss increase
            cost = loss_label(output_after, labels) - self.lamba * loss_concept(explanation_after, explanations)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
