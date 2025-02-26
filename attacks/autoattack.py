import torch
from attacks import Attack, Apgd, Apgdt, Fab, Square


class AutoAttack(Attack):
    def __init__(self, eps: float = 0.3, **kwargs):
        self.attacks = [
            Apgd(eps=eps, **kwargs),
            Apgdt(eps=eps, **kwargs),
            Fab(eps=eps, **kwargs),
            Square(eps=eps, **kwargs),
        ]

    def get_logits(self, x):
        return self.model(x)

    def attack(self, model, images, labels):
        self.device = images.device
        self.model = model
        batch_size = images.shape[0]
        remains = torch.arange(batch_size).to(self.device)
        final_images = images
        labels = labels
        for attack in self.attacks:
            adv_images = attack(model, images[remains], labels[remains])

            outputs = self.get_logits(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = pre == labels[remains]
            wrongs = ~corrects

            succeeds = torch.masked_select(remains, wrongs)
            succeeds_of_remains = torch.masked_select(
                torch.arange(remains.shape[0]).to(self.device), wrongs
            )

            final_images[succeeds] = adv_images[succeeds_of_remains]

            remains = torch.masked_select(remains, corrects)

            if len(remains) == 0:
                break

        return final_images
