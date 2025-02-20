import torch
from attacks import Attack, Apgd, Apgdt, Fab, Square


class AutoAttack(Attack):
    def __init__(self, **kwargs):
        self.attacks = [
            Apgd(**kwargs),
            Apgdt(**kwargs),
            Fab(**kwargs),
            Square(**kwargs),
        ]

    def get_logits(self, x):
        return self.model(x)

    def attack(self, model, images, labels):
        self.device = images.device
        self.model = model
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]
        for attack in self.attacks:
            adv_images = attack(model, images[fails], labels[fails])

            outputs = self.get_logits(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = pre == labels[fails]
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(
                torch.arange(fails.shape[0]).to(self.device), wrongs
            )

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)
            multi_atk_records.append(len(fails))

            if len(fails) == 0:
                break

        return final_images
