# PGD-Variants

## PGD-Variant 1 

Use PGD-V2V to attack concept->label  
  
**Arguments:**  
        model (nn.Module): model to attack.  
        eps (float): maximum perturbation. (Default: 5e-2)  
        alpha (float): step size. (Default: 1e-2)  
        steps (int): number of steps. (Default: 10)  
        random_start (bool): using random initialization of delta. (Default: True)  
  
**Shape:**  
        - images: :math:`(N, L)` where `N = number of batches`, `L = number of concepts`.  
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.  
        - output: :math:`(N, L)`.  
  
**Examples:**  
```py
from torchattacks import PGD_V2V  
attack = torchattacks.PGD_V2V(model, eps=5e-2, alpha=1e-2, steps=10, random_start=True)  
adv_images = attack(images, labels)
```
