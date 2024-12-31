import torch

def mtl(losses, pl, mode):
    if mode == "equal":
        gradient_normalize(losses, pl)
    elif mode == "ordered":
        gradient_ordered(losses, pl)
    pl.optimizers().step()
def get_grad(loss, model):
    model.zero_grad()
    loss.backward(retain_graph=True)
    grad = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad[name] = param.grad.detach().clone()
        else:
            grad[name] = torch.zeros_like(param)
    model.zero_grad()
    return grad

def calc_grad_norm(grad_dict):
    grads = []
    for _, param in grad_dict.items():
        grads.append(param.view(-1))
    if len(grads) == 0:
        return torch.tensor(0.0)
    grads = torch.cat(grads)
    norm = torch.norm(grads, p=2)
    if norm < 1e-10:
        norm = 1e-10
    return norm
        
def gradient_normalize(losses, model):
    grads = []
    grads_norm = []
    for loss in losses:
        grad = get_grad(loss, model)
        grads.append(grad)
        grads_norm.append(calc_grad_norm(grad))
    for name, param in model.named_parameters():
        param.grad = sum([grad[name] / grad_norm for grad, grad_norm in zip(grads, grads_norm)])

def gradient_ordered(losses, model):
    grads = []
    for loss in losses:
        grad = get_grad(loss, model)
        grads.append(grad)
    g0, g1 = grads
    g1_norm = calc_grad_norm(g1)
    # \gamma = \frac{\text{relu}(-\langle \boldsymbol{g}_0,\boldsymbol{g}_1\rangle)}{\Vert\boldsymbol{g}_1\Vert^2}
    for name, param in model.named_parameters():
        gamma = torch.relu(-torch.sum(g0[name] * g1[name]) / (g1_norm**2))
        param.grad = g0[name] + gamma * g1[name]

