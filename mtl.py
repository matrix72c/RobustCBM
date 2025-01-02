import torch


def mtl(losses, pl, mode):
    if mode == "equal":
        mgda(losses, pl)
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


def grad2vec(grad_dict):
    vec = []
    for _, param in grad_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)


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
        param.grad = sum(
            [grad[name] / grad_norm for grad, grad_norm in zip(grads, grads_norm)]
        )


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


def mgda(losses, model):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
    as min |u|_2 st. u = \\sum c_i vecs[i] and \\sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
    """

    def min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        return c and norm
        """
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def min_norm_2d(grad_mat):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_i >= 0 for all i.
        Only correct in 2d.
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence.
        """
        dmin = 1e10
        for i in range(grad_mat.size()[0]):
            for j in range(i + 1, grad_mat.size()[0]):
                c, norm = min_norm_element_from2(
                    grad_mat[i, i], grad_mat[i, j], grad_mat[j, j]
                )
                if norm < dmin:
                    dmin = norm
                    sol = [(i, j), c, norm]
        return sol

    def projection2simplex(y):
        m = len(y)
        sorted_y = torch.sort(y, descending=True)[0]
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros(m).to(y.device))

    def next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        t = torch.ones(1).to(grad.device)
        if (tm1 > 1e-7).sum() > 0:
            t = torch.min(tm1[tm1 > 1e-7])
        if (tm2 > 1e-7).sum() > 0:
            t = torch.min(t, torch.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = projection2simplex(next_point)
        return next_point

    MAX_ITER = 250
    STOP_CRIT = 1e-5
    grads = []
    grad_vec = []
    grad_index = []
    for name, param in model.shared_params().items():
        grad_index.append(param.data.numel())
    grad_dim = sum(grad_index)
    for loss in losses:
        grad = get_grad(loss, model)
        grads.append(grad)

        vec = torch.zeros(grad_dim)
        count = 0
        for name, param in model.shared_params().items():
            beg = 0 if count == 0 else sum(grad_index[:count])
            end = sum(grad_index[: (count + 1)])
            vec[beg:end] = grad[name].view(-1)
            count += 1
        grad_vec.append(vec)

    grad_vec = torch.stack(grad_vec)
    grad_mat = torch.matmul(grad_vec, grad_vec.t())

    init_sol = min_norm_2d(grad_mat)
    n = grad_vec.size()[0]
    sol_vec = torch.zeros(n, device=grad_vec.device)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]
    if n > 2:
        for _ in range(MAX_ITER):
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = next_point(sol_vec, grad_dir, n)
            v1v1 = torch.sum(
                sol_vec.unsqueeze(1).repeat(1, n)
                * sol_vec.unsqueeze(0).repeat(n, 1)
                * grad_mat
            )
            v1v2 = torch.sum(
                sol_vec.unsqueeze(1).repeat(1, n)
                * new_point.unsqueeze(0).repeat(n, 1)
                * grad_mat
            )
            v2v2 = torch.sum(
                new_point.unsqueeze(1).repeat(1, n)
                * new_point.unsqueeze(0).repeat(n, 1)
                * grad_mat
            )

            nc, nd = min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < STOP_CRIT:
                break
            sol_vec = new_sol_vec
    for name, param in model.named_parameters():
        param.grad = sum(
            [grad[name] * sol_vec[i].item() for i, grad in enumerate(grads)]
        )
