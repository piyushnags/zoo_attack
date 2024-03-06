import os
import torch



def float_to_uint8(x: torch.Tensor) -> torch.Tensor:
    min_x = x.min()
    max_x = x.max()
    x = (((x - min_x)/(max_x - min_x))*255).round()
    x = x.to(torch.uint8)
    return x


def threshold(x: torch.Tensor, t: float):
    return torch.where(x < t, 0., 1.)


def compute_loss(x, t_0):
    '''
    hinge-like loss function to evaluate strength
    of adversarial example
    '''
    B, _ = x.size()
    val, ind = torch.topk(x, 2, dim=1)
    t = t_0.item()
    f = torch.zeros((B,))
    for b in range(B):
        if ind[b][0] == t:
            f[b] = val[b][0] - val[b][1]
        else:
            f[b] = torch.tensor(0)
    
    return f


def untargeted_obj(x, t_0):
    '''
    :param x: logits
    :param t: original class
    :return: return 
    '''
    val, ind = torch.topk(x, 2, dim=1)
    val, ind = val.squeeze(), ind.squeeze()
    t = t_0.item()
    if ind[0] == t:
        return torch.tensor([val[0] - val[1]])
    else:
        return torch.tensor([x[0][t] - val[0]])


def create_exp_dir():
    exp_no = 0
    exp_path = None
    if not os.path.exists('results/'):
        os.makedirs('results/')
    while True:
        if os.path.exists(f'results/exp_{exp_no}'):
            exp_no += 1
        else:
            exp_path = f'results/exp_{exp_no}'
            os.makedirs(exp_path)   
            break

    return exp_no, exp_path   