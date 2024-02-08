import torch
import matplotlib.pyplot as plt
import random, os
from CNN.resnet import ResNet18
from load_data import load_data
from tqdm import tqdm
#import torch.backends.cudnn as cudnn



# load the mnist dataset (images are resized into 32 * 32)
training_set, test_set = load_data(data='mnist')

# define the model
model = ResNet18(dim=1)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# load the learned model parameters
model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))

model.to(device)
model.eval()


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


# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''

def zoo_attack_naive(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    # N = 1 batches of images having C channels with dimensions H x W
    N, C, H, W = image.size()

    # Initialize counter to track number of iterations
    # needed to converge
    step = 0
    while untargeted_obj(network(image), t_0) >= 0:

        # Choose a random pixel in the image
        ind_h = torch.randint(0, H, (N,))
        ind_w = torch.randint(0, W, (N,))
        
        # Track the optimal solution of delta
        # For each pixel intensity, compute the value of obj. function
        # when that level of intensity is added as perturbation to the image at chosen
        # pixel location
        delta_opt = None
        for d in range(255):
            image_ = image.clone()
            if d == 0:
                logits = network(image_)
                delta_opt = untargeted_obj(logits, t_0)
                continue
            
            d_ = torch.tensor(d/255, device=device)
            for n in range(N):
                x, y = ind_h[n], ind_w[n]
                image_[n,0][x][y] += d_

            logits = network(image_)
            f = untargeted_obj(logits, t_0)
            delta_opt  = torch.cat( (delta_opt, f), dim=-1 )
        
        adv_inds = torch.argmin(delta_opt.unsqueeze(0), dim=1)
        for n in range(N):
            i = adv_inds[n]
            x, y = ind_h[n], ind_w[n]
            i_ = torch.tensor(i/255, device=device).detach()
            image[n,0][x][y] += i_

        image = torch.clamp(image, 0, 1)
        step += 1
    
    print(f'{step} number of steps taken to generate adversarial image')
    return image


def zoo_attack_subset(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    # N = 1 batches of images having C channels with dimensions H x W
    N, C, H, W = image.size()

    # Initialize counter to track number of iterations
    # needed to converge
    step = 0
    while untargeted_obj(network(image), t_0) >= 0:

        # Choose a random pixel in the image
        ind_h = torch.randint(0, H, (N,))
        ind_w = torch.randint(0, W, (N,))
        
        # Track the optimal solution of delta
        # For each pixel intensity, compute the value of obj. function
        # when that level of intensity is added as perturbation to the image at chosen
        # pixel location
        delta_opt = None
        xi = image[0][0][ind_h[0]][ind_w[0]]
        upper = torch.tensor((xi + 15/255), device=device).detach() if ((xi+255)<=1) else torch.tensor(1, device=device).detach()
        lower = torch.tensor((xi - 15/255), device=device).detach() if ((xi+255)>=0) else torch.tensor(0, device=device).detach()
        d_ = torch.tensor(lower, device=device).detach()
        for d in range(31):
            if d_ > upper:
                break
            image_ = image.clone()
            for n in range(N):
                x, y = ind_h[n], ind_w[n]
                image_[n,0][x][y] += d_

            d_ += torch.tensor(1/255, device=device)
            logits = network(image_)
            f = untargeted_obj(logits, t_0)
            if d == 0:
                delta_opt = f
            else:
                delta_opt  = torch.cat( (delta_opt, f), dim=-1 )
        
        adv_inds = torch.argmin(delta_opt.unsqueeze(0), dim=1)
        for n in range(N):
            i = adv_inds[n]
            x, y = ind_h[n], ind_w[n]
            i_ = torch.tensor(i/255+lower, device=device).detach()
            image[n,0][x][y] += i_

        image = torch.clamp(image, 0, 1)
        step += 1
    
    print(f'{step} number of steps taken to generate adversarial image')
    return image


def zoo_attack_adam(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    image = image.detach()
    N, C, H, W = image.size()

    # Params taken from ZOO paper: https://arxiv.org/pdf/1708.03999.pdf
    B = 128
    b1, b2, eps = 0.9, 0.999, 1e-8
    M, v, T = torch.zeros((B, C, H, W)), torch.zeros((B, C, H, W)), torch.zeros((B, C, H, W))
    M_, v_ = torch.zeros_like(M), torch.zeros_like(v)
    h = 1e-4
    eta = 1e-2

    logits = network(image)
    # Create a batch to compute gradients of multiple pixels
    img = image.clone()
    img = img.repeat(B, 1, 1, 1)
    steps = 0
    while (untargeted_obj(logits, t_0) >= 0):
        # Choose random indices for each image in the batch
        ind_h = torch.randint(0, H, (B,))
        ind_w = torch.randint(0, W, (B,))

        # Initialize the standard basis vectors with h at chosen
        # indices
        ei = torch.zeros_like(img, device=device)
        for i in range(B):
            x, y = ind_h[i], ind_w[i]
            ei[i,0,x,y] += h
            T[i,0,x,y] += 1
        
        # Compute approximate gradients, dim of gi = (B,)
        f2, f1 = compute_loss(network(img+ei), t_0).detach(), compute_loss(network(img-ei), t_0).detach()
        gi = (f2-f1)/(2*h)

        # Update adam parameters and compute delta * for each image clone
        for i in range(B):
            x, y = ind_h[i], ind_w[i]
            M[i,0,x,y] = b1*M[i,0,x,y] + (1-b1)*gi[i]
            v[i,0,x,y] = b2*v[i,0,x,y] + (1-b2)*torch.square(gi[i])
            M_[i,0,x,y] = M[i,0,x,y]/(1-torch.pow(b1, T[i,0,x,y]))
            v_[i,0,x,y] = v[i,0,x,y]/(1-torch.pow(b2, T[i,0,x,y]))
            del_star = (-eta*M_[i,0,x,y])/(torch.sqrt(v_[i,0,x,y])+eps)
            image[0,0,x,y] += del_star
        
        image = torch.clamp(image, 0, 1)
        logits = network(image)
        steps += 1
    
    print(f'{steps} steps taken to generate adversarial image')
    return image

# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])


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

total = 0
success = 0
num_image = 10 # number of images to be attacked

for i, (images, labels) in tqdm(enumerate(testloader)):
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    #adv_image = zoo_attack(network=model, image=images, target=target_label)
    adv_image = zoo_attack_adam(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        plt.imsave( os.path.join(exp_path, f'adv_img_{success+1}.png'), adv_image.squeeze().cpu(), cmap='gray' )
        plt.imsave( os.path.join(exp_path, f'clean_img_{success+1}.png'), images.squeeze().cpu(), cmap='gray' )
        print(f'Original label: {labels}')
        print(f'Predicted label: {adv_pred}')
        success += 1

    if total >= num_image:
        break

print('success rate : %.4f'%(success/total))



