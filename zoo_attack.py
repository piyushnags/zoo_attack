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
    # print(val)
    t = t_0.item()
    if ind[0] == t:
        return torch.tensor([val[0] - val[1]])
    else:
        return torch.tensor([x[0][t] - val[0]])

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
        step += 1
    
    print(f'{step} number of steps taken to generate adversarial image')
    return image


def zoo_attack(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    
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
    adv_image = zoo_attack_naive(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        plt.imsave( os.path.join(exp_path, f'adv_img_{success+1}.png'), adv_image.squeeze().cpu() )
        plt.imsave( os.path.join(exp_path, f'clean_img_{success+1}.png'), images.squeeze().cpu() )
        print(labels)
        success += 1

    if total >= num_image:
        break

print('success rate : %.4f'%(success/total))



