import torch
import random
from CNN.resnet import ResNet18
from load_data import load_data
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
    N, _ = x.size()
    val, ind = torch.topk(x, 2, dim=1)
    f = torch.empty((N,))
    for i in range(N):
        t = t_0[i]
        if ind[i][0] == t:
            f[i] = val[i][0] - val[i][1]
        else:
            f[i] = val[i][t] - val[i][0]
    return f

# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''

def zoo_attack(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    # N batches of images having C channels with dimensions H x W
    N, C, H, W = image.size()

    # Choose a random pixel for each image in the batch
    ind_h = torch.randint(0, H, (N,))
    ind_w = torch.randint(0, W, (N,))
    
    # Track the optimal solution of delta
    # For each pixel intensity, compute the value of obj. function
    # when that level of intensity is added as perturbation to the input
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
        image[n,0][x][y] += torch.tensor(i/255, device=device)
    
    return image

# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])



total = 0
success = 0
num_image = 10 # number of images to be attacked

for i, (images, labels) in enumerate(testloader):
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    #adv_image = zoo_attack(network=model, image=images, target=target_label)
    adv_image = zoo_attack(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        success += 1

    if total >= num_image:
        break

print('success rate : %.4f'%(success/total))



