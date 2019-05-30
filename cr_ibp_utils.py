import torch
import torch.nn as nn
import torch.nn.functional as F

from cr_ibp_modules import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from copy import deepcopy
from time import time

def train(model, loader, val_loader, loss_func, optimizer,
          lr_schedule=None, h_schedule=None, mode='clean', cr_weight=4, epochs=30):
    assert mode in ['clean', 'ibp', 'cr', 'cr+ibp'], 'Mode not recognized'
    
    if mode.startswith('cr'):
        assert lr_schedule is not None and h_schedule is not None, 'Provide lr_schedule and h_schedule to use mode cr'
    
    if lr_schedule is None:
        use_update_rate = False
        lr_schedule = [1e-3] * epochs
    else:
        use_update_rate = True
        
    if h_schedule is None:
        h_schedule = [0] * epochs
        
    for epoch, lr, h in zip(range(epochs), lr_schedule, h_schedule):
        ts = time()
        if use_update_rate:
            update_rate(optimizer, lr)
        losses = []
        answers = []
        for idx, (batch, labels) in enumerate(loader):
            model.train()
            optimizer.zero_grad()
            logits = model(batch)
            if mode == 'clean':
                loss = loss_func(logits, labels)
            elif mode == 'ibp':
                bounds = compute_bounds(model, batch)
                z = bounds[-1]
                loss, _, _ = ibp_loss(loss_func, logits, z, labels)
            elif mode == 'cr':
                loss_orig = loss_func(logits, labels)
                loss = loss_orig + cr_weight * curvature_regularization_loss(model, loss_func, batch, h, labels)
            elif mode == 'cr+ibp':
                loss_orig = loss_func(logits, labels)
                bounds = compute_bounds(model, batch)
                z = bounds[-1]
                loss_ibp, _, _ = ibp_loss(loss_func, logits, z, labels)
                loss_cr = curvature_regularization_loss(model, loss_func, batch, h, labels)
                loss = loss_ibp + cr_weight * loss_cr
            losses.append(loss.item())
            preds = np.argmax(F.softmax(logits).detach().numpy(), axis=-1)
            answers.extend(preds.flatten() == labels.detach().numpy().flatten())
            loss.backward()
            optimizer.step()
        tf = time()
        print('[%2d]\ttrain loss\t%.7f\ttime\t%ds\ttrain acc\t%.2f' % (epoch+1, np.mean(losses), int(tf-ts), np.mean(answers)))
        if epoch % 2 == 0:
            losses = []
            answers = []
            for idx, (batch, labels) in enumerate(val_loader):
                model.eval()
                optimizer.zero_grad()
                logits = model(batch)
                preds = np.argmax(F.softmax(logits).detach().numpy(), axis=-1)
                loss = loss_func(logits, labels)
                losses.append(loss.item())
                answers.extend(preds.flatten() == labels.detach().numpy().flatten())
            print('[%2d]\tvalid loss\t%.7f\tvalid acc\t%.2f' % (epoch+1, np.mean(losses), np.mean(answers)))

def update_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def req_grad(model, state=True):
    """
    Detaches all parameters of the given model
    """
    for param in model.parameters():
        param.requires_grad_(state)

def plotter(x, x_adv, label, pred, pred_adv, method, noise=None):
    """
    Plots image, its adversarial perturbation, difference between then and added noise
    """ 
    
#     noise = x_adv - x
    img_noise = torchvision.transforms.functional.to_pil_image((x_adv - x).squeeze())
    x_adv = (x_adv - x_adv.min())/(x_adv.max() - x_adv.min())
    img_diff = torchvision.transforms.functional.to_pil_image((x_adv - x).squeeze())
    
    img = torchvision.transforms.functional.to_pil_image(x.squeeze())
    img_adv = torchvision.transforms.functional.to_pil_image(x_adv.squeeze())
    title_1 = "label: {} | pred: {}".format(label.item(), pred.item())
    title_2 = "label: {} | adversarial: {}".format(label.item(), pred_adv.item())
    
    plt.figure(figsize=(10,12))
    # original image
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(title_1)
    # adversarial 
    plt.subplot(1, 4, 2)
    plt.imshow(img_adv, cmap='gray')
    plt.title(title_2)
    # image diff
    plt.subplot(1, 4, 3)
    plt.imshow(img_diff, cmap='gray')
    plt.title('image diference')
    # noise
    plt.subplot(1, 4, 4)
    plt.imshow(img_noise, cmap='gray')
    plt.title('rorshah test')
        
    plt.show()
        
def test_on_adv(model, loader, loss, params, method='fgsm', verbose=True, isplot=False, ismean=True, n_images=10):
    
    if isplot and loader.batch_size > 1:
        print('Can visualize only batches of size 1')
        isplot = False

            
            
    model.eval()
    loss_hist = []
    acc_hist = []
    
    loss_adv_hist = []
    acc_adv_hist = []
    
    req_grad(model, state=False) # detach all model's parameters

    N = 0
    for x, label in loader:

        N += 1
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        
        # prediction for original input 
        logits = model(x_adv)
        preds = (logits.data.max(1)[1]).detach().cpu()
        
        loss_val = loss(logits, label)
        loss_val.backward()
        
        x_adv.data = x_adv.data + params['eps'] * torch.sign(x_adv.grad.data)
        
        # perturbations
        if method == 'fgsm':
            steps = 0
            noise = params['eps'] * torch.sign(x_adv.grad.data)
            
        elif method == 'pgd':
            steps = params['steps'] - 1
            x_adv.data = torch.max(x-params['eps'], torch.min(x_adv.data, x+params['eps'])) # clipping to (x-eps; x+eps)
            
        logits_adv = model(x_adv)
        loss_adv = loss(logits_adv, label)
        loss_adv.backward()
        
        for k in range(steps):
            x_adv.data = x_adv.data + params['alpha'] * torch.sign(x_adv.grad.data)
            x_adv.data = torch.max(x-params['eps'], torch.min(x_adv.data, x+params['eps'])) # clipping to (x-eps; x+eps)

            logits_adv = model(x_adv)
            loss_adv = loss(logits_adv, label)
            loss_adv.backward()
            
        # predictions for adversarials 
        preds_adv = (logits_adv.data.max(1)[1]).detach().cpu()

        # accuracy
        acc_val = np.mean((preds == label).numpy())
        acc_adv = np.mean((preds_adv == label).numpy())

        loss_hist.append(loss_val.cpu().item())
        acc_hist.append(acc_val)
        
        loss_adv_hist.append(loss_adv.cpu().item())
        acc_adv_hist.append(acc_adv)
        
        if verbose:
            print('Batch', N)
            print('true      | loss: {:.2f} | accuracy: {:.2f}'.format(loss_val, acc_val * 100))
            print('perturbed | loss: {:.2f} | accuracy: {:.2f}%'.format(loss_adv, acc_adv * 100))
            
        if isplot:
            plotter(x, x_adv, label, preds, preds_adv, method)
            if N >= n_images: break
                
    
    if ismean:
        print('Loss: true x: {:.2f} | adversarial: {:.2f}'.format(np.mean(loss_hist), np.mean(loss_adv_hist)) )
        print('Accuracy: true x: {:.2f} | adversarial: {:.2f}'.format(np.mean(acc_hist) * 100, np.mean(acc_adv_hist) * 100))

            
    return (loss_hist, loss_adv_hist), (acc_hist, acc_adv_hist)
