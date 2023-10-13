import sys
from cmath import isnan
import torch
import torch.nn.functional as F
import torch.nn as nn


def l1(output, target):
    torch.set_printoptions(profile="full")
    #velocity roll has constant MIDI velocity through out note frame, not only onset point. 

    # loss = nn.L1Loss(reduction='mean')
    # velocity_loss = loss(output["velocity_output"], target["velocity_roll"]/128)

    small_noise = 1e-8 * torch.rand(output.shape, device = "cuda")
    #gt_onset_vel = torch.mul(target["velocity_roll"], target["onset_roll"]) #ignoring offset and looking at velocity on onset only. 
    l1error = torch.abs(output-target/128)
    if torch.count_nonzero(l1error) == 0: 
        return small_noise
    else:
        return torch.mean(l1error)

#Modified version of SoTA by bytedance 
def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    #TODO: check if masking to remove offset velocity is not needed.  
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    original_bce_loss = torch.sum(matrix * mask) / torch.sum(mask)
    #bce_wo_division = torch.mean(matrix * mask)

    return original_bce_loss

def bce_l1(output, target):
    frame_roll = target.clone()
    frame_roll[frame_roll != 0] = 1

    bce_loss = bce(output, target/128, frame_roll)
    l1_loss = l1(output, target)
    # print("bce_loss",bce_loss)
    
    if torch.isnan(bce_loss):
        bcel1_loss = l1_loss
    else:
        theta = 0.5 # changable
        bcel1_loss = theta*bce_loss + (1-theta)*l1_loss

    return bcel1_loss




