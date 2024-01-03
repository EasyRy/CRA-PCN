import torch
import torch.nn as nn

from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import fps_subsample
chamfer_dist = chamfer_3DDist()


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    return d1







def get_loss_clamp(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    #-------------------pyramid-----------------------#
    Pc, P1, P2, P3 = pcds_pred
    
    gt_3 = gt
    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    

    cdc = CD(Pc, gt_c) 
    cd1 = CD(P1, gt_1) 
    cd2 = CD(P2, gt_2) 
    cd3 = CD(P3, gt_3)

    loss_decoder = cdc + cd1 + cd2 +cd3


    partial_matching = cdc #PM(partial, P1)
    

    loss_all = (loss_decoder) * 1e3 



    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt, gt_c, gt_c]

def get_loss_mvp(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    #-------------------pyramid-----------------------#
    Pc, P1, P2, P3 = pcds_pred

    gt_2048 = gt
    gt_512 = fps_subsample(gt_2048, 512)
    gt_256 = fps_subsample(gt_512, 256)
    
    

    cdc = CD(Pc, gt_256) 
    cd1 = CD(P1, gt_512) 
    cd2 = CD(P2, gt_512)
    cd3 = CD(P3, gt_2048)  
    
    loss_decoder = cdc + cd1 +cd2 +cd3


    partial_matching = PM(partial, P2)
    

    loss_all = (loss_decoder ) * 1e3 



    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2048, gt_512, gt_256]


def get_loss_clamp_xxxnet(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    
    Pc1, Pc2, Pc3, P1, P2, P3= pcds_pred
    

    
    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c3 = fps_subsample(gt_1, Pc3.shape[1])
    gt_c2 = fps_subsample(gt_c3, Pc2.shape[1])
    gt_c1 = fps_subsample(gt_c2, Pc1.shape[1])
    

    
    cdc1 = CD(Pc1, gt_c1)    
    cdc2 = CD(Pc2, gt_c2)    
    cdc3 = CD(Pc3, gt_c3)    
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)
    
    """
    cdc = CD(Pc, gt)    
    cd0 = CD(P0, gt) 
    cd1 = CD(P1, gt)
    cd2 = CD(P2, gt)
    cdout = CD(P_out, gt)
    """

    loss_decoder = cdc1 +cdc2+ cdc3 +  cd1 + cd2 + cd3



    partial_matching = PM(partial, P3)
        

    loss_all = (loss_decoder) * 1e3 

    losses = [cdc1, cd1, cd2, cd3, partial_matching]

    return loss_all, losses, [gt_2, gt_1, gt_1]

