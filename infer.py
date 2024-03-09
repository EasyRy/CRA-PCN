import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import open3d as o3d
from models.crapcn import CRAPCN

if __name__ == "__main__":

    # load pre-trained model
    model = CRAPCN()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('./pretrain/pcn/ckpt-best.pth')
    model.load_state_dict(checkpoint['model'])

    # inference
    pc = o3d.io.read_point_cloud('example_pc/air.ply')
    pc = torch.Tensor(np.asarray(pc.points)).cuda() # (2048, 3)
    with torch.no_grad():
        re = model(pc.unsqueeze(0)) 
        result = re[-1].squeeze(0).detach().cpu().numpy() # (16384, 3)
    
    # save result
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result)
    o3d.io.write_point_cloud('example_pc/air_complete.ply', pcd)