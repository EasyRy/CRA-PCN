import torch
from Chamfer3D import dist_chamfer_3D

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.state_dict()}, path)


def generator_step(net_d, out2, net_loss, optimizer):
    set_requires_grad(net_d, False)
    d_fake = net_d(out2[:, 0:2048, :])
    errG_loss_batch = torch.mean((d_fake - 1) ** 2)
    total_gen_loss_batch = errG_loss_batch + net_loss * 200
    total_gen_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda(), retain_graph=True, )
    optimizer.step()
    return d_fake


def discriminator_step(net_d, gt, d_fake, optimizer_d):
    set_requires_grad(net_d, True)
    d_real = net_d(gt[:, 0:2048, :])
    d_loss_fake = torch.mean(d_fake ** 2)
    d_loss_real = torch.mean((d_real - 1) ** 2)
    errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
    total_dis_loss_batch = errD_loss_batch
    total_dis_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda())
    optimizer_d.step()



def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

    
def calc_cd(output, gt, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t