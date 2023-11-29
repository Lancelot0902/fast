import torch


def gamma(self, x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def cov(x):
    cov_matrix = torch.matmul(x, x.t()) / x.shape[0]
    return cov_matrix

def style_loss(compute_style_loss):
    feat1 = compute_style_loss[0]
    feat2 = compute_style_loss[1]
    target = compute_style_loss[2]
    
    gama1 = cov(feat1)
    gama2 = cov(feat2)
    gama_tar = cov(target)

    loss = torch.nn.MSELoss()

    loss1 = loss(gama1, gama_tar)
    loss2 = loss(gama2, gama_tar)

    return loss1 + loss2

def content_loss(feat, target):
    loss = torch.nn.MSELoss()
    return loss(feat, target)
