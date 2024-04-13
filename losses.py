import torch
import torch.nn as nn


def get_gram_matrix(mat):
    b = mat.shape[0]
    c = mat.shape[1]
    a = mat.view(b, c, -1) # [b, c, hxw]
    at = a.permute(0, 2, 1) # [b, hxw, c]
    gram = torch.bmm(a, at) # input: [p,m,n] bmm [p,n,a], output: [p,m,a]
    # debug: <yes>
    return gram

def StyleLoss(code_b1, code_b2, hidden_b1, hidden_b2):
    L1_loss = nn.L1Loss()
    part1 = L1_loss(get_gram_matrix(code_b1), get_gram_matrix(hidden_b1))
    part2 = L1_loss(get_gram_matrix(code_b2), get_gram_matrix(hidden_b2))
    style_loss = 0.5 * (part1 + part2)
    return style_loss

def SparseLoss(info_n, weight):
    dh = 1 - torch.mul(info_n, info_n) # [b, n, h, w]
    dh = dh.permute(0, 2, 3, 1) # [b, h, w, n]
    sparse_loss = torch.matmul(torch.mul(dh, dh), torch.mul(weight, weight)) # input: [b,h,w,n] matmul [n,3], output: [b,h,w,3]
    # debug: <yes>
    return torch.mean(sparse_loss)
