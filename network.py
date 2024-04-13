import os
import kornia
import torch
import torch.nn as nn
from PIL import Image
from losses import StyleLoss, SparseLoss
from models import Pure
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset


class PureSeal:
    def __init__(self, n, image_size, secret_size, device, device_ids, lr):
        super(PureSeal, self).__init__()
        
        # basic
        self.n = n
        self.image_h, self.image_w = image_size
        self.secret_h, self.secret_w = secret_size

        # device
        self.device = device
        self.device_ids = device_ids

        # network
        self.pure = Pure((image_size[0], image_size[1]), (1, 48), (3, 3), (secret_size[0], secret_size[1]), n, (3, n), (n, 3), (48, 1)).to(device)
        
        # self.pure = nn.DataParallel(self.pure, device_ids=device_ids).cuda()

        # optimizer
        print("learning rate: {}".format(lr))
        
        self.opt_pure = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.pure.parameters()), lr=lr,
            betas=(0.9, 0.999), eps=0.1)
        
        # loss function
        self.L1_loss = nn.L1Loss().to(device)

        # weight of losses
        self.style_loss_weight = 1
        self.image_loss_weight = 5
        self.secret_loss_weight = 5
        self.sparse_loss_weight = 0.01

    def train(self, image, secret):
        self.pure.train()

        with torch.enable_grad():
            # use device to compute
            image, secret = image.to(self.device), secret.to(self.device)
            # debug: <yes>
            encoded_secret, code_b1, code_b2, encoded_image, hidden_b1, hidden_b2, info_n, extracted_code, decoded_secret = self.pure(image, secret)
            
            # compute losses
            loss_about_style = StyleLoss(code_b1, code_b2, hidden_b1, hidden_b2)
            loss_about_image = self.L1_loss(image, encoded_image)
            loss_about_secret = self.L1_loss(secret, decoded_secret)
            loss_about_sparse = SparseLoss(info_n, self.pure.invariance.dense_n.weight)
            
            loss_total = self.secret_loss_weight * loss_about_secret + self.sparse_loss_weight * loss_about_sparse +\
                         self.style_loss_weight * loss_about_style + self.image_loss_weight * loss_about_image
            
            # empty old gradient
            self.opt_pure.zero_grad()
            
            # backward to compute gradients
            loss_total.backward()
            
            # update the network parameters once
            self.opt_pure.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_image.detach(), image, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_image.detach(), image, window_size=5, reduction="mean")
            
        result = {
            "psnr": psnr,
            "ssim": ssim,
            "loss_about_image": loss_about_image,
            "loss_about_secret": loss_about_secret,
            "loss_about_style": loss_about_style,
            "loss_about_sparse": loss_about_sparse,
            "loss_total": loss_total
        }
        return result

    def validation(self, image, secret):
        self.pure.eval()

        with torch.no_grad():
            # use device to compute
            image, secret = image.to(self.device), secret.to(self.device)
            encoded_secret, code_b1, code_b2, encoded_image, hidden_b1, hidden_b2, info_n, extracted_code, decoded_secret = self.pure(image, secret)

            # compute loss components
            loss_about_style = StyleLoss(code_b1, code_b2, hidden_b1, hidden_b2)
            loss_about_image = self.L1_loss(image, encoded_image)
            loss_about_secret = self.L1_loss(secret, decoded_secret)
            loss_about_sparse = SparseLoss(info_n, self.pure.invariance.dense_n.weight)

            loss_total = self.secret_loss_weight * loss_about_secret + self.sparse_loss_weight * loss_about_sparse +\
                         self.style_loss_weight * loss_about_style + self.image_loss_weight * loss_about_image

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_image.detach(), image, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_image.detach(), image, window_size=5, reduction="mean")
            
        result = {
            "psnr": psnr,
            "ssim": ssim,
            "loss_about_image": loss_about_image,
            "loss_about_secret": loss_about_secret,
            "loss_about_style": loss_about_style,
            "loss_about_sparse": loss_about_sparse,
            "loss_total": loss_total
        }
        return (result, encoded_secret, encoded_image, extracted_code, decoded_secret)
        
    def decoded_secret_error_rate(self, secret, decoded_secret):
        length = secret.shape[1] * secret.shape[2] # 32x32

        secret = secret.gt(0.5)
        decoded_secret = decoded_secret.gt(0.5)
        error_rate = float(sum(secret != decoded_secret)) / length
        return error_rate

    def decoded_secret_error_rate_batch(self, secret, decoded_secret):
        error_rate = 0.0
        batch_size = secret.size(0)
        for i in range(batch_size):
            error_rate += self.decoded_secret_error_rate(secret[i], decoded_secrt[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_pure):
        torch.save(self.pure.state_dict(), path_pure)

    def load_model(self, path_pure):
        self.pure.load_state_dict(torch.load(path_pure)) # when multi-gpu, pure.module.load_state_dict, when single-gpu, pure.load_state_dict 


class PureDataset(Dataset):
    def __init__(self, image_path, secret_path, image_size=(128, 128), secret_size=(32, 32)):
        super(PureDataset, self).__init__()

        self.image_h, self.image_w = image_size
        
        self.image_path = image_path
        self.secret_path = secret_path
        self.image_list = os.listdir(image_path)
        self.secret_list = os.listdir(secret_path)
        
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(), # [0,255] --> [0,1], [h,w,c] --> [c,h,w]
            transforms.Normalize([0.445, 0.419, 0.380], [0.249, 0.240, 0.240]) # [0,1] --> [-1,1]
        ])
        self.secret_transform = transforms.Compose([
            transforms.Resize(secret_size),
            transforms.ToTensor(), # [0,255] --> [0,1], [h,w,c] --> [c,h,w]
            transforms.Normalize([0.484], [0.186]) # [0,1] --> [-1,1]
        ])

    def transform_image_and_secret(self, image, secret):
        # Resize, ToTensor and Normalize
        image = self.image_transform(image)
        secret = self.secret_transform(secret)
        # debug: <yes>
        return (image, secret)

    def __getitem__(self, index):
        while True:
            image = Image.open(os.path.join(self.image_path, self.image_list[index])).convert("RGB")
            secret = Image.open(os.path.join(self.secret_path, self.secret_list[index])).convert("L")
            image, secret = self.transform_image_and_secret(image, secret)
            if image is not None and secret is not None:
                return (image, secret)
            index += 1

    def __len__(self):
        return min(len(self.image_list), len(self.secret_list))



    
