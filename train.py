import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from network import PureSeal, PureDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def save_images(saved_all, epoch, folder, resize_to=None):
    # saved_all is a list of elements [n,c,h,w]
    image = saved_all[0] # image value range: [-1, 1]
    secret = saved_all[1] # image value range: [-1, 1]
    encoded_secret = saved_all[2] # image value range: [0, +inf]
    encoded_image = saved_all[3] # image value range:[0, 1]
    extracted_code = saved_all[4] # image value range: [0, +inf]
    decoded_secret = saved_all[5] # image value range: [0, 1]
    print('zero:', secret.shape, decoded_secret.shape)
    # debug: <yes>

    image = image[:image.shape[0], :, :, :].cpu()
    secret = secret[:secret.shape[0], :, :, :].cpu()
    encoded_secret = encoded_secret[:encoded_secret.shape[0], :, :, :].cpu()
    encoded_image = encoded_image[:encoded_image.shape[0], :, :, :].cpu()
    extracted_code = extracted_code[:extracted_code.shape[0], :, :, :].cpu()
    decoded_secret = decoded_secret[:decoded_secret.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    image = (image + 1) / 2
    secret = (secret + 1) / 2
    # scale values to range [0, 1] from original range of [0, +inf]
    encoded_secret = encoded_secret / encoded_secret.max()
    extracted_code = extracted_code / extracted_code.max()
    
    print('one:', secret.shape, decoded_secret.shape)

    if resize_to is not None:
        image = F.interpolate(image, size=resize_to)
        secret = F.interpolate(secret, size=resize_to)
        encoded_secret = F.interpolate(encoded_secret, size=resize_to)
        encoded_image = F.interpolate(encoded_image, size=resize_to)
        extracted_code = F.interpolate(extracted_code, size=resize_to)
        decoded_secret = F.interpolate(decoded_secret, size=resize_to)
        
    print('two:', secret.shape, decoded_secret.shape)
        
    # transform L mode to RGB mode
    secret_r = secret[:, 0:1, :, :]
    secret_g = secret[:, 0:1, :, :]
    secret_b = secret[:, 0:1, :, :]
    secret = torch.cat((secret_r, secret_g, secret_b), 1)
    
    decoded_secret_r = decoded_secret[:, 0:1, :, :]
    decoded_secret_g = decoded_secret[:, 0:1, :, :]
    decoded_secret_b = decoded_secret[:, 0:1, :, :]
    decoded_secret = torch.cat((decoded_secret_r, decoded_secret_g, decoded_secret_b), 1)   
    
    print('three:', secret.shape, decoded_secret.shape)

    # debug: <no!!!>
    print(image.unsqueeze(0).shape, secret.unsqueeze(0).shape, encoded_image.unsqueeze(0).shape, encoded_secret.unsqueeze(0).shape,\
                                extracted_code.unsqueeze(0).shape, decoded_secret.unsqueeze(0).shape)
    stacked_images = torch.cat([image.unsqueeze(0), secret.unsqueeze(0), encoded_image.unsqueeze(0), encoded_secret.unsqueeze(0),\
                                extracted_code.unsqueeze(0), decoded_secret.unsqueeze(0)], dim=0)
    shape = stacked_images.shape # [6,n,c,h,w]
    stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
    stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

    saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
    saved_image.save(filename)

def get_random_images(image, secret, encoded_secret, encoded_image, extracted_code, decoded_secret):
    selected_id = np.random.randint(1, image.shape[0]) if image.shape[0] > 1 else 1 # choose one int number in range [1,batch]
    resize_to = (image.shape[2], image.shape[3])

    # choose img depends on selected id: [b,c,h,w] --> [c,h,w]
    image = image.cpu()[selected_id - 1: selected_id, :, :, :] # choose number range in [1,batch], index range in [0,batch-1]
    secret = secret.cpu()[selected_id - 1: selected_id, :, :, :]
    secret = F.interpolate(secret, resize_to)
    
    encoded_secret = encoded_secret.cpu()[selected_id - 1: selected_id, :, :, :]
    encoded_image = encoded_image.cpu()[selected_id - 1: selected_id, :, :, :]
    
    extracted_code = extracted_code.cpu()[selected_id - 1: selected_id, :, :, :]
    decoded_secret = decoded_secret.cpu()[selected_id - 1: selected_id, :, :, :]
    decoded_secret = F.interpolate(decoded_secret, resize_to)
    
    # debug: <yes>
    return [image, secret, encoded_secret, encoded_image, extracted_code, decoded_secret]

def concatenate_images(saved_all, image, secret, encoded_secret, encoded_image, extracted_code, decoded_secret):
    # saved_all is a list with i groups of images, saved is a list with 1 group of images
    saved = get_random_images(image, secret, encoded_secret, encoded_image, extracted_code, decoded_secret)

    # secret size should be modified to image size for concatenating images 
    resize_to = (image.shape[2], image.shape[3]) # [128,128]
    saved[1] = F.interpolate(saved[1], size=resize_to) # secret: [32,32] --> [128,128]
    saved[5] = F.interpolate(saved[5], size=resize_to) # decoded secret: [32,32] --> [128,128]
    
    for i in range(6):
        saved_all[i] = torch.cat((saved_all[i], saved[i]), 0) # [i,c,h,w] --> [i+1,c,h,w]
    # debug: <yes>
    return saved_all # a list of elements whose shape is [i+1,c,h,w]


# basic setting
n = 150
image_size = (128, 128)
secret_size = (32, 32)

image_dataset_path = "./data/images/"
secret_dataset_path = "./data/secrets/"
result_folder = "./results/"

lr = 5e-4
batch_size = 8
epoch_number = 6
save_images_number = 5

train_continue = True
train_continue_epoch = 13
train_continue_path = ""

# cudnn
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# detect where the inplace operation is
torch.autograd.set_detect_anomaly(True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

# network
network = PureSeal(n, image_size, secret_size, device, device_ids, lr)

# prepare data
train_dataset = PureDataset(os.path.join(image_dataset_path, "train"),
                            os.path.join(secret_dataset_path, "train"),
                            image_size, secret_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)

valid_dataset = PureDataset(os.path.join(image_dataset_path, "valid"),
                            os.path.join(secret_dataset_path, "valid"),
                            image_size, secret_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)

# resume
if train_continue:
    M_path = "results/" + train_continue_path + "/models/M_" + str(train_continue_epoch) + ".pth"
    network.load_model(M_path)

# start training
print("\nStart training : \n")

for epoch in range(epoch_number):

    epoch += train_continue_epoch if train_continue else 0

    # ------------------- training ----------------------
    running_result = {
        "psnr": 0.0,
        "ssim": 0.0,
        "loss_about_image": 0.0,
        "loss_about_secret": 0.0,
        "loss_about_style": 0.0,
        "loss_about_sparse": 0.0,
        "loss_total": 0.0
    }
    start_time = time.time()

    num = 0
    for _, items in enumerate(train_dataloader):
        images, secrets = items
        image = images.to(device)
        secret = secrets.to(device)

        result = network.train(image, secret)
        for key in result:
            running_result[key] += float(result[key])
        num += 1

    # train results
    content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
    for key in running_result:
        content += key + "=" + str(running_result[key] / num) + ","
    content += "\n"

    with open(result_folder + "/train_log.txt", "a") as file:
        file.write(content)
    print(content)
    
    # ------------------- validation ----------------------
    valid_result = {
        "psnr": 0.0,
        "ssim": 0.0,
        "loss_about_image": 0.0,
        "loss_about_secret": 0.0,
        "loss_about_style": 0.0,
        "loss_about_sparse": 0.0,
        "loss_total": 0.0
    }
    start_time = time.time()

    saved_iterations = np.random.choice(np.arange(len(valid_dataloader)), size=save_images_number, replace=False)
    saved_all = None

    num = 0
    for i, items in enumerate(valid_dataloader):
        images, secrets = items
        image = images.to(device)
        secret = secrets.to(device)

        result, encoded_secret, encoded_image, extracted_code, decoded_secret = network.validation(image, secret)
        for key in result:
            valid_result[key] += float(result[key])
        num += 1

        if i in saved_iterations:
            print('here')
            if saved_all is None:
                saved_all = get_random_images(image, secret, encoded_secret, encoded_image, extracted_code, decoded_secret)
            else:
                saved_all = concatenate_images(saved_all, image, secret, encoded_secret, encoded_image, extracted_code, decoded_secret)
                
    save_images(saved_all, epoch, result_folder + "images/", resize_to=(image_size[0], image_size[1]))
    
    # validation results
    content = "Epoch" + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
    for key in valid_result:
        content += key + "=" + str(valid_result[key] / num) + ","
    content += "\n"

    with open(result_folder + "/valid_log.txt", "a") as file:
        file.write(content)
    print(content)

    # ------------------- save model ----------------------
    path_model = result_folder + "models/"
    path_pure = path_model + "M_" + str(epoch) + ".pth"
    network.save_model(path_pure)
    



            

    
