from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
import cv2
from sawyer_control.core.image_env import ImageEnv
from sawyer_control.helper.video_capture import VideoCapture

SUFFIX = "4"

resource = 0
cap = VideoCapture(resource)

# env = ImageEnv(SawyerReachXYZEnv())
# img = env.get_image(width=84, height=84)
img = cap.read()
cv2.imwrite("input%s.png" % SUFFIX, img)

def crop(img):
    img = resize(img[0:270, 90:570, ::-1], (48, 48), anti_aliasing=True) * 255
    img = img.astype(np.uint8)
    return img
    # return img.transpose([2, 1, 0]).flatten()

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import glob
import numpy as np
import torch
from rlkit.torch import pytorch_util as ptu
from torchvision.utils import save_image

ptu.set_gpu_mode(True)

imgs = crop(np.array(img)).transpose([2, 1, 0]).flatten().reshape(1, 6912) / 255

NAME = "final_drawer_close1"
# x = np.load("/home/ashvin/data/s3doodad/demos/icra2021/datasets/%s.npy" % NAME, allow_pickle=True)
# imgs = x.item()['initial_image_observation'] # ['image_desired_goal'] # (10, 6912)
imgs = ptu.from_numpy(imgs)

# vqvae = torch.load("/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp-pretrain1/run1/id0/best_vqvae.pt")
pretrained_vae_path="/home/ashvin/data/sasha/awac-exps/real-world/vqvae/run2/id0/best_vqvae.pt"
# pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp-augment1/run9/id0/best_vqvae.pt"
vqvae = torch.load(pretrained_vae_path)
pixel_cnn = vqvae.pixel_cnn

# cond = ptu.get_numpy(vqvae.encode(imgs, cont=False))
cond = vqvae.encode(imgs, ) # cont=False) # .long()

root_len = vqvae.root_len

# s = vqvae.sample_prior(1, cond=cond)

goals = []
x0 = []
decoded = []

REPETITIONS = 20

print("bz", REPETITIONS * len(imgs))

for _ in range(5):
    e_indices = pixel_cnn.generate(shape=(root_len, root_len), batch_size=REPETITIONS * len(imgs), cond=cond.repeat(REPETITIONS, 1)).reshape(-1, root_len**2)

    sampled_latents = vqvae.discrete_to_cont(e_indices)
    decoded_imgs = vqvae.decode(sampled_latents)
    decoded_x0_imgs = vqvae.decode(cond.repeat(REPETITIONS, 1))

    input_channels = 3
    imsize = 48
    save_image(
        decoded_imgs.data.view(-1, input_channels, imsize, imsize).transpose(2, 3),
        "samples%s.png" % SUFFIX
    )
    save_image(
        imgs.data.view(-1, input_channels, imsize, imsize).transpose(2, 3),
        "x_0%s.png" % SUFFIX
    )
    save_image(
        decoded_x0_imgs.data.view(-1, input_channels, imsize, imsize).transpose(2, 3),
        "x_0_reconstructed%s.png" % SUFFIX
    )

    numpy_latents = ptu.get_numpy(sampled_latents).reshape(-1, 5*12*12)
    numpy_cond = ptu.get_numpy(decoded_x0_imgs).reshape(-1, 48*48*3)
    decoded_imgs = ptu.get_numpy(decoded_imgs).reshape(-1, 48*48*3)
    goals.append(numpy_latents)
    x0.append(numpy_cond)
    decoded.append(decoded_imgs)

goals = np.concatenate(goals, axis=0)
x0 = np.concatenate(x0, axis=0)
decoded = np.concatenate(decoded, axis=0)

data = {}
data["initial_image_observation"] = x0
data["image_desired_goal"] = decoded
data["latent_desired_goal"] = goals

np.save("/home/ashvin/data/s3doodad/demos/icra2021/goals_v3/%s_pixelcnn.npy" % NAME, data)