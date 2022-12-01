import json
from rlkit.torch.sets import models
import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from torch.distributions import kl_divergence
import rlkit.pythonplusplus as ppp
from torchvision.utils import save_image

ptu.set_gpu_mode(True)
base_path = ''
create_vae_kwargs = json.load(open(base_path + 'variant.json', 'r'))['create_vae_kwargs']
vae = models.create_image_set_vae((3, 48, 48), **create_vae_kwargs)
# state_dict = torch.load(base_path + 'vae_only.pt')
# vae.load_state_dict(state_dict)
vae = torch.load('raw_vae.pt')
vae.eval()
vae.to(ptu.device)
train_sets = [ptu.from_numpy(t) for t in np.load(base_path + 'train_sets.npy')]
eval_sets = [ptu.from_numpy(t) for t in np.load(base_path + 'eval_sets.npy')]
set_images = train_sets[0]  # 0 = closed door, 1 = open door
prior_c = vae.encoder_c(set_images)
c = prior_c.mean
prior = vae.prior_z_given_c(c)
print(prior_c.mean)
print(prior.mean)

# import ipdb; ipdb.set_trace()

set_images = train_sets[0]  # 0 = closed door, 1 = open door
prior_c = vae.encoder_c(set_images)
c = prior_c.mean
prior = vae.prior_z_given_c(c)
reward_fn = lambda q_z: -kl_divergence(q_z, prior)
x = eval_sets[0]
save_image(x[0, :, :, :], "load_test.png")
q_z = vae.q_zs_given_independent_xs(x)
reward = ptu.get_numpy(reward_fn(q_z))
print(reward)
print(sorted(reward))
