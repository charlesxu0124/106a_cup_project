import numpy as np
import matplotlib.pyplot as plt
import time
import glob

import sys

import skvideo.io

videodata = []

# for filename in glob.glob("/media/ashvin/data1/s3doodad/demos/icra2021/outputs_dataset_v4/obj_tilt*.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_tomatoblue1.npy"):
# for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj_dice_grasp1.npy"):

filenames = glob.glob("/media/ashvin/data1/s3doodad/demos/icra2021/outputs_dataset_v4/obj_test*.npy")
filenames.sort()
for filename in filenames:

    print(filename)
    x = np.load(filename, allow_pickle=True)

    for traj_i in range(1):
        traj = x[traj_i]["observations"]
        print(traj_i, len(traj))
        for t in range(len(traj)):
            # print("frame", t)
            if not traj[t]:
                print(traj_i, t)
                continue

            img = traj[t]["image_observation"][:, :, ::-1]
            videodata.append(img)

new_videodata = np.array(videodata)

skvideo.io.vwrite("/home/ashvin/scoop.mp4", new_videodata)
