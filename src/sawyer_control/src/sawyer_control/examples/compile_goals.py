import numpy as np
import matplotlib.pyplot as plt
import time
import glob

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp/run0/id0/itr_1500.pt"

for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj*.npy"):
    print(filename)
    x = np.load(filename, allow_pickle=True)

    goals = []

    for traj_i in range(len(x)):
        traj = x[traj_i]["observations"]
        print(traj_i, len(traj))
        for t in range(len(traj)):
            # print("frame", t)
            if not traj[t]:
                print(traj_i, t)
                continue

            img = traj[t]["image_observation"]
