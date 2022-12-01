from re import X
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torchvision.transforms.functional as F
from PIL import Image

SIZE = 64
def aug(img):
    img = img[140:470, 130:460, ::]
    img = Image.fromarray(img, mode='RGB')
    img = F.resize(img, (SIZE, SIZE), Image.ANTIALIAS)
    img = np.array(img)
    # img = x.transpose([2, 1, 0]).flatten()
    cv2.imshow("preview", img)
    return img

import sys
import os
list_of_files = glob.glob("data/slot_demo/with_gripper/demos_2022-07-13-17-08-25_with_gripper.*")

latest_file = max(list_of_files, key=os.path.getctime)
for filename in (latest_file, ):
    print(filename)
    x = np.load(filename, allow_pickle=True)
    import pdb; pdb.set_trace()
    for traj_i in range(x.shape[0]):
        obs = x[traj_i]["observations"]
        rewards = x[traj_i]["rewards"]
        print("TRAJ: ", traj_i, "     LENGTH: ", len(obs))
        for t in range(len(obs)):
            import pdb; pdb.set_trace()
            print("REWARD: ", int(rewards[t]))
            if not obs[t]:
                print(traj_i, t)
                continue

            img = obs[t]["image_observation"]
            img = img[:, :, :] # .transpose([1, 2, 0])
            aug(img)
            # cv2.imshow("preview", img)
            # key = cv2.waitKey() # cv2.waitKey(20)
            key = cv2.waitKey(100)
            # print "key pressed: " + str(key)
            # exit on ESC, you may want to uncomment the print to know which key is ESC for you
            if key == 27 or key == 1048603:
                break

#cv2.destroyWindow("preview")
