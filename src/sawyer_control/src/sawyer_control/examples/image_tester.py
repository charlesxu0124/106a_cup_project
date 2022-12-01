from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
import cv2
from sawyer_control.core.image_env import ImageEnv
from sawyer_control.helper.video_capture import VideoCapture

resource = 0
cap = VideoCapture(resource)

# env = ImageEnv(SawyerReachXYZEnv())
# img = env.get_image(width=84, height=84)
img = cap.read()
cv2.imwrite("test.png", img)
