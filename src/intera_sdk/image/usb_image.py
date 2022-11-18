import numpy as np
# import skimage.transform
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from scipy import interpolate
import matplotlib.pyplot as plt

RAW_IMAGE_HEIGHT = 480
RAW_IMAGE_WIDTH = 640
L_MARGIN = 90
R_MARGIN = (RAW_IMAGE_WIDTH - RAW_IMAGE_HEIGHT) - L_MARGIN
UL = (160, 170)
UR = (561, 164)
LL = (89, 450)
LR = (593, 466)

class USBImagePuller:
    def __init__(self, camport=0):
        self.camport = camport
        self.cap = cv2.VideoCapture(self.camport)
        assert self.cap.isOpened(), "/dev/video{} is not opened.".format(self.camport)

    def stream_image(self):
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('video', frame)
            # time.sleep(0.01)
            break

    def pull_image(self, image_name=None):
        ret, frame = self.cap.read()
        if image_name:
            assert image_name[-4:] in [".jpg", ".png"], "Invalid image_name {}".format(image_name)
            cv2.imwrite(image_name, frame)
        return frame

class CSpline:
    def __init__(self, points, skip=[], duration=1., bc_type='clamped'):
        n_points = points.shape[0] + len(skip)
        self._duration = duration
        while True:
            if skip == [] or skip[-1] != n_points - 1:
                break
            else:
                skip = skip[:-1]
                n_points -= 1
        x = np.linspace(0, duration, n_points)
        x = np.delete(x, skip)
            
        # self._cs = interpolate.CubicSpline(np.linspace(0, duration, n_points-len(skip)), points, bc_type=bc_type)
        self._cs = interpolate.CubicSpline(x, points, bc_type=bc_type)

    def get_cs(self):
        return self._cs

    def get(self, t):
        t = np.array(min(t, self._duration))

        return self._cs(t), self._cs(t, nu=1), self._cs(t, nu=2)

def process_image_rgb(image, desired_h=64, desired_w=64):
    # Currently only supporting downsampling to square, 2**i x 2**i image.
    assert desired_h == desired_w, \
        "desired_h: {} should equal desired_w: {}".format(desired_h, desired_w)
    assert desired_h == int(2 ** np.round(np.log2(desired_h))), "desired_h {} not a power of 2".format(desired_h)
    assert desired_w == int(2 ** np.round(np.log2(desired_w))), "desired_w {} not a power of 2".format(desired_w)

    # flip upside-down (0), then leftside-right (1)
    # image = np.flip(image, axis=(0,1))
    h, w, _ = image.shape
    assert h == RAW_IMAGE_HEIGHT and w == RAW_IMAGE_WIDTH, \
        "Dimensions {}, {} do not match expected raw image dimensions {}, {}".format(
            h, RAW_IMAGE_HEIGHT, w, RAW_IMAGE_WIDTH
        )

    # Crop left and right.
    # image = image[:,L_MARGIN : RAW_IMAGE_WIDTH - R_MARGIN]

    # Resize square image to a power of 2.
    resize_to = next(
        2 ** i for i in reversed(range(10))
        if 2 ** i < image.shape[0])
    image = skimage.transform.resize(
        image, (resize_to, resize_to), anti_aliasing=True, mode='constant')

    # Downsample 2**i x 2**i dimensioned square image.
    height_factor = image.shape[0] // desired_h
    width_factor = image.shape[1] // desired_w
    image = skimage.transform.downscale_local_mean(
        image, (width_factor, height_factor, 1))
    image = skimage.util.img_as_ubyte(image)

    return image

def find_color(name, hsv, lower, upper, sagments):
    # cv2.imshow("hsv", hsv)

    if lower[0] > upper[0]:
        mask1 = cv2.inRange(hsv, (0, lower[1], lower[2]), upper)
        mask2 = cv2.inRange(hsv, lower, (179, upper[1], upper[2]))
        res1 = cv2.bitwise_and(frame, frame, mask=mask1)
        res2 = cv2.bitwise_and(frame, frame, mask=mask2)
        res = cv2.bitwise_or(res1, res2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask= mask)
    # cv2.imshow("blue", hsv)
    cv2.imshow(name, res)
    contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    backup = 0
    for cnt in contours:
        cnt = np.asarray(cnt)
        center = np.mean(cnt,0)[0]
        if not inRegion(center):
            pass
        else:
            if len(cnt) < 2 or len(cnt) > 40:
                pass
            elif np.count_nonzero(sagments) > 0 and min([np.linalg.norm(center - prev_centers) for prev_centers in sagments if prev_centers != None]) < 10:
                backup = center
            else:
                return tuple([int(c) for c in center])
    if isinstance(backup, tuple):
        return backup

    # cv2.imshow("Mask", mask)

def inRegion(pos):
    cof = np.polyfit([UL[0], LL[0]], [UL[1], LL[1]], 1)
    if pos[1] < cof[0]*pos[0] + cof[1]:
        return False
    cof = np.polyfit([UL[0], UR[0]], [UL[1], UR[1]], 1)
    if pos[1] < cof[0]*pos[0] + cof[1]:
        return False
    cof = np.polyfit([LL[0], LR[0]], [LL[1], LR[1]], 1)
    if pos[1] > cof[0]*pos[0] + cof[1]:
        return False
    cof = np.polyfit([UR[0], LR[0]], [UR[1], LR[1]], 1)
    if pos[1] < cof[0]*pos[0] + cof[1]:
        return False
    return True

if __name__ == "__main__":
    resource = "/dev/video0"
    cap = cv2.VideoCapture(resource)
    # import pdb; pdb.set_trace()
    rval, frame = cap.read()

    while rval:
        rval, frame = cap.read()

        cv2.imshow('', frame)
        cv2.waitKey(1)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -240)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    # if not cap.isOpened():
    #     print "Error opening resource: " + str(resource)
    #     print "Maybe opencv VideoCapture can't open it"
    #     exit(0)

    # print "Correctly opened resource, starting to show feed."
    # rval, frame = cap.read()
    # while rval:
    #     # cv2.imshow("Stream", frame)
    #     rval, frame = cap.read()

    #     sagments = []
    #     #seperate color blocks
    #     blurred = cv2.GaussianBlur(frame, (7, 7), sigmaX=10, sigmaY=10)
    #     cv2.imshow("blurred", blurred)
    #     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #     cv2.imshow("hsv", hsv)

    #     #blue
    #     lower = (90 , 100, 50)
    #     upper = (130, 255, 255)
    #     center = find_color("blue", hsv, lower, upper, sagments)
    #     sagments.append(center)
    #     cv2.circle(frame, center, 2, (0,0,255), thickness = 3)

    #     #green
    #     lower = (60 , 10, 30)
    #     upper = (90, 255, 255)
    #     center = find_color("green", hsv, lower, upper, sagments)
    #     sagments.append(center)
    #     cv2.circle(frame, center, 2, (0,0,255), thickness = 3)

    #     #orange
    #     lower = (0, 30, 100)
    #     upper = (16, 255, 255)
    #     center = find_color("orange", hsv, lower, upper, sagments)
    #     sagments.append(center)
    #     cv2.circle(frame, center, 2, (0,0,255), thickness = 3)

    #     #yellow
    #     lower = (16, 20, 10)
    #     upper = (60, 30, 255)
    #     center = find_color("yellow", hsv, lower, upper, sagments)
    #     sagments.append(center)
    #     cv2.circle(frame, center, 2, (0,0,255), thickness = 3)

    #     # pink
    #     lower = (150, 20, 100)
    #     upper = (165, 255, 200)
    #     center = find_color("pink", hsv, lower, upper, sagments)
    #     sagments.append(center)
    #     cv2.circle(frame, center, 2, (0,0,255), thickness = 3)
        

    #     x = [i[0] for i in sagments if i != None]
    #     y = [i[1] for i in sagments if i != None]
    #     skip = [i for i in range(len(sagments)) if sagments[i] == None] 
    #     # x.insert(0, 393)
    #     # y.insert(0, 338)
    #     # tck = interpolate.splprep([x, y], u=0)
    #     if len(x) > 2:
    #         points = np.c_[x, y]
    #         spline = CSpline(np.asarray(points), skip=skip)
    #         cs = spline.get_cs()
    #         xs = np.linspace(0, 1, 20)
    #         points = np.ndarray.tolist(cs(xs))
    #         for point in points:
    #             cv2.circle(frame, (int(point[0]), int(point[1])), 1, (255, 0, 0), thickness=2)
    #     cv2.imshow("result", frame)


        # blurred = cv2.GaussianBlur(frame, (5, 5), 5)
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # canny = cv2.Canny(gray, 50, 100)

        # cv2.imshow('Canny', canny)
        # mask = cv2.inRange(gray, 50, 70)
        # contours, _= cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # cv2.imshow("mask", mask)
        # # for cnt in contours:
        # #     print(cnt[0][0])
        # #     if inRegion(cnt[0][0]):
        # #         cont = cv2.drawContours(frame, cnt, -1, (0, 0, 255))
        # cont = cv2.drawContours(frame, contours, -1, (0, 0, 255))
        # cv2.imshow("countour", cont)
        
        # key = cv2.waitKey(20)
        # # print "key pressed: " + str(key)
        # # exit on ESC, you may want to uncomment the print to know which key is ESC for you
        # if key == 27 or key == 1048603:
        #     break
    # cv2.destroyWindow("preview")


    # image_puller = USBImagePuller()
    # count = 0
    # while True:
    #     frame = image_puller.pull_image('picture_{}.png'.format(count))
    #     frame = process_image_rgb(frame, 64, 64)
    #     cv2.imwrite("randompicture_{}_processed.png".format(count), frame)
    #     print("frame.shape", frame.shape)
    #     # time.sleep(1)
    #     count += 1
    #     if count == 1:
    #         break
    # #stream_image()
    # #time.sleep(10)