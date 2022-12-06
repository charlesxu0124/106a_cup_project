import numpy as np
import cv2

def find_empty_space(img):
	zone1 = img[57:122,188:252]
	zone2 = img[57:122,252:312]
	zone3 = img[62:126,314:372]
	zone4 = img[121:183,188:249]
	zone5 = img[126:184,252:309]
	zone6 = img[126:188,312:369]
	zone7 = img[183:243,190:237]
	zone8 = img[185:244,251:309]
	zone9 = img[185:245,310:366]
	zones = [zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, zone9]
	available_zones = []
	for i in range(len(zones)):
		lower_red = np.array([160, 100, 0])
		upper_red = np.array([180, 255, 255])
		img = cv2.cvtColor(zones[i], cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(img, lower_red, upper_red)
		count = np.sum(np.nonzero(mask))
		print(count)
		if count < 9000:
			available_zones.append(i+1)
	print(available_zones)
	if max(available_zones) == 6 and len(available_zones) > 1:
		available_zones.remove(6)
	return max(available_zones)

def find_bottle_center(img):
	img = img[280:450,150:400]
	output = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Blur using 3 * 3 kernel.
	gray_blurred = cv2.blur(gray, (3, 3))

	# Apply Hough transform on the blurred image.
	detected_circles = cv2.HoughCircles(gray_blurred,
										cv2.HOUGH_GRADIENT, 1, 20, param1=50,
										param2=30, minRadius=1, maxRadius=40)
	if detected_circles is not None:
		# Convert the circle parameters a, b and r to integers.
		detected_circles = np.uint16(np.around(detected_circles))
		for pt in detected_circles[0, :]:
			return 150+pt[0], 280+pt[1]	
	# no bottles found
	return None

if __name__ == '__main__':
	img_dir = '/Users/francisgeng/Desktop/cv/new_bottles/1.png' # feel free to change
	print(find_empty_space(img_dir))
	print(find_bottle_center(img_dir))