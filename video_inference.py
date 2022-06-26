from utils.general import check_img_size, non_max_suppression, scale_coords
from models.experimental import attempt_load
from utils.plots import plot_one_box
from playsound import playsound
import numpy as np
import torch
import gtts
import cv2
import pyttsx3


img_size = 416
model = attempt_load("yolov5n.pt")
stride = int(model.stride.max())
names = model.names

meter_pixel_ratio = 70

img_size = check_img_size(img_size, s=stride)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)

engine = pyttsx3.init()
def predict(confidence_thresh = 0.5, iou_thresh = 0.45, use_webcam = True):
	global model, stride, img_size
	
	if use_webcam:
		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	frame_ctr = 0
	while True:
		ret, img = cap.read()
		if not ret:
			break
		
		if not use_webcam:
			if frame_ctr%500 != 0:
				frame_ctr += 1
				continue
		
		frame_ctr += 1
		# img = cv2.imread(img_path)
		img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = letterbox(img0, img_size, stride=stride)[0]

		img = img[:, :, ::-1].transpose(2, 0, 1)
		img = np.ascontiguousarray(img)

		img = torch.from_numpy(img)
		img = img/255.0

		if img.ndimension() == 3:
			img = img.unsqueeze(0)


		pred = model(img, augment=True)[0]

		pred = non_max_suppression(pred, confidence_thresh, iou_thresh, classes=None, agnostic=True)

		im0 = img0
		
		final_text = ""
		
		for i, det in enumerate(pred):  # detections per image
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
				for *xyxy, conf, cls in reversed(det):
					x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3] 
					w = abs(x2 - x1)
					h = abs(y2 - y1)

					center_x = int(x1 + w//2)
					center_y = int(y1 + h//2)
					
					cv2.circle(im0, (center_x, center_y), 10, (0, 0, 255), -1)
					
					object_name = names[int(cls)]
					final_text = f"{object_name} Detected. " + final_text

	
					label = f'{object_name} : {conf:.2f}'
					plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=3)

		im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
		cv2.imshow("Frame", im0)


		k = cv2.waitKey(30)
		if k == ord('q'):
			break
		if len(final_text)>0:
			engine.say(final_text)
			engine.runAndWait()

predict(use_webcam = True)
