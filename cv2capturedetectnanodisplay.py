
import cv2
import numpy as np
import sys

import jetson.inference
import jetson.utils


def gstreamer_pipeline (capture_width=800, capture_height=600, display_width=800, display_height=600, framerate=30, flip_method=2):
	return (
		'nvarguscamerasrc ! '
		'video/x-raw(memory:NVMM), '
		'width=(int)%d, height=(int)%d, '
		'format=(string)NV12, framerate=(fraction)%d/1 ! '
		'nvvidconv flip-method=%d ! '
		'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
		'videoconvert ! '
		'video/x-raw, format=(string)BGR ! appsink' \
		% (capture_width, capture_height, framerate, flip_method, display_width, display_height)
	)


def main():

	net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.7)

	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

	if cap.isOpened():

		while True:

			# capture image
			ret, img = cap.read()

			if ret:

				# print("Info de la imagen de opencv a pelo")
				# print(img.dtype)
				# print(img.shape)
				# print("")

				# convert image to RGBA format for cudaToNumpy
				img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)

				# print("Info de la imagen convertida para pasarla por cudaToNumpy")
				# print(img_rgba.dtype)
				# print(img_rgba.shape)
				# print("")

				# cudaToNumpy conversion and detection
				img_cuda = jetson.utils.cudaFromNumpy(img_rgba)
				detections = net.Detect(img_cuda, 800, 600, "box,labels")

				# revert back the cuda-format image into opencv's bgr and display
				img_rgba_w_detections = jetson.utils.cudaToNumpy(img_cuda, 800, 600, 4)

				# print("Info de la imagen img_rgba_w_detections")
				# print(img_rgba_w_detections.dtype)
				# print(img_rgba_w_detections.shape)
				# print("")

				img_bgr_w_detections = cv2.cvtColor(img_rgba_w_detections, cv2.COLOR_RGBA2BGR).astype(np.uint8)

				# print("Info de la imagen img_bgr_w_detections")
				# print(img_bgr_w_detections.dtype)
				# print(img_bgr_w_detections.shape)
				# print("")

				cv2.imshow('sth...', img_bgr_w_detections)

			keyCode = cv2.waitKey(1) & 0xFF
			if keyCode == 27 or keyCode == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

	else:
		print('Unable to open camera')


if __name__ == '__main__':
	main()