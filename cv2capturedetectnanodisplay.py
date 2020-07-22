
import cv2
import numpy as np

import jetson.inference


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

	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

	if cap.isOpened():
		while True:

			# capture image
			ret, img = cap.read()

			if ret:

				cv2.imshow('sth...', img)

				print("Info de la imagen de opencv a pelo")
				print(img.dtype)
				print(img.shape)
				print("")

				# convert image to RGBA format for net.Detect()
				img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)

				print("Info de la imagen convertida para pasarla por net.Detect()")
				print(img_rgba.dtype)
				print(img_rgba.shape)
				print("")

				# convert numpy array of uint8 to CUDA format in float32 for net.Detect()
				# img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA).astype(np.float32)
				# img = jetson.utils.cudaFromNumpy(img)
				# detections = net.Detect(img, 1280, 720)
				# img = jetson.utils.cudaToNumpy(img, 1280, 720, 4)
				# img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB).astype(np.uint8)
				# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

				# jejeje, negativo xD (pon img para ver bien, pero claro necesitas poner
				# las buenas infos de detecciones antes...)
				# cv2.imshow('sth...', img_rgba)

			keyCode = cv2.waitKey(30) & 0xFF
			if keyCode == 27 or keyCode == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

	else:
		print('Unable to open camera')


if __name__ == '__main__':
	main()