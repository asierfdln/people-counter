
# comando de pruebas -> python3 cv2capturedetectnanodisplay.py --network=coco-bottle
# comando de pruebas -> python3 cv2capturedetectnanodisplay.py --network=multiped

import cv2
import dlib
import numpy as np
import sys

import jetson.inference
import jetson.utils


WIDTH = 800
HEIGHT = 600

COUNTING_OFFSET = 50


def gstreamer_pipeline (capture_width=WIDTH, capture_height=HEIGHT, display_width=800, display_height=600, framerate=30, flip_method=2):
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

	net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.99)

	trackers= []
	startX = None
	startY = None
	endX = None
	endY = None

	redo_detection = False

	contador_yendo_derecha = 0
	contador_yendo_izquierda = 0

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

				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				if len(trackers) == 0 or redo_detection:

					redo_detection = False
					trackers = []
					print("Tamosssss detectin' brooooooooo")

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
					# img_rgba_w_detections = jetson.utils.cudaToNumpy(img_cuda, 800, 600, 4)
# TODO sobra
					# print("Info de la imagen img_rgba_w_detections")
					# print(img_rgba_w_detections.dtype)
					# print(img_rgba_w_detections.shape)
					# print("")

					# img_bgr_w_detections = cv2.cvtColor(img_rgba_w_detections, cv2.COLOR_RGBA2BGR).astype(np.uint8)
# TODO sobra
					# print("Info de la imagen img_bgr_w_detections")
					# print(img_bgr_w_detections.dtype)
					# print(img_bgr_w_detections.shape)
					# print("")

					rects = [] # TODO move up??
					for detection in detections:
						rectangle = [ \
							int(detection.Center[0] - detection.Width / 2), \
							int(detection.Center[1] - detection.Height / 2), \
							int(detection.Center[0] + detection.Width / 2), \
							int(detection.Center[1] + detection.Height / 2) \
						]
						rects.append(rectangle)

					# TODO merge with loopthang above??
					if len(rects) > 0:
						for rectangle in rects:
							list_w_tracker = []
							tracker = dlib.correlation_tracker()
							rect_dlib = dlib.rectangle(rectangle[0], rectangle[1], rectangle[2], rectangle[3])
							tracker.start_track(img_rgb, rect_dlib)
							# trackers.append(tracker)
							list_w_tracker.append(tracker)
							if (rectangle[0] + (rectangle[2] - rectangle[0]) / 2) <= (WIDTH / 2):
								list_w_tracker.append(-1)
								print(f'Así andamos en la izquierda bro ------> {list_w_tracker}')
							elif (rectangle[0] + (rectangle[2] - rectangle[0]) / 2) >= (WIDTH / 2):
								list_w_tracker.append(1)
								print(f'Así andamos en la derecha bro ------> {list_w_tracker}')
							trackers.append(list_w_tracker)

					print(f'UEEEEEEEEEEEEEEEE tenemos {str(len(trackers))} tracker(s) en marcha hulio')

				else:

					for list_w_tracker in trackers:

						list_w_tracker[0].update(img_rgb)
						pos = list_w_tracker[0].get_position()

						startX = int(pos.left())
						startY = int(pos.top())
						endX = int(pos.right())
						endY = int(pos.bottom())

						if (startX + (endX - startX) / 2) <= ((WIDTH / 2) - COUNTING_OFFSET) and list_w_tracker[1] == 1:
							list_w_tracker[1] = -1
							contador_yendo_izquierda = contador_yendo_izquierda + 1
							if contador_yendo_derecha > 0:
								contador_yendo_derecha = contador_yendo_derecha - 1
						elif (startX + (endX - startX) / 2) >= ((WIDTH / 2) + COUNTING_OFFSET) and list_w_tracker[1] == -1:
							list_w_tracker[1] = 1
							contador_yendo_derecha = contador_yendo_derecha + 1
							if contador_yendo_izquierda > 0:
								contador_yendo_izquierda = contador_yendo_izquierda - 1

						print(f'Así seguimos bro ------> {list_w_tracker}')

						cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

				cv2.putText( \
					img, \
					f'thingz pa la izquierda -> {contador_yendo_izquierda}', \
					(0, 23), \
					cv2.FONT_HERSHEY_SIMPLEX, \
					0.45, \
					(0, 255, 0), \
					1 \
				)

				cv2.putText( \
					img, \
					f'thingz pa la derecha -> {contador_yendo_derecha}', \
					(0, 10), \
					cv2.FONT_HERSHEY_SIMPLEX, \
					0.45, \
					(0, 255, 0), \
					1 \
				)

				cv2.imshow('sth...', img)

			keyCode = cv2.waitKey(1) & 0xFF
			if keyCode == 27 or keyCode == ord('q'):
				break
			elif keyCode == ord('a'):
				contador_yendo_izquierda = 0
				contador_yendo_derecha = 0
			elif keyCode == ord('r'):
				redo_detection = True

		cap.release()
		cv2.destroyAllWindows()

	else:
		print('Unable to open camera')


if __name__ == '__main__':
	main()