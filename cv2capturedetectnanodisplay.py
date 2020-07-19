
import cv2

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

				# cv2.imshow('sth...', img)

				# convert image to RGBA format for net.Detect()
				img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

				print(img_rgba.dtype)
				print(img_rgba.shape)

				# convert numpy array of float32 to CUDA format for net.Detect()


				# jejeje
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