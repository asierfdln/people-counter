
import jetson.inference
import jetson.utils

import argparse
import sys
import os

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load, see below for options")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.7, help="minimum detection threshold to use")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0)\nor for VL42 cameras the /dev/video node to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=800, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=600, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create the camera and display and font
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()
font = jetson.utils.cudaFont()

# dlib tracker for detections
tracker = None

# variables for text-overlaying
INCR = 35

# process frames until user exits
while display.IsOpen():

	# capture the image
	img, width, height = camera.CaptureRGBA(zeroCopy=1)

	# cudatonumpy conversion
	img_numpy = jetson.utils.cudaToNumpy(img, width, height, 4)
	print(type(img_numpy))
	print(img_numpy.shape)
	print(img_numpy.dtype)

	# numpytocuda conversion
	img_cuda = jetson.utils.cudaFromNumpy(img_numpy)

	# detect objects in the image (with overlay)
	detections = net.Detect(img, width, height, opt.overlay)

	# print the detections
	# print("detected {:d} objects in image".format(len(detections)))

	ypos = 5
	rects = []
	os.system("clear")
	for detection in detections:
		print(f'{net.GetClassDesc(detection.ClassID)} - {detection.Center}')
		print(f'    {str(int(detection.Center[1] + detection.Height / 2))}')
		font.OverlayText(img, width, height, \
			"({:s}) - ({}, {})".format(
				net.GetClassDesc(detection.ClassID), \
				str(int(detection.Center[0])), \
				str(int(detection.Center[1])) \
			), \
			5, ypos, font.White, font.Gray40)
		ypos = ypos + INCR
		# rectangle = [ \
		# 	int(detection.Center[0] - detection.Width / 2), \
		# 	int(detection.Center[1] - detection.Height / 2), \
		# 	int(detection.Center[0] + detection.Width / 2), \
		# 	int(detection.Center[1] + detection.Height / 2) \
		# ]
		# rects.append(rectangle)

	# # putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
	# cv2.putText( \
	# 	img, \
	# 	"wusup", \
	# 	(50, 50), \
	# 	cv2.FONT_HERSHEY_SIMPLEX, \
	# 	0.45, \
	# 	(0, 255, 0), \
	# 	2 \
	# )

	# render the image
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# synchronize with the GPU
	# if len(detections) > 0:
	#	jetson.utils.cudaDeviceSynchronize()

	# print out performance info
	# net.PrintProfilerTimes()
