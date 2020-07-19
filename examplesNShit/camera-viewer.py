
import jetson.utils
import argparse

# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")

opt = parser.parse_args()
print(opt)

# create display window
display = jetson.utils.glDisplay()

# create camera device
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)

# open the camera for streaming
camera.Open()

# capture frames until user exits
while display.IsOpen():
	image, width, height = camera.CaptureRGBA(zeroCopy=1)
	array = jetson.utils.cudaToNumpy(image, width, height, 4)
	print(type(array))
	print(array.shape)
	print(array.dtype)
	display.RenderOnce(image, width, height)
	display.SetTitle("{:s} | {:d}x{:d} | {:.0f} FPS".format("Camera Viewer", width, height, display.GetFPS()))

# close the camera
camera.Close()
