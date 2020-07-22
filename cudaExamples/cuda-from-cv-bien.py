
import cv2
import jetson.utils
import argparse


# parse the command line
parser = argparse.ArgumentParser(description='Convert an image from OpenCV to CUDA')

parser.add_argument("file_in", type=str, default="granny_smith_1.jpg", nargs='?', help="filename of the input image to process")
parser.add_argument("file_out", type=str, default="jpg-cuda-from-cv.jpg", nargs='?', help="filename of the output image to save")

opt = parser.parse_args()


# load the image
cv_img = cv2.imread(opt.file_in)

print('OpenCV image size: ' + str(cv_img.shape))
print('OpenCV image type: ' + str(cv_img.dtype))

# convert to CUDA (cv2 images are numpy arrays, in BGR format)
converted_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
bgr_img = jetson.utils.cudaFromNumpy(converted_img)

# print('BGR image: ')
# print(bgr_img)

# # convert from BGR -> RGB
# rgb_img = jetson.utils.cudaAllocMapped((cv_img.shape[0], cv_img.shape[1]), 'rgb8')

# jetson.utils.cudaConvertColor(bgr_img, rgb_img)

# print('RGB image: ')
# print(rgb_img)

# save the image
if opt.file_out is not None:
	jetson.utils.cudaDeviceSynchronize()
	jetson.utils.saveImageRGBA(opt.file_out, bgr_img, cv_img.shape[0], cv_img.shape[1])
	print("saved {:d}x{:d} test image to '{:s}'".format(cv_img.shape[0], cv_img.shape[1], opt.file_out))

