import numpy as np
import argparse
import time
import cv2


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2):
  """
  Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
  """
  return (
      f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
      f'width=(int){capture_width}, height=(int){capture_height}, ' +
      f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
      f'nvvidconv flip-method={flip_method} ! ' +
      f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
      'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", default="pednet/snapshot_iter_70800.caffemodel", help="path to Caffe pre-trained model")
  parser.add_argument("-p", "--prototxt", default="pednet/deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
  parser.add_argument("-l", "--labels", default="pednet/class_labels.txt", help="path to class labels")
  parser.add_argument("-c", "--confidence", type=float, default=0.7, help="minimum probability to filter weak detections")
  parser.add_argument("-o", "--output", type=str, help="path to optional output video file")
  parser.add_argument("-x", "--width", type=int, default=1280, help="capture width (default 1280)") # TODO debería haber moar resols??...
  parser.add_argument("-y", "--height", type=int, default=720, help="capture height (default 720)")
  args = vars(parser.parse_args())

  W = args["width"]
  H = args["height"]
  print("[INFO] loading classes...")
  CLASSES = []
  with open(args["labels"]) as f:
    CLASSES = f.readlines()

  CLASSES = [line.strip() for line in CLASSES]

  print("[INFO] loading model...")
  net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

  print("[INFO] starting video stream...")
  capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)

  while True:
    ret, frame = capture.read()
    if ret:
      cv2.imshow("Wusup", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


if __name__ == "__main__":
  main()
