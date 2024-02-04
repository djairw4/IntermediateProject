import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize


def run(model: str, max_results: int, score_threshold: float, 
        image: str) -> None:

  # Initialize the object detection model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.ObjectDetectorOptions(base_options=base_options,
                                         running_mode=vision.RunningMode.IMAGE,
                                         max_results=max_results, score_threshold=score_threshold)
  detector = vision.ObjectDetector.create_from_options(options)

  mp_image = mp.Image.create_from_file(image)
  detection_result = detector.detect(mp_image)
  image_copy = np.copy(mp_image.numpy_view())
  annotated_image = visualize(image_copy, detection_result)
  rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  cv2.imshow('object_detection', rgb_annotated_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max number of detection results.',
      required=False,
      default=5)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of detection results.',
      required=False,
      type=float,
      default=0.25)
  parser.add_argument(
      '--image',
      help='Path of the image file.',
      required=False,
      default="test/8.jpg")
  args = parser.parse_args()

  run(args.model, int(args.maxResults),
      args.scoreThreshold, args.image)


if __name__ == '__main__':
  main()
