import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# def init_detector():
#     # some inits
#     VisionRunningMode = mp.tasks.vision.RunningMode
#     base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
#     options = vision.HandLandmarkerOptions(
#         base_options=base_options, num_hands=1  # , running_mode=VisionRunningMode.VIDEO
#     )
#     detector = vision.HandLandmarker.create_from_options(options)
#     return detector

VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(
    model_asset_path="../../resource/hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=VisionRunningMode.VIDEO,  # , running_mode=VisionRunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)


def predict_single(image, i):
    # myimage = image.copy()
    if image.shape[-1] == 4:
        image = image[:, :, :3].copy()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect_for_video(
        mp_image, i
    )  # detect_for_video(mp_image, i), detect(mp_image)
    if len(detection_result.hand_landmarks) > 0:
        mp_np = np.array(
            [
                [landmark.x, landmark.y]
                for landmark in detection_result.hand_landmarks[0]
            ]
        ).astype(np.float32)
    else:
        mp_np = np.zeros(1)
    # print(mp_np.dtype)
    # print(mp_np.shape)
    mp_np = (mp_np * 2) - 1  # 0:1 to -1:1
    mp_np[:, 1] = -mp_np[:, 1]  # flip y
    return mp_np


def myprint(x):
    print(x.shape)


def iden(x):
    return x


def myprofile(x):
    y = x[:, :, :3].copy()
    # y = y[:, :, :3].copy()
    return y


if __name__ == "__main__":
    import time
    import cv2

    image = cv2.imread("../../resource/hand.png")
    # image = cv2.resize(image, (512, 512))
    # print(image.shape)
    times = []
    test = predict_single(image, 0)
    for i in range(1, 500):
        start = time.time()
        test = predict_single(image, i)
        times.append(time.time() - start)
        print(times[-1])
    print(np.array(times).mean())
