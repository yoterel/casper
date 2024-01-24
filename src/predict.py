import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# nowhere smart to put this: a map between mediapipe and leap motion hand landmarks
# MP
# 	wrist: 0
# 	thumb0: 1
# 	thumb1: 2
# 	thumb2: 3
# 	thumbtip: 4
# 	index0: 5
# 	index1: 6
# 	index2: 7
# 	indextip: 8
# 	middle0: 9
# 	middle1: 10
# 	middle2: 11
# 	middletip: 12
# 	ring0: 13
# 	ring1: 14
# 	ring2: 15
# 	ringtip: 16
# 	pinky0: 17
# 	pinky1: 18
# 	pinky2: 19
# 	pinkytip: 20
# LEAP
# 	wrist:1
# 	thumb0: 2
# 	thumb1: 5
# 	thumb2: 7
# 	thumbtip:9
# 	index0: 10
# 	index1: 11
# 	index2: 13
# 	index3:15
# 	indextip: 17
# 	middle0: 18
# 	middle1: 19
# 	middle2: 21
# 	middle3: 23
# 	middletip: 25
# 	ring0: 26
# 	ring1: 27
# 	ring2: 29
# 	ring3: 31
# 	ringtip: 33
# 	pinky0: 34
# 	pink1: 35
# 	pink2: 37
# 	pinky3: 39
# 	pinkytip:41

VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(
    model_asset_path="../../resource/models/hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=VisionRunningMode.VIDEO,  # , running_mode=VisionRunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)


def init_detector(video=False):
    my_base_options = python.BaseOptions(
        model_asset_path="../../resource/models/hand_landmarker.task"
    )
    if video:
        my_options = vision.HandLandmarkerOptions(
            base_options=my_base_options,
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
    else:
        my_options = vision.HandLandmarkerOptions(
            base_options=my_base_options,
            num_hands=2,
        )
    my_detector = vision.HandLandmarker.create_from_options(my_options)
    return my_detector


def predict_video(image_orig, i):
    # print(image_orig.shape)
    # if len(image_orig.shape) == 2:
    #     # duplicate the image to make it 3 channel
    #     image = image_orig[:, :, np.newaxis].repeat(3, axis=2).copy()
    # elif image_orig.shape[-1] == 1:
    #     # duplicate the image to make it 3 channel
    #     image = image_orig.repeat(3, axis=2).copy()
    # elif image_orig.shape[-1] == 4:
    #     image = image_orig[:, :, :3].copy()
    # else:
    #     image = image_orig.copy()
    image = image_orig.copy()
    # image = image_orig
    # see https://developers.google.com/mediapipe/api/solutions/python/mp/ImageFormat
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)  # GRAY8
    detection_result = detector.detect_for_video(
        mp_image, i
    )  # detect_for_video(mp_image, i), detect(mp_image)
    if len(detection_result.hand_landmarks) == 0:
        return np.zeros(1), np.zeros(1), False, False
    right_landmarks = np.zeros((21, 2), dtype=np.float32)
    left_landmarks = np.zeros((21, 2), dtype=np.float32)
    right_detected = False
    left_detected = False
    for j in range(len(detection_result.hand_landmarks)):
        handness = detection_result.handedness[j][0].display_name == "Right"
        landmarks = np.array(
            [
                [landmark.x, landmark.y]
                for landmark in detection_result.hand_landmarks[j]
            ]
        ).astype(np.float32)
        landmarks = (landmarks * 2) - 1  # 0:1 to -1:1
        landmarks[:, 1] = -landmarks[:, 1]  # flip y
        if handness:
            right_landmarks = landmarks
            right_detected = True
        else:
            left_landmarks = landmarks
            left_detected = True
    # else:
    #     mp_np = np.zeros(1)
    # print(mp_np.dtype)
    # print(mp_np.shape)
    return left_landmarks, right_landmarks, left_detected, right_detected


def predict_video_aprox(image_orig, i, my_detector):
    # print(image_orig.shape)
    # if len(image_orig.shape) == 2:
    #     # duplicate the image to make it 3 channel
    #     image = image_orig[:, :, np.newaxis].repeat(3, axis=2).copy()
    # elif image_orig.shape[-1] == 1:
    #     # duplicate the image to make it 3 channel
    #     image = image_orig.repeat(3, axis=2).copy()
    # elif image_orig.shape[-1] == 4:
    #     image = image_orig[:, :, :3].copy()
    # else:
    #     image = image_orig.copy()
    image = image_orig.copy()
    # image = image_orig
    # see https://developers.google.com/mediapipe/api/solutions/python/mp/ImageFormat
    if image.shape[-1] == 1 or len(image.shape) == 2:
        image_format = mp.ImageFormat.GRAY8
    elif image.shape[-1] == 4:
        image_format = mp.ImageFormat.SRGBA
    else:
        image_format = mp.ImageFormat.SRGB
    mp_image = mp.Image(image_format=image_format, data=image)
    detection_result = my_detector.detect_for_video(
        mp_image, i
    )  # detect_for_video(mp_image, i), detect(mp_image)
    if len(detection_result.hand_landmarks) == 0:
        return np.zeros(1), np.zeros(1), 0
    right_landmarks = np.zeros((21, 2), dtype=np.float32)
    left_landmarks = np.zeros((21, 2), dtype=np.float32)
    for j in range(len(detection_result.hand_landmarks)):
        handness = detection_result.handedness[j][0].display_name == "Right"
        landmarks = np.array(
            [
                [landmark.x, landmark.y]
                for landmark in detection_result.hand_landmarks[j]
            ]
        ).astype(np.float32)
        landmarks = (landmarks * 2) - 1  # 0:1 to -1:1
        landmarks[:, 1] = -landmarks[:, 1]  # flip y
        if handness:
            right_landmarks = landmarks
        else:
            left_landmarks = landmarks
    # else:
    #     mp_np = np.zeros(1)
    # print(mp_np.dtype)
    # print(mp_np.shape)
    return left_landmarks, right_landmarks, len(detection_result.hand_landmarks)


def predict_single(image_orig, my_detector=None, verbose=False):
    # print(image_orig.shape)
    # if len(image_orig.shape) == 2:
    #     # duplicate the image to make it 3 channel
    #     image = image_orig[:, :, np.newaxis].repeat(3, axis=2).copy()
    # elif image_orig.shape[-1] == 1:
    #     # duplicate the image to make it 3 channel
    #     image = image_orig.repeat(3, axis=2).copy()
    # elif image_orig.shape[-1] == 4:
    #     image = image_orig[:, :, :3].copy()
    # else:
    if verbose:
        print("copying image")
    image = image_orig.copy()
    # image = image_orig
    if my_detector is None:
        if verbose:
            print("initializing detector")
        single_base_options = python.BaseOptions(
            model_asset_path="../../resource/models/hand_landmarker.task"
        )
        single_options = vision.HandLandmarkerOptions(
            base_options=single_base_options, num_hands=1
        )
        my_detector = vision.HandLandmarker.create_from_options(single_options)
    if verbose:
        print("setting image format")
    if image.shape[-1] == 1 or len(image.shape) == 2:
        image_format = mp.ImageFormat.GRAY8
        if verbose:
            print("GRAY8")
    elif image.shape[-1] == 4:
        image_format = mp.ImageFormat.SRGBA
        if verbose:
            print("SRGBA")
    else:
        image_format = mp.ImageFormat.SRGB
        if verbose:
            print("SRGB")
    mp_image = mp.Image(image_format=image_format, data=image)
    if verbose:
        print("detection")
    detection_result = my_detector.detect(mp_image)
    if verbose:
        print("detection done")
    if len(detection_result.hand_landmarks) > 0:
        if verbose:
            print("detection sucess, converting to numpy")
        mp_np = np.array(
            [
                [landmark.x, landmark.y]
                for landmark in detection_result.hand_landmarks[0]
            ]
        ).astype(np.float32)
        mp_np = (mp_np * 2) - 1  # 0:1 to -1:1
        mp_np[:, 1] = -mp_np[:, 1]  # flip y
    else:
        if verbose:
            print("detection failed")
        mp_np = np.zeros(1)
    # print(mp_np.dtype)
    # print(mp_np.shape)
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
    # benchmark mediapipe
    import time
    import cv2
    from pathlib import Path

    raw_image_path = "../../resource/hand.png"  # "../../resource/two_hands.jpg" # "../../resource/hand.png"
    image_path = str(Path(raw_image_path).resolve())
    image_4channel = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 4 channel
    if image_4channel.shape[-1] != 4:
        image_4channel = cv2.cvtColor(image_4channel, cv2.COLOR_RGB2RGBA)
    image_3channel = cv2.imread(image_path)  # 3 channel
    image_1channel = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 1 channel
    image_1channel_dummy = image_1channel.copy()[
        :, :, None
    ]  # 1 channel with dummy axis
    image_3channel_resized = cv2.resize(image_3channel, (512, 512))
    images = {
        "image_4channel": image_4channel,
        "image_3channel": image_3channel,
        "image_1channel": image_1channel,
        "image_1channel_dummy": image_1channel_dummy,
        "image_3channel_resized": image_3channel_resized,
    }
    short_iters = 20
    long_iters = 100
    for key, image in images.items():
        print(image.shape)
        # image = image[:, :, 0:1]
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = cv2.resize(image, (512, 512))
        my_detector = init_detector(False)
        test = predict_single(image, my_detector)  # warmup
        times = []
        failures = 0
        for i in range(0, short_iters):
            start = time.time()
            test = predict_single(image, my_detector)
            times.append(time.time() - start)
            if test.shape == (1,):
                failures += 1
            # print(times[-1])
        single_avg_with_detector = np.array(times).mean()
        single_avg_with_detector_failures = failures

        times = []
        failures = 0
        for i in range(0, short_iters):
            start = time.time()
            test = predict_single(image)
            times.append(time.time() - start)
            if test.shape == (1,):
                failures += 1
            # print(times[-1])
        single_avg_no_detector = np.array(times).mean()
        single_avg_no_detector_failures = failures

        times = []
        failures = 0
        my_video_detector = init_detector(True)
        _, _, n_hands = predict_video_aprox(image, 0, my_video_detector)  # warmup
        if n_hands == 0:
            failures += 1
        for i in range(1, long_iters):
            start = time.time()
            _, _, n_hands = predict_video_aprox(image, i, my_video_detector)
            times.append(time.time() - start)
            if n_hands == 0:
                failures += 1
            # print(times[-1])
        video_avg = np.array(times).mean()
        video_avg_failures = failures

        print("{}:".format(key))
        print(
            "single with detector: {} ({}/{} fails)".format(
                single_avg_with_detector, single_avg_with_detector_failures, short_iters
            )
        )
        print(
            "single without detector: {} ({}/{} fails)".format(
                single_avg_no_detector, single_avg_no_detector_failures, short_iters
            )
        )
        print(
            "video: {} ({}/{} fails)".format(video_avg, video_avg_failures, long_iters)
        )
        print("")
