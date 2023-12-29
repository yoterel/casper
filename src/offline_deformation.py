import gsoup
from pathlib import Path
import cv2
import numpy as np
from deformations import (
    mls_affine_deformation,
    mls_similarity_deformation,
    mls_rigid_deformation,
)


def to_binary():
    input_path = Path("C:/src/augmented_hands/debug/ss")
    output_path = Path("C:/src/augmented_hands/debug/registration")
    raw = [x for x in input_path.glob("*raw_cam.png")]
    render = [x for x in input_path.glob("*render.png")]
    binary = [Path(x.stem + "_bin.png") for x in raw]
    binary_render = [Path(x.stem + "_render_bin.png") for x in raw]
    raw_images = gsoup.load_images(raw, as_grayscale=True)
    render_images = gsoup.load_images(render)
    render_images = render_images[:, :, :, -1]
    thr = 10
    raw_images[raw_images >= thr] = 255
    raw_images[raw_images < thr] = 0
    gsoup.save_images(raw_images[:, :, :, None], output_path, binary)
    gsoup.save_images(render_images[:, :, :, None], output_path, binary_render)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def NDC_to_pixel(landmarks, width, height, flipy=True, is_zero_to_one=False):
    new_landmarks = np.copy(landmarks)
    if not is_zero_to_one:
        multiplier = -1 if flipy else 1
        new_landmarks[:, 0] = (landmarks[:, 0] + 1) / 2
        new_landmarks[:, 1] = (multiplier * landmarks[:, 1] + 1) / 2
    new_landmarks[:, 0] = new_landmarks[:, 0] * width
    new_landmarks[:, 1] = new_landmarks[:, 1] * height
    new_landmarks = new_landmarks.round().astype(np.int32)
    return new_landmarks


def draw_landmark(rgb_image, detection_result, index, mp=True):
    annotated_image = np.copy(rgb_image)
    color = (0, 0, 255)
    if mp:
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            point = np.array(
                [
                    hand_landmarks[index].x * rgb_image.shape[1],
                    hand_landmarks[index].y * rgb_image.shape[0],
                ]
            )
    else:
        point = np.array(
            [
                ((detection_result[index, 0] + 1) / 2) * rgb_image.shape[1],
                ((-detection_result[index, 1] + 1) / 2) * rgb_image.shape[0],
            ]
        )
    point = point.astype(np.int32)
    cv2.circle(
        annotated_image, tuple(point), 5, color, thickness=1, lineType=8, shift=0
    )
    return annotated_image


def draw_landmarks_on_image(rgb_image, detection_result):
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def predict():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import time

    root_folder = Path("C:/src/augmented_hands/debug/ss")
    output_folder = Path("C:/src/augmented_hands/debug/registration")
    keypoints_files = [x for x in root_folder.glob("*keypoints.npy")]
    raw_cam_files = [x for x in root_folder.glob("*raw_cam.png")]
    render_files = [x for x in root_folder.glob("*render.png")]

    # some inits
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1  # , running_mode=VisionRunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    for i in range(len(keypoints_files)):
        print(render_files[i])
        keypoints = np.load(keypoints_files[i])
        raw_cam = gsoup.load_image(raw_cam_files[i])
        render = gsoup.load_image(render_files[i])
        # first predict keypoints from raw_cam
        raw_cam_rgb = raw_cam[:, :, :3].copy()
        raw_cam_single = raw_cam[:, :, 0].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_cam_rgb)
        detection_result = detector.detect(mp_image)  # detect_for_video(mp_image, i)
        leap_pixel_sapce = NDC_to_pixel(
            keypoints, raw_cam_rgb.shape[1], raw_cam_rgb.shape[0], True, False
        )
        mp_np = np.array(
            [
                [landmark.x, landmark.y]
                for landmark in detection_result.hand_landmarks[0]
            ]
        )
        mp_pixel_sapce = NDC_to_pixel(
            mp_np, raw_cam_rgb.shape[1], raw_cam_rgb.shape[0], False, True
        )

        # for i in range(len(detection_result.hand_landmarks[0])):
        #     annotated_image = draw_landmark(raw_cam_rgb, detection_result, i)
        #     print("MP: {}".format(i))
        #     cv2.imshow("window", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)
        # for i in range(len(keypoints)):
        #     print("LEAP: {}".format(i))
        #     annotated_image = draw_landmark(raw_cam_rgb, keypoints, i, False)
        #     cv2.imshow("window", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)
        # closing all open windows
        # cv2.destroyAllWindows()
        # now deform render using these keypoints
        # create grid size of image
        height, width, _ = render.shape
        gridX = np.arange(width, dtype=np.int16)
        gridY = np.arange(height, dtype=np.int16)
        vy, vx = np.meshgrid(gridX, gridY)
        # define control points and their target
        # source (leap, render)
        p = np.array(
            [
                # fix
                leap_pixel_sapce[1],
                leap_pixel_sapce[2],
                leap_pixel_sapce[5],
                leap_pixel_sapce[10],
                leap_pixel_sapce[11],
                leap_pixel_sapce[18],
                leap_pixel_sapce[19],
                leap_pixel_sapce[26],
                leap_pixel_sapce[27],
                leap_pixel_sapce[34],
                leap_pixel_sapce[35],
                # move
                # tips
                leap_pixel_sapce[9],
                leap_pixel_sapce[17],
                leap_pixel_sapce[25],
                leap_pixel_sapce[33],
                leap_pixel_sapce[41],
                # one before tips
                leap_pixel_sapce[7],
                leap_pixel_sapce[15],
                leap_pixel_sapce[23],
                leap_pixel_sapce[31],
                leap_pixel_sapce[39],
            ]
        )
        # destination (mp, raw_cam)
        q = np.array(
            [
                # fix
                leap_pixel_sapce[1],
                leap_pixel_sapce[2],
                leap_pixel_sapce[5],
                leap_pixel_sapce[10],
                leap_pixel_sapce[11],
                leap_pixel_sapce[18],
                leap_pixel_sapce[19],
                leap_pixel_sapce[26],
                leap_pixel_sapce[27],
                leap_pixel_sapce[34],
                leap_pixel_sapce[35],
                # move
                # tips
                mp_pixel_sapce[4],
                mp_pixel_sapce[8],
                mp_pixel_sapce[12],
                mp_pixel_sapce[16],
                mp_pixel_sapce[20],
                # one before tips
                mp_pixel_sapce[3],
                mp_pixel_sapce[7],
                mp_pixel_sapce[11],
                mp_pixel_sapce[15],
                mp_pixel_sapce[19],
            ]
        )
        p_viz = p.copy()
        q_viz = q.copy()
        # swap columns of p and q
        p[:, [1, 0]] = p[:, [0, 1]]
        q[:, [1, 0]] = q[:, [0, 1]]

        # deform using affine
        affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(render)
        aug1[vx, vy] = render[tuple(affine)]
        # show undistorted vs distorted image by blending with cam_image

        similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
        aug2 = np.ones_like(render)
        aug2[vx, vy] = render[tuple(similar)]

        rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        aug3 = np.ones_like(render)
        aug3[vx, vy] = render[tuple(rigid)]

        result0 = visualize_deformation(render, raw_cam_rgb, raw_cam_single)
        for point in p_viz:
            color = (0, 0, 255)
            cv2.circle(
                result0, tuple(point), 5, color, thickness=1, lineType=8, shift=0
            )
        for point in q_viz:
            color = (0, 255, 0)
            cv2.circle(
                result0, tuple(point), 5, color, thickness=1, lineType=8, shift=0
            )
        result1 = visualize_deformation(aug1, raw_cam_rgb, raw_cam_single)
        result2 = visualize_deformation(aug2, raw_cam_rgb, raw_cam_single)
        result3 = visualize_deformation(aug3, raw_cam_rgb, raw_cam_single)

        top_half = np.hstack((result0, result1))
        bot_half = np.hstack((result2, result3))
        full_image = np.vstack((top_half, bot_half))
        cv2.imshow("window", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        # gsoup.save_image(aug1, Path(output_folder, "aug1.png"))
        # gsoup.save_image(aug2, Path(output_folder, "aug2.png"))
        # gsoup.save_image(aug3, Path(output_folder, "aug3.png"))

    # raw_cam_image_path = "C:/src/augmented_hands/debug/ss/sg0o.f_raw_cam.png"
    # raw_cam_image_path = "C:/src/augmented_hands/debug/ss/sg0o.f_raw_cam.png"
    # cam_image = gsoup.load_image(raw_cam_image_path)
    # cam_image = cam_image[:, :, :3].copy()
    # # img = cv2.imread(image_path)
    # # cv2.imshow("window", image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # mp_image = mp.Image.create_from_file(image_path)
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cam_image)
    # # STEP 2: Create an HandLandmarker object.
    # VisionRunningMode = mp.tasks.vision.RunningMode
    # base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    # options = vision.HandLandmarkerOptions(
    #     base_options=base_options, num_hands=1  # , running_mode=VisionRunningMode.VIDEO
    # )
    # detector = vision.HandLandmarker.create_from_options(options)

    # # STEP 3: Load the input image.
    # # mp_image = mp.Image.create_from_file("image.jpg")

    # # STEP 4: Detect hand landmarks from the input image.
    # # for i in range(10):
    # # start = time.time()
    # detection_result = detector.detect(mp_image)  # detect_for_video(mp_image, i)
    # # print(time.time() - start)
    # # STEP 5: Process the classification result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    # cv2.imshow("window", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # # closing all open windows
    # cv2.destroyAllWindows()


def visualize_deformation(deformed_image, raw_cam_rgb, raw_cam_single):
    thr_image = np.ones_like(raw_cam_single)
    thr_image[raw_cam_single > 10] = 255
    thr_image[raw_cam_single <= 10] = 0
    red_image = np.zeros_like(raw_cam_rgb)
    red_image[:, :, 0] = 255
    purple_image = np.zeros_like(raw_cam_rgb)
    purple_image[:, :, 0] = 255
    purple_image[:, :, 2] = 255
    purple_mask = (thr_image <= 0) & (deformed_image[:, :, -1] > 0)
    purple_mask = purple_mask[:, :, None].repeat(3, axis=2)
    red_mask = (thr_image > 0) & (deformed_image[:, :, -1] <= 0)
    red_mask = red_mask[:, :, None].repeat(3, axis=2)
    draw_mask = (thr_image > 0) & (deformed_image[:, :, -1] > 0)
    draw_mask = draw_mask[:, :, None].repeat(3, axis=2)
    # draw_mask4 = draw_mask[:, :, None].repeat(4, axis=2)
    new_image = np.zeros_like(raw_cam_rgb)
    new_image[purple_mask] = purple_image[purple_mask]
    new_image[draw_mask] = deformed_image[:, :, :3][draw_mask]
    new_image[red_mask] = red_image[red_mask]
    return new_image


if __name__ == "__main__":
    predict()
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
