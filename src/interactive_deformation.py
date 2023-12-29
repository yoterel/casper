import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os
import re
from deformations import (
    mls_affine_deformation,
    mls_similarity_deformation,
    mls_rigid_deformation,
)
global deformation_output

np.seterr(divide="ignore", invalid="ignore")





def demo():
    p = np.array(
        [
            [155, 30],
            [155, 125],
            [155, 225],
            [235, 100],
            [235, 160],
            [295, 85],
            [293, 180],
        ]
    )
    q = np.array(
        [
            [211, 42],
            [155, 125],
            [100, 235],
            [235, 80],
            [235, 140],
            [295, 85],
            [295, 180],
        ]
    )

    image = np.array(Image.open("toy.jpg"))

    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(affine)]

    similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
    aug2 = np.ones_like(image)
    aug2[vx, vy] = image[tuple(similar)]

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug3 = np.ones_like(image)
    aug3[vx, vy] = image[tuple(rigid)]

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(aug1)
    ax[1].set_title("Affine Deformation")
    ax[2].imshow(aug2)
    ax[2].set_title("Similarity Deformation")
    ax[3].imshow(aug3)
    ax[3].set_title("Rigid Deformation")

    for x in ax.flat:
        x.axis("off")

    plt.tight_layout(w_pad=0.1)
    plt.show()


def demo2():
    """Smiled Monalisa"""
    np.random.seed(1234)

    image = np.array(Image.open("monalisa.jpg"))
    height, width, _ = image.shape

    # Define deformation grid
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    # ================ Control points group 1 (manually specified) ==================
    p1 = np.array(
        [
            [0, 0],
            [517, 0],
            [0, 798],
            [517, 798],
            [140, 186],
            [135, 295],
            [181, 208],
            [181, 261],
            [203, 184],
            [202, 304],
            [225, 213],
            [225, 243],
            [244, 211],
            [244, 253],
            [254, 195],
            [281, 232],
            [252, 285],
        ]
    )
    q1 = np.array(
        [
            [0, 0],
            [517, 0],
            [0, 798],
            [517, 798],
            [140, 186],
            [135, 295],
            [181, 208],
            [181, 261],
            [203, 184],
            [202, 304],
            [225, 213],
            [225, 243],
            [238, 207],
            [237, 261],
            [253, 199],
            [281, 232],
            [249, 279],
        ]
    )

    rigid1 = mls_rigid_deformation(vy, vx, p1, q1, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(rigid1)]

    # ====================== Control points group 1 (random) =======================
    p2 = np.stack(
        (
            np.random.randint(0, height, size=13),
            np.random.randint(0, width, size=13),
        ),
        axis=1,
    )
    q2 = p2 + np.random.randint(-20, 20, size=p2.shape)

    rigid2 = mls_rigid_deformation(vy, vx, p2, q2, alpha=1)
    aug2 = np.ones_like(image)
    aug2[vx, vy] = image[tuple(rigid2)]

    fig, ax = plt.subplots(1, 3, figsize=(13, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(aug1)
    ax[1].set_title("Manually specified control points")
    ax[2].imshow(aug2)
    ax[2].set_title("Random control points")

    for x in ax.flat:
        x.axis("off")

    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


def read_tif(frame):
    image_pil = Image.open("images/train-volume.tif")
    image_pil.seek(frame)
    image = np.array(image_pil)
    label_pil = Image.open("images/train-labels.tif")
    label_pil.seek(frame)
    label = np.array(label_pil)

    return image, label


def demo3():
    image, label = read_tif(1)
    image = np.pad(image, ((30, 30), (30, 30)), mode="symmetric")
    label = np.pad(label, ((30, 30), (30, 30)), mode="symmetric")

    height, width = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    def augment(p, q, mode="affine"):
        if mode.lower() == "affine":
            transform = mls_affine_deformation(vy, vx, p, q, alpha=1)
        elif mode.lower() == "similar":
            transform = mls_similarity_deformation(vy, vx, p, q, alpha=1)
        elif mode.lower() == "rigid":
            transform = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        else:
            raise ValueError

        aug_img = np.ones_like(image)
        aug_img[vx, vy] = image[tuple(transform)]
        aug_lab = np.ones_like(label)
        aug_lab[vx, vy] = label[tuple(transform)]

        return aug_img, aug_lab

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[1, 0].imshow(label, cmap="gray")
    ax[1, 0].set_title("Original Label")

    np.random.seed(1234)
    p = np.c_[
        np.random.randint(0, height, size=32), np.random.randint(0, width, size=32)
    ]
    q = p + np.random.randint(-15, 15, size=p.shape)
    q[:, 0] = np.clip(q[:, 0], 0, height)
    q[:, 1] = np.clip(q[:, 1], 0, width)
    p = np.r_[
        p, np.array([[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]])
    ]  # fix corner points
    q = np.r_[
        q, np.array([[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]])
    ]  # fix corner points

    for i, mode in enumerate(["Affine", "Similar", "Rigid"]):
        aug_img, aug_lab = augment(p, q, mode)
        ax[0, i + 1].imshow(aug_img, cmap="gray")
        ax[0, i + 1].set_title(f"{mode} Deformated Image")
        ax[1, i + 1].imshow(aug_lab, cmap="gray")
        ax[1, i + 1].set_title(f"{mode} Deformated Label")

    for x in ax.flat:
        x.axis("off")

    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


def benchmark_numpy(image, p, q):
    height, width = image.shape[:2]

    # Define deformation grid
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug = np.ones_like(image)
    aug[vx, vy] = image[tuple(rigid)]
    return aug


def run_benckmark(i):
    sizes = [  # (height, width)
        (100, 100),
        (500, 500),
        (500, 500),
        (500, 500),
        (1000, 1000),
        (2000, 2000),
    ]
    num_pts = [16, 16, 32, 64, 64, 64]

    times = []
    for _ in range(3):
        image = np.random.randint(0, 256, sizes[i])
        height, width = image.shape[:2]
        p = np.stack(
            (
                np.random.randint(0, height, size=num_pts[i]),
                np.random.randint(0, width, size=num_pts[i]),
            ),
            axis=1,
        )
        q = p + np.random.randint(-20, 20, size=p.shape)

        start = time.time()
        _ = benchmark_numpy(image, p, q)
        elapse = time.time() - start
        times.append(elapse)
    print("Time (numpy):", sum(times) / len(times))


class ControlPoints:
    def __init__(self):
        self.points = []
        self.original_points = []
        self.selected_point = None
        self.dragging = False
        self.dragged_points = set()
        self.insert_index = 0
        self.mouse_position = (0, 0)

    def dragging_point(self, point):
        x, y = point
        px, py = self.pending_point
        threshold = 5  # Adjust the threshold if necessary
        return abs(x - px) > threshold or abs(y - py) > threshold

    def add_point(self, point):
        self.points.append(point)
        self.original_points.append(point)

    def remove_selected_point(self):
        if self.selected_point is not None:
            self.points.pop(self.selected_point)
            self.original_points.pop(self.selected_point)
            self.dragged_points.discard(self.selected_point)
            self.selected_point = None

    def select_point(self, point, max_distance=10):
        for i, p in enumerate(self.points):
            if np.linalg.norm(np.array(p) - np.array(point)) <= max_distance:
                self.selected_point = i
                return True
        return False

    def unselect_point(self):
        self.selected_point = None

    def update_point(self, point):
        if self.selected_point is not None:
            self.points[self.selected_point] = point
            self.dragged_points.add(self.selected_point)

    def deselect_point(self):
        self.selected_point = None

    def is_selected(self, point):
        return (
            self.selected_point is not None
            and self.points[self.selected_point] == point
        )

    def has_been_dragged(self, index):
        return index in self.dragged_points

    def get_original_point(self, index):
        return self.original_points[index]

    # debug function
    def insert_point(self, x, y):
        self.points.append((x, y))
        self.original_points.append((x, y))


def draw_image_with_points(image, control_points):
    def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((1 - r) * pt1[0] + r * pt2[0])
            y = int((1 - r) * pt1[1] + r * pt2[1])
            pts.append((x, y))
        for p in pts[::2]:
            cv2.circle(img, p, thickness, color, -1)

    new_image = image.copy()
    for idx, point in enumerate(control_points.points):
        color = (0, 0, 255)  # Red for non-dragged points
        if idx in control_points.dragged_points:
            color = (128, 0, 128)  # Dark purple for dragged points
            # Draw dotted line between initial and current location
            draw_dotted_line(
                new_image, control_points.original_points[idx], point, (0, 0, 0), 1, 5
            )

        if idx == control_points.selected_point:
            color = (255, 0, 0)  # Blue for selected point

        cv2.circle(new_image, point, 5, color, -1)

    x, y = control_points.mouse_position
    cv2.putText(
        new_image,
        f"({x}, {y})",
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    return new_image


def mouse_callback(event, x, y, flags, param):
    image, control_points = param

    control_points.mouse_position = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        point_selected = control_points.select_point((x, y))
        if point_selected:
            if control_points.selected_point is not None:
                original_point = control_points.get_original_point(
                    control_points.selected_point
                )
                current_point = control_points.points[control_points.selected_point]
                print(
                    f"Original point: {original_point} -- Current point: {current_point}"
                )
        else:
            control_points.unselect_point()  # Unselect the current point if clicked outside the selection zone
        control_points.dragging = True
        control_points.pending_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if not control_points.select_point((x, y)):
            if (
                control_points.pending_point is not None
                and not control_points.dragging_point((x, y))
            ):
                control_points.add_point((x, y))
                control_points.select_point((x, y))
                control_points.pending_point = None
        control_points.dragging = False

    elif event == cv2.EVENT_MOUSEMOVE and control_points.dragging:
        if control_points.selected_point is not None:
            control_points.update_point((x, y))

    updated_image = draw_image_with_points(image, control_points)
    cv2.imshow(window_name, updated_image)


def clear_points(control_points):
    control_points.points = []
    control_points.original_points = []
    control_points.dragged_points = set()
    control_points.selected_point = None


def key_callback(image, control_points, file_path):
    key = cv2.waitKey(1) & 0xFF

    insert_points_list = [[209, 236], [258, 218], [221, 336], [283, 321]]

    if key == ord("q") or key == 27:  # 'q' or ESC key to quit
        return False

    if key == ord("d"):  # 'd' key to delete the selected point
        control_points.remove_selected_point()

    if key == ord("c"):  # 'c' key to clear all control points
        clear_points(control_points)

    if key == ord("a"):  # 'a' key to create an affine deformation
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Deformation")

        control_points_list = [[point[1], point[0]] for point in control_points.points]
        original_points_list = [
            [point[1], point[0]] for point in control_points.original_points
        ]

        demo(file_path, original_points_list, control_points_list, "affine")

    if key == ord("s"):  # 's' key to create a similarity deformation
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Deformation")
        control_points_list = [[point[1], point[0]] for point in control_points.points]
        original_points_list = [
            [point[1], point[0]] for point in control_points.original_points
        ]

        demo(file_path, original_points_list, control_points_list, "similarity")

    if key == ord("r"):  # 'r' key to create a rigid deformation
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Deformation")
        control_points_list = [[point[1], point[0]] for point in control_points.points]
        original_points_list = [
            [point[1], point[0]] for point in control_points.original_points
        ]

        demo(file_path, original_points_list, control_points_list, "rigid")

    if key == ord("i"):  # 'i' key to insert the next point from the list
        if control_points.insert_index < len(insert_points_list):
            x, y = insert_points_list[control_points.insert_index]
            control_points.insert_point(x, y)
            control_points.insert_index += 1

    if key == ord("w"):  # '1' key to save the displayed image in "Deformation" window
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            global deformation_output
            current_path = os.path.dirname(os.path.realpath(__file__))
            image_folder = os.path.join(current_path, "images")

            os.makedirs(image_folder, exist_ok=True)

            filenames = os.listdir(image_folder)
            deform_numbers = [
                int(re.findall(r"\d+", f)[0])
                for f in filenames
                if f.startswith("deform_") and f.endswith(".jpg")
            ]

            if not deform_numbers:
                last_deform_number = 0
            else:
                last_deform_number = max(deform_numbers)

            new_deform_number = last_deform_number + 1
            new_filename = f"deform_{new_deform_number}.jpg"

            # Save the image with the new filename
            new_filepath = os.path.join(image_folder, new_filename)
            cv2.imwrite(new_filepath, deformation_output)
            print(f"Image saved as {new_filepath}")

    updated_image = draw_image_with_points(image, control_points)
    cv2.imshow(window_name, updated_image)

    return True


def demo(image, p, q, mode):
    global deformation_output

    p = np.array(p)
    q = np.array(q)

    image = np.array(Image.open(image))

    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    if mode == "affine":
        affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(affine)]
    elif mode == "similarity":
        similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(similar)]
    elif mode == "rigid":
        rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(rigid)]

    deformation_output = cv2.cvtColor(aug1, cv2.COLOR_RGB2BGR)
    cv2.imshow("Deformation", deformation_output)


# if __name__ == "__main__":
#     demo()
#     # demo2()
#     # demo3()
#     # demo_torch()


#     # run_benckmark(i=0)
def get_landmarks():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import time
    from pathlib import Path
    import gsoup

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

    i = 1
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
        [[landmark.x, landmark.y] for landmark in detection_result.hand_landmarks[0]]
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
    return p, q


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


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Usage: python live_demo.py <image_path>")
        print("\nHotkeys:")
        print("  q or ESC - Quit ")
        print("  d - Delete the selected control point")
        print("  c - Clear all control points")
        print("  a - Create an affine deformation and display it in a separate window")
        print(
            "  s - Create a similarity deformation and display it in a separate window"
        )
        print("  r - Create a rigid deformation and display it in a separate window")
        print("  w - Write the last deformation to the images folder")
    else:
        file_path = sys.argv[1]
        image = cv2.imread(file_path)

        if image is None:
            print(f"Error: Could not load image from '{file_path}'.")
            sys.exit(1)

        window_name = "Moving Least Squares Demo"
        control_points = ControlPoints()
        p, q = get_landmarks()
        for i in range(len(p)):
            control_points.insert_point(p[i][0], p[i][1])
            assert control_points.select_point(p[i])
            control_points.update_point(q[i])
            control_points.deselect_point()
        height, width, _ = image.shape
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO
        )

        cv2.setMouseCallback(window_name, mouse_callback, (image, control_points))
        cv2.imshow(window_name, image)

        while key_callback(image, control_points, file_path):
            pass

        cv2.destroyAllWindows()
