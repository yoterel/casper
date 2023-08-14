import numpy as np
import gsoup
from pathlib import Path
import basler
import dynaflash
import leap
import cv2
import time


def pix2pix(root_path):
    output_path = Path(root_path, "debug", "pix2pix")
    resource_path = Path(root_path, "resource")
    test_pattern = gsoup.load_image(
        Path(resource_path, "XGA_colorbars_hor.png"))
    proj_wh = (1024, 768)
    gray = gsoup.GrayCode()
    patterns = gray.encode(proj_wh)
    proj = dynaflash.projector()
    proj.init()
    proj.project(test_pattern)
    proj.project_white()

    # gray_code_pattern = np.repeat(patterns[6], 3, axis=-1)
    # proj.project(gray_code_pattern)
    # proj.kill()

    cam = basler.camera()
    cam.init(12682.0)
    cam.balance_white()
    for i in range(100):
        cam.capture()  # warmup
    captures = []
    for i, pattern in enumerate(patterns):
        print("pattern", i)
        pattern = np.repeat(pattern, 3, axis=-1)
        proj.project(pattern)
        image = cam.capture()
        captures.append(image)
    proj.kill()
    cam.kill()
    captures = np.array(captures)
    gsoup.save_images(captures, Path(output_path, "captures"))
    forward, fg = gray.decode(
        captures, proj_wh, output_dir=output_path, debug=True)
    composed = forward * fg[..., None]
    composed_normalized = (
        composed / np.array([proj_wh[1], proj_wh[0]])).astype(np.float32)
    composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
    # composed_normalized = np.concatenate((composed_normalized, np.zeros_like(composed_normalized[..., :1])), axis=-1)
    gsoup.save_image(composed_normalized[..., 0:1], Path(
        output_path, "texturex.tiff"), extension="tiff")
    gsoup.save_image(composed_normalized[..., 1:2], Path(
        output_path, "texturey.tiff"), extension="tiff")


def acq_procam(root_path):
    # resource_path = Path(root_path, "resource")
    dst_path = Path(root_path, "debug", "calibration")
    proj_wh = (1024, 768)
    gray = gsoup.GrayCode()
    patterns = gray.encode(proj_wh)
    proj = dynaflash.projector()
    proj.init()
    proj.project_white()
    cam = basler.camera()
    cam.init(12682.0)
    cam.balance_white()
    for i in range(100):
        cam.capture()  # warmup
    cur_session = 0
    sessions = sorted([int(x.name) for x in Path(dst_path).glob("*")])
    if len(sessions) > 0:
        cur_session = sessions[-1] + 1
    while True:
        # display current image
        while (True):
            frame = cam.capture()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        text = input(
            "current # of sessions: {}, continue?".format(cur_session))
        if text == "q" or text == "n" or text == "no":
            break
        captures = []
        for i, pattern in enumerate(patterns):
            print("pattern", i)
            pattern = np.repeat(pattern, 3, axis=-1)
            proj.project(pattern)
            image = cam.capture()
            captures.append(image)
        captures = np.array(captures)
        gsoup.save_images(captures, Path(
            dst_path, "{:02d}".format(cur_session)))
        proj.project_white()
        cur_session += 1
    proj.kill()
    cam.kill()


def calibrate_procam(root_path, force_calib=False):
    proj_wh = (1024, 768)
    dst_path = Path(root_path, "debug", "calibration")
    if Path(dst_path, "calibration.npz").exists() and not force_calib:
        res = np.load(Path(dst_path, "calibration.npz"))
        res = {k: res[k] for k in res.keys()}
    else:
        res = gsoup.calibrate_procam(proj_wh, dst_path, 10, 7, 10, 2.0, "lower_half",
                                     verbose=True, output_dir=Path(root_path, "debug", "calib_debug"), debug=True)

        np.savez(Path(dst_path, "calibration.npz"), **res)
    test_pattern = gsoup.generate_checkerboard(proj_wh[1], proj_wh[0], 20)
    test_pattern = np.repeat(test_pattern, 3, axis=-1).astype(np.uint8)*255
    proj = dynaflash.projector()
    proj.init()
    proj.project_white()
    cam = basler.camera()
    cam.init(12682.0)
    cam.balance_white()
    for i in range(100):
        cam.capture()  # warmup
    proj.project(test_pattern)
    image = cam.capture()
    gsoup.save_image(image, Path(Path(root_path, "debug"), "orig.png"))
    # cam_wh = image.shape[:2][::-1]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(res["cam_intrinsics"], res["cam_distortion"], cam_wh, 1, (w,h))
    undistorted = cv2.undistort(
        image, res["cam_intrinsics"], res["cam_distortion"], None, None)
    gsoup.save_image(undistorted, Path(
        Path(root_path, "debug"), "undistorted.png"))
    cam.kill()
    proj.kill()


def auto_trigger_leap(leapmotion):
    state_counter = 0
    cur_tip = np.array(leapmotion.get_index_tip())
    tips = []
    while True:
        time.sleep(0.01)
        new_tip = np.array(leapmotion.get_index_tip())
        if len(cur_tip) != 0 and len(new_tip) != 0:
            if np.linalg.norm(new_tip - cur_tip) < 10.0:
                print("\r triggering..{}%".format(
                    state_counter), end='')
                state_counter += 1
                tips.append(new_tip)
                if state_counter > 100:
                    print("triggered")
                    return cur_tip, tips
            else:
                state_counter = 0
                cur_tip = np.array(leapmotion.get_index_tip())
                tips = []
        else:
            state_counter = 0
            cur_tip = np.array(leapmotion.get_index_tip())
            tips = []


def acq_leap_projector(root_path):
    proj_wh = (1024, 768)
    dst_path = Path(root_path, "debug", "leap_calibration")
    dst_path.mkdir(parents=True, exist_ok=True)
    test_pattern = gsoup.generate_checkerboard(proj_wh[1], proj_wh[0], 20)
    test_pattern = np.repeat(test_pattern, 3, axis=-1).astype(np.uint8)*255
    proj = dynaflash.projector()
    proj.init()
    proj.project(test_pattern)
    leapmotion = leap.device()
    tip_locations = []
    session = 0
    while True:
        text = input(
            "current # of sessions: {}, continue?".format(session))
        if text == "q" or text == "n" or text == "no":
            break
        new_loc_x = np.random.randint(0, 1024, size=(1,))
        new_loc_y = np.random.randint(0, 768, size=(1,))
        image = np.zeros((768, 1024, 3), dtype=np.uint8)
        image[:, new_loc_x, :] = 255
        image[new_loc_y, :, :] = 255
        proj.project(image)
        tmp_locs = []
        _, tips = auto_trigger_leap(leapmotion)
        avg_tip = np.mean(tips, axis=0)
        tip_locations.append(avg_tip)
        session += 1
    tip_locations = np.array(tip_locations)
    np.save(Path(dst_path, "calibration_data.npy"), tip_locations)
    proj.kill()
    leapmotion.kill()


def calibrate_leap_projector(root_path, force_calib=False):
    proj_wh = (1024, 768)
    dst_path = Path(root_path, "debug", "leap_calibration")
    if Path(dst_path, "leap_calibration.npz").exists() and not force_calib:
        res = np.load(Path(dst_path, "calibration_results.npz"))
        res = {k: res[k] for k in res.keys()}
    else:
        tip_locations = np.load(Path(dst_path, "calibration_data.npy"))
        image_locations = None
        procam_calib_res = np.load(
            Path(root_path, "debug", "calibration", "calibration.npz"))
        procam_calib_res = {k: procam_calib_res[k]
                            for k in procam_calib_res.keys()}
        res = cv2.solvePnP(tip_locations, image_locations,
                           procam_calib_res["proj_intrinsics"], procam_calib_res["proj_dist"])
        np.savez(Path(dst_path, "calibration_results.npz"), **res)


if __name__ == "__main__":
    root_path = Path("C:/src/augmented_hands")
    # pix2pix(root_path)
    # acq_procam(root_path)
    # calibrate_procam(root_path)
    acq_leap_projector(root_path)
    calibrate_leap_projector(root_path)
