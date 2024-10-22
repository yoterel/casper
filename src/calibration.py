import numpy as np
import gsoup
from pathlib import Path
import dynaflash
import leap
import basler
import cv2
import time
from PIL import Image, ImageDraw
from gsoup.viewer_drivers import calibration_static_view, pcs_slide_view


def create_circle_image(width, height, loc, radii=10):
    img = Image.new("RGB", (width, height), "black")
    img1 = ImageDraw.Draw(img)
    img1.ellipse(
        [loc[0] - radii, loc[1] - radii, loc[0] + radii, loc[1] + radii],
        fill="blue",
        outline="blue",
    )
    return np.array(img)


def reconstruct(root_path):
    calib_res_path = Path(root_path, "debug", "calibration")
    res = np.load(Path(calib_res_path, "calibration.npz"))
    calib_res = {k: res[k] for k in res.keys()}
    cam_int, cam_dist, proj_int, proj_dist, proj_transform = (
        calib_res["cam_intrinsics"],
        calib_res["cam_distortion"],
        calib_res["proj_intrinsics"],
        calib_res["proj_distortion"],
        calib_res["proj_transform"],
    )
    leap_calib_res_path = Path(root_path, "debug", "leap_calibration")
    world2projector = np.load(Path(leap_calib_res_path, "w2p.npy"))
    c2w = np.linalg.inv(world2projector) @ proj_transform
    mode = "ij"
    output_path = Path(root_path, "debug", "reconstruct")
    resource_path = Path(root_path, "resource")
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
        captures, proj_wh, output_dir=Path(output_path, "decode"), debug=True, mode=mode
    )
    cam_transform = c2w  # np.eye(4)
    # cam_transform @ np.linalg.inv(proj_transform)
    proj_transform = np.linalg.inv(world2projector)
    pc = gsoup.reconstruct_pointcloud(
        forward,
        fg,
        cam_transform,
        proj_transform,
        cam_int,
        cam_dist,
        proj_int,
        mode=mode,
        color_image=captures[-2],
    )
    gsoup.save_pointcloud(
        pc[:, :3],
        Path(output_path, "points.ply"),
        vertex_colors=pc[:, 3:].astype(np.uint8),
    )


def pix2pix(root_path, find_homography=False, coaxial_setup=True):
    output_path = Path(root_path, "debug", "pix2pix")
    resource_path = Path(root_path, "resource")
    proj_wh = (1024, 768)
    cam_wh = (720, 540)
    # test_pattern = gsoup.load_image(
    #     Path(resource_path, "XGA_colorbars_hor.png"))
    test_pattern = gsoup.generate_checkerboard(proj_wh[1], proj_wh[0], 20)
    test_pattern = np.repeat(test_pattern, 3, axis=-1).astype(np.uint8) * 255
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
    cam.init(2500.0)
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
    if coaxial_setup:  # flip the captures as there is a mirror in the setup
        captures = captures[:, :, ::-1, :]
    gsoup.save_images(captures, Path(output_path, "captures"))
    forward, fg = gray.decode(captures, proj_wh, output_dir=output_path, debug=True)
    composed = forward * fg[..., None]
    composed_normalized = (composed / np.array([proj_wh[1], proj_wh[0]])).astype(
        np.float32
    )
    composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
    # composed_normalized = np.concatenate((composed_normalized, np.zeros_like(composed_normalized[..., :1])), axis=-1)
    gsoup.save_image(
        composed_normalized[..., 0:1],
        Path(output_path, "texturex.tiff"),
        extension="tiff",
    )
    gsoup.save_image(
        composed_normalized[..., 1:2],
        Path(output_path, "texturey.tiff"),
        extension="tiff",
    )
    if find_homography:
        src_points = np.stack(
            np.meshgrid(np.arange(cam_wh[1]), np.arange(cam_wh[0]), indexing="ij"),
            axis=-1,
        )
        mask = np.zeros_like(src_points[..., 0], dtype=bool)
        mask[102:500, 183:437] = True  # filter some points from image
        src_points = src_points[mask]
        src_points = src_points.reshape(-1, 2)
        src_points = src_points / (np.array(cam_wh) - 1)[::-1]
        src_points = src_points * 2 - 1
        # mask = fg
        # src_points = np.argwhere(fg) / (np.array(cam_wh) - 1)[::-1]
        dst_points = forward[mask] / (np.array(proj_wh) - 1)[::-1]
        dst_points = dst_points * 2 - 1
        h_mat, inliers = cv2.findHomography(
            np.array(src_points),
            np.array(dst_points),
            cv2.RANSAC,
        )
        my_points = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        hom_points = gsoup.to_hom(my_points)
        transformed_points = h_mat @ hom_points.transpose()
        gsoup.homogenize(transformed_points.T)
        print(transformed_points)


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
        while True:
            frame = cam.capture()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        text = input("current # of sessions: {}, continue?".format(cur_session))
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
        gsoup.save_images(captures, Path(dst_path, "{:02d}".format(cur_session)))
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
        res = gsoup.calibrate_procam(
            proj_wh,
            dst_path,
            10,
            7,
            10,
            20.0,
            "lower_half",
            verbose=True,
            output_dir=Path(root_path, "debug", "calib_debug"),
            debug=True,
        )

        np.savez(Path(dst_path, "calibration.npz"), **res)
    test_pattern = gsoup.generate_checkerboard(proj_wh[1], proj_wh[0], 20)
    test_pattern = np.repeat(test_pattern, 3, axis=-1).astype(np.uint8) * 255
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
        image, res["cam_intrinsics"], res["cam_distortion"], None, None
    )
    gsoup.save_image(undistorted, Path(Path(root_path, "debug"), "undistorted.png"))
    cam.kill()
    proj.kill()


def acq_cam(root_path):
    CHECKERBOARD_WIDTH = 10
    CHECKERBOARD_HEIGHT = 7
    # resource_path = Path(root_path, "resource")
    dst_path = Path(root_path, "resource", "calibrations", "cam_calibration")
    dst_path.mkdir(parents=True, exist_ok=True)
    cam = basler.camera()
    cam.init(10000.0)
    cam.balance_white()
    for i in range(100):
        cam.capture()  # warmup
    cur_session = 0
    sessions = sorted([int(x.stem) for x in Path(dst_path).glob("*")])
    if len(sessions) > 0:
        cur_session = sessions[-1] + 1
    while True:
        text = input("current # of sessions: {}, continue?".format(cur_session))
        if text == "q" or text == "n" or text == "no":
            break
        # display current image
        while True:
            frame = cam.capture()
            display_frame = frame.copy()
            ret, corners = cv2.findChessboardCorners(
                display_frame,
                (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT),
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            cv2.drawChessboardCorners(
                display_frame, (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT), corners, ret
            )
            cv2.imshow("frame", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if ret:
            gsoup.save_image(frame, Path(dst_path, "{:02d}".format(cur_session)))
        cur_session += 1
    cam.kill()


def calibrate_cam(root_path, force_calib=False):
    CHECKERBOARD_WIDTH = 10
    CHECKERBOARD_HEIGHT = 7
    pattern_size = (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT)
    chess_block_size = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= chess_block_size
    dst_path = Path(root_path, "resource", "calibrations", "cam_calibration")
    if Path(dst_path, "cam_calibration.npz").exists() and not force_calib:
        res = np.load(Path(dst_path, "cam_calibration.npz"))
        res = {k: res[k] for k in res.keys()}
    else:
        images = gsoup.load_images(Path(dst_path), as_grayscale=True)
        imgpoints = []
        objpoints = []
        for image in images:
            ret, corners = cv2.findChessboardCorners(
                image,
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ret:
                corners2 = cv2.cornerSubPix(
                    image, corners, (11, 11), (-1, -1), criteria
                )
                imgpoints.append(corners2)
                objpoints.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            images[0].shape[::-1],
            None,
            None,
            None,
            None,
            cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT,
        )
        res = {"cam_intrinsics": mtx, "cam_distortion": dist}
        np.savez(Path(dst_path, "cam_calibration.npz"), **res)
        errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = np.linalg.norm(imgpoints[i] - imgpoints2, axis=-1)
            # error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(error)
        count, values = np.histogram(errors, bins=10)
        print("counts: ", count)
        print("errors: ", values)
        print("average error (in pixel units): {}".format(np.mean(errors)))
    cam = basler.camera()
    cam.init(12682.0)
    cam.balance_white()
    for i in range(100):
        cam.capture()  # warmup
    while True:
        orig_frame = cam.capture()
        frame = orig_frame.copy()
        ret, corners = cv2.findChessboardCorners(
            frame,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        if ret:
            # print("objp:", objp.shape)
            # print("corners:", corners.shape)
            ret, rvec, tvec = cv2.solvePnP(
                objp, corners, res["cam_intrinsics"], res["cam_distortion"]
            )
            if ret:
                frame = cv2.drawFrameAxes(
                    frame,
                    res["cam_intrinsics"],
                    res["cam_distortion"],
                    rvec,
                    tvec,
                    chess_block_size,
                    3,
                )
        cv2.imshow("frame: ", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.kill()


def auto_trigger_leap(leapmotion, proj=None, loc2d=None, threshold=100):
    """
    auto triggers the leap if the tip of the hand is steady enough for a certain amount of time
    :param leapmotion: leap device
    :param threshold: every 10 ms holidng the tip steady will increase the counter by 1, if the counter reaches threshold, the function will return
    :return: the tip location used for asserting no movement, and the list of tip locations acquired during countdown
    """
    state_counter = 0
    cur_tip = np.array(leapmotion.get_index_tip())
    if proj is not None and loc2d is not None:
        orig_image = np.zeros((768, 1024, 3), dtype=np.uint8)
        orig_image[:, loc2d[0], :] = 255
        orig_image[loc2d[1], :, :] = 255
    tips = []
    while True:
        time.sleep(0.01)
        new_tip = np.array(leapmotion.get_index_tip())
        if len(cur_tip) != 0 and len(new_tip) != 0:
            if np.linalg.norm(new_tip - cur_tip) < 10.0:
                print("\r triggering..{}%".format(state_counter), end="")
                if proj is not None:
                    image = create_circle_image(1024, 768, loc2d, 100 - state_counter)
                    image[:, loc2d[0], :] = 255
                    image[loc2d[1], :, :] = 255
                    proj.project(image)
                state_counter += 1
                tips.append(new_tip)
                if state_counter > threshold:
                    print("triggered")
                    return cur_tip, tips
            else:
                state_counter = 0
                cur_tip = np.array(leapmotion.get_index_tip())
                if proj is not None:
                    proj.project(orig_image)
                tips = []
        else:
            state_counter = 0
            cur_tip = np.array(leapmotion.get_index_tip())
            if proj is not None:
                proj.project(orig_image)
            tips = []


def acq_leap_projector(root_path, n_sessions=10, manual=False):
    proj_wh = (1024, 768)
    dst_path = Path(root_path, "debug", "leap_calibration")
    dst_path.mkdir(parents=True, exist_ok=True)
    test_pattern = gsoup.generate_checkerboard(proj_wh[1], proj_wh[0], 20)
    test_pattern = np.repeat(test_pattern, 3, axis=-1).astype(np.uint8) * 255
    proj = dynaflash.projector()
    proj.init()
    proj.project(test_pattern)
    leapmotion = leap.device()
    tip_locations = []
    tip_locations_2d = []
    session = 0
    while session < n_sessions:
        # text = input(
        #     "current # of sessions: {}, continue?".format(session))
        # if text == "q" or text == "n" or text == "no":
        #     break
        # new_loc_x = np.random.randint(0, 1024, size=(1,))
        # new_loc_y = np.random.randint(0, 768, size=(1,))
        if manual:
            loc_2d = (np.array(proj_wh) / 2).astype(np.int32)
            _, tips = auto_trigger_leap(leapmotion, proj, loc_2d)
        else:
            new_loc_x = np.random.randint(200, 824, size=(1,))
            new_loc_y = np.random.randint(300, 668, size=(1,))
            loc_2d = np.array([new_loc_x, new_loc_y])
            image = np.zeros((768, 1024, 3), dtype=np.uint8)
            image[:, new_loc_x, :] = 255
            image[new_loc_y, :, :] = 255
            proj.project(image)
            _, tips = auto_trigger_leap(leapmotion, proj, loc_2d)
        avg_tip = np.mean(tips, axis=0)
        tip_locations.append(avg_tip)
        tip_locations_2d.append(loc_2d)
        session += 1
    tip_locations = np.array(tip_locations)
    tip_locations_2d = np.array(tip_locations_2d).astype(np.float32)
    if not manual:
        np.savez(
            Path(dst_path, "calibration_data.npz"),
            tip_locations=tip_locations,
            tip_locations_2d=tip_locations_2d,
        )
    else:
        np.savez(
            Path(dst_path, "calibration_data_manual.npz"),
            tip_locations=tip_locations,
            tip_locations_2d=tip_locations_2d,
        )
    proj.kill()
    leapmotion.kill()


def calibrate_leap_projector(root_path, force_calib=False):
    proj_wh = (1024, 768)
    dst_path = Path(root_path, "debug", "leap_calibration")
    procam_calib_res = np.load(
        Path(root_path, "debug", "calibration", "calibration.npz")
    )
    procam_calib_res = {k: procam_calib_res[k] for k in procam_calib_res.keys()}
    if Path(dst_path, "w2p.npy").exists() and not force_calib:
        world2projector = np.load(Path(dst_path, "w2p.npy"))
    else:
        res = np.load(Path(dst_path, "calibration_data.npz"))
        res = {k: res[k] for k in res.keys()}
        image_locations = res["tip_locations_2d"].squeeze().astype(np.float32)
        # success, rotation_vector, translation_vector = cv2.solvePnP(res["tip_locations"], image_locations,
        #                                                             procam_calib_res["proj_intrinsics"], procam_calib_res["proj_distortion"])
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            res["tip_locations"],
            image_locations,
            procam_calib_res["proj_intrinsics"],
            procam_calib_res["proj_distortion"],
        )
        rot_mat, _ = cv2.Rodrigues(rotation_vector)
        translation_vector = translation_vector.squeeze()
        world2projector = gsoup.compose_rt(
            rot_mat[None, ...], translation_vector[None, ...], square=True
        )[0]

        np.save(Path(dst_path, "w2p.npy"), world2projector)

    rot_vec, _ = cv2.Rodrigues(world2projector[:3, :3])
    rot = rot_vec
    trans = world2projector[:3, -1]
    leapmotion = leap.device()
    proj = dynaflash.projector()
    proj.init()
    counter = 0
    while counter < 1000:
        _, tips = auto_trigger_leap(leapmotion, threshold=1)
        counter += 1
        avg_tip = np.mean(tips, axis=0)

        tip_2d, _ = cv2.projectPoints(
            avg_tip,
            rot,
            trans,
            procam_calib_res["proj_intrinsics"],
            procam_calib_res["proj_distortion"],
        )
        tip_2d = tip_2d[0, 0, :].round()
        if 0 <= tip_2d[0] < 1024 and 0 <= tip_2d[1] < 768:
            image = np.zeros((768, 1024, 3), dtype=np.uint8)
            image[:, int(tip_2d[0]), :] = 255
            image[int(tip_2d[1]), :, :] = 255
            proj.project(image)
    proj.kill()
    leapmotion.kill()


if __name__ == "__main__":
    root_path = Path("C:/src/casper")
    pix2pix(root_path, True)
    # acq_cam(root_path)
    # calibrate_cam(root_path)
    # acq_procam(root_path)
    # calibrate_procam(root_path)
    # acq_leap_projector(root_path, 20, False)
    # calibrate_leap_projector(root_path, True)

    # dst_path = Path(root_path, "debug", "leap_calibration")
    # res = np.load(Path(dst_path, "calibration_data_manual.npz"))
    # res = {k: res[k] for k in res.keys()}
    # pcs_slide_view(res["tip_locations"][None, :, :])
    # reconstruct(root_path)

    # v, f = gsoup.load_mesh(
    #     "C:/src/augmented_hands/resource/GenericHand/GenericHand.obj")
    # v, f, vc = gsoup.load_mesh(
    #     "C:/src/augmented_hands/resource/reconst2.ply", return_vert_color=True)
    # v1, vc1 = gsoup.load_pointcloud(
    #     "C:/src/augmented_hands/debug/reconstruct/points.ply", return_vert_color=True)
    # calib_res_path = Path(root_path, "debug", "calibration")
    # res = np.load(Path(calib_res_path, "calibration.npz"))
    # calib_res = {k: res[k] for k in res.keys()}
    # cam_int, cam_dist,\
    #     proj_int, proj_dist,\
    #     proj_transform = calib_res["cam_intrinsics"], calib_res["cam_distortion"],\
    #     calib_res["proj_intrinsics"], calib_res["proj_distortion"],\
    #     calib_res["proj_transform"]
    # leap_calib_res_path = Path(root_path, "debug", "leap_calibration")
    # world2projector = np.load(Path(leap_calib_res_path, "w2p.npy"))
    # p2w = np.linalg.inv(world2projector)
    # c2w = p2w @ proj_transform
    # forward_map = np.load(
    #     "C:/src/augmented_hands/debug/reconstruct/decode/forward_map.npy")
    # fg = np.load("C:/src/augmented_hands/debug/reconstruct/decode/fg.npy")
    # camera_wh = (720, 540)
    # proj_wh = (1024, 768)
    # color_image = gsoup.load_image(
    #     "C:/src/augmented_hands/debug/reconstruct/captures/00040.png", float=True)[..., :3]
    # calibration_static_view(c2w, p2w, camera_wh, proj_wh,
    #                         cam_int, cam_dist, proj_int, forward_map, fg, color_image, mode="ij")
    # print("done")
