import gsoup
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator, splprep, splev
from scipy.optimize import fmin
from scipy.spatial.distance import euclidean
from scipy import ndimage
import cv2
from PIL import Image, ImageDraw
import csv
import matplotlib.pyplot as plt


def paint_skin():
    image = gsoup.load_image(
        Path("C:/src/augmented_hands/resource/left_hand_uvunwrapped.png"), float=True
    )
    human_skin_color = np.array([255 / 255, 219 / 255, 172 / 255])
    image[:, :, 0] = image[:, :, 0] * human_skin_color[0]
    image[:, :, 1] = image[:, :, 1] * human_skin_color[1]
    image[:, :, 2] = image[:, :, 2] * human_skin_color[2]
    gsoup.save_image(
        image, Path("C:/src/augmented_hands/resource/left_hand_uvunwrapped_skin.png")
    )


def do_vid():
    path = Path("C:/data/game")
    gsoup.images_to_video(path, "C:/data/game.mp4", 30)


def jnd_process_images(src_path, mask_path, dst_path):
    # JND image process
    # create mask
    mask = gsoup.load_image(mask_path)
    height, width = mask.shape[:2]
    # mask = mask[:, :, :3]  # remove alpha channel
    # mask = np.all(mask == np.array([0, 0, 255]), axis=-1)  # select blue channel
    # mask = ndimage.binary_dilation(mask).astype(mask.dtype)  # dilate abit
    # gsoup.save_image(mask, dst_path / "mask.png")

    # load images

    # path = Path("C:/Users/sens/Desktop/images/jnd_results/baseline")
    images, paths = gsoup.load_images(src_path, return_paths=True)
    per_subject = []
    per_subject_sum = []
    for i, image in enumerate(images):
        image = image[:, :, :3]  # remove alpha channel
        lit_mask = np.all(image == np.array([237, 28, 36]), axis=-1)
        # gsoup.save_image(lit_mask, dst_path / "scribble.png")
        nlabels, labels = cv2.connectedComponents(lit_mask.astype(np.uint8))
        img = Image.new("RGB", (width, height), "black")
        for label in range(1, nlabels):
            points = np.array(np.where(labels == label)).T
            if len(points) < 3:
                continue
            hull = ConvexHull(points)
            simplex_vertices = points[hull.vertices]
            simplex_vertices[:, [0, 1]] = simplex_vertices[:, [1, 0]]  # swap x and y
            img1 = ImageDraw.Draw(img)
            img1.polygon(list(map(tuple, simplex_vertices)), fill="white")
        # gsoup.save_image(np.array(img), dst_path / "hull.png")
        hull_image = np.array(img)[
            :, :, 0
        ]  # select 1st channel, they are all equivalent
        per_subject.append(hull_image)
        per_subject_sum.append(np.sum(hull_image > 0))

    # compute weighted sum over subjects per pixel
    full_image = np.zeros((height, width), dtype=np.float32)
    for i, image in enumerate(per_subject):
        full_image += per_subject[i] * (1 / per_subject_sum[i])
    denom = np.sum(1 / x for x in per_subject_sum)
    full_image = full_image / denom
    full_image = (full_image - full_image.min()) / (
        full_image.max() - full_image.min()
    )  # normalize
    # interpolate over all pixels (in mask)
    points = np.argwhere(full_image != 0)
    data = full_image[full_image != 0]
    interp = LinearNDInterpolator(points, data, fill_value=0.0)
    X, Y = np.meshgrid(np.arange(height), np.arange(width))
    interpolated = interp(X, Y).transpose(1, 0)
    # gsoup.save_image(interpolated, dst_path / "interp.png")
    # discretize
    final_image_gray = gsoup.to_8b(interpolated)
    # make red channel indicate signal
    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    final_image[:, :, 0] = final_image_gray
    # mask out hand and save
    final_image[mask == 0] = 255
    gsoup.save_image(final_image, dst_path / "final_image.png")


def jnd_plot(dst_path):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    baseline = np.array(
        [4.2255, 3.1405, 3.8625, 3.5545, 2.7795, 4.8805, 0.217, 0.926, 3.3805, 3.8665]
    )
    ours = np.array(
        [6.8555, 3.376, 4.413, 18.638, 2.973, 4.603, 4.196, 6.6755, 3.8855, 10.565]
    )
    # report mean and stds
    print(f"Baseline: {baseline.mean():.2f} ± {baseline.std():.2f}")
    print(f"Ours: {ours.mean():.2f} ± {ours.std():.2f}")
    session_times = np.array(
        [
            14.7656,
            12.589,
            16.2561,
            16.2402,
            17.2429,
            12.6386,
            11.0314,
            10.4284,
            26.5882,
            12.5154,
            19.7092,
            17.8119,
            9.69475,
            11.5,
            5.91688,
            7.75539,
            17.3458,
            22.8213,
            13.5791,
            15.354,
            32.1,
            12.3,
            13.56,
            9.61529,
            11.1667,
            12.0407,
            12.7014,
            14.5462,
            16.4917,
            20.418,
            12.9531,
            11.4878,
            8.31638,
            13.86,
            11.1048,
            11.5897,
            8.97708,
            9.8952,
            19.7321,
            13.5763,
        ]
    )
    print(f"Session times: {session_times.mean():.2f} ± {session_times.std():.2f}")
    # set bar plots
    n_subjects = len(baseline)
    ind = np.arange(n_subjects)
    width = 0.3
    plt.bar(ind, baseline, width, label="baseline", color="blue", alpha=0.8)
    plt.bar(ind + width, ours, width, label="ours", color="orange", alpha=0.8)
    plt.xticks(ind + width / 2, tuple(str(i) for i in range(1, n_subjects + 1)))
    # set means as horizontal lines
    plt.axhline(baseline.mean(), color="blue", linewidth=2, linestyle="--")
    plt.axhline(ours.mean(), color="orange", linewidth=2, linestyle="--")
    # Add arrows annotating the means:
    for dat, xoff in zip([baseline, ours], [15, 15]):
        mean_value = dat.mean()
        # plt.annotate(
        #     f"Mean: {mean_value:.2f}",
        #     xy=(1, mean_value),
        #     xytext=(1.02, mean_value),
        #     textcoords=("axes fraction", "data"),
        #     ha="left",
        #     va="center",
        #     color="black",
        # )
        align = "left" if xoff > 0 else "right"
        plt.annotate(
            "Mean: {:0.2f} ms".format(mean_value),
            xy=(1, mean_value),
            xytext=(15, xoff),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            horizontalalignment=align,
            verticalalignment="center",
            arrowprops=dict(
                arrowstyle="-|>",
                fc="black",
                shrinkA=0,
                shrinkB=0,
                connectionstyle="angle,angleA=90,angleB=0,rad=10",
            ),
        )
    # set x and y titles
    plt.legend()
    plt.xlabel("Subject ID")
    plt.ylabel("JND [ms]")
    plt.tight_layout()
    plt.savefig(str(dst_path / "jnd.pdf"))
    plt.cla()
    plt.clf()


def guesschar_plot(src_path, dst_path):
    sids = []
    methods = []
    palms = []
    scores = []
    accuracies = []
    q1s = []
    with open(src_path, mode="r") as f:
        csvFile = csv.reader(f)
        for i, line in enumerate(csvFile):
            if i == 0:
                continue
            sid, method, palm, score, accuracy, q1 = line
            if score != "":
                sids.append(sid)
                methods.append(method)
                palms.append(palm)
                scores.append(float(score))
                accuracies.append(float(accuracy))
                q1s.append(int(q1))

    baseline_mask = np.array(methods) == "wo"
    front_mask = np.array(palms) == "f"
    baseline_scores = np.array(scores)[baseline_mask]
    baseline_acc = np.array(accuracies)[baseline_mask]
    baseline_q1 = np.array(q1s)[baseline_mask]
    baseline_scores_front = np.array(scores)[baseline_mask & front_mask]
    baseline_scores_back = np.array(scores)[baseline_mask & ~front_mask]
    baseline_acc_front = np.array(accuracies)[baseline_mask & front_mask]
    baseline_acc_back = np.array(accuracies)[baseline_mask & ~front_mask]
    ours_scores = np.array(scores)[~baseline_mask]
    ours_acc = np.array(accuracies)[~baseline_mask]
    ours_q1 = np.array(q1s)[~baseline_mask]
    ours_scores_front = np.array(scores)[~baseline_mask & front_mask]
    ours_scores_back = np.array(scores)[~baseline_mask & ~front_mask]
    ours_acc_front = np.array(accuracies)[~baseline_mask & front_mask]
    ours_acc_back = np.array(accuracies)[~baseline_mask & ~front_mask]

    # report mean and stds
    print("baseline: {} ± {}".format(baseline_scores.mean(), baseline_scores.std()))
    print("baseline acc: {} ± {}".format(np.mean(baseline_acc), np.std(baseline_acc)))
    print("baseline q1: {} ± {}".format(np.mean(baseline_q1), np.std(baseline_q1)))
    print("ours: {} ± {}".format(ours_scores.mean(), ours_scores.std()))
    print("ours acc: {} ± {}".format(np.mean(ours_acc), np.std(ours_acc)))
    print("ours q1: {} ± {}".format(np.mean(ours_q1), np.std(ours_q1)))

    quad_ratio_scores_front = baseline_scores_front / (
        baseline_scores_front + ours_scores_front
    )
    quad_ratio_scores_back = baseline_scores_back / (
        baseline_scores_back + ours_scores_back
    )
    sorted_quad_ratio_scores = np.sort(
        np.concatenate((quad_ratio_scores_front, quad_ratio_scores_back))
    )
    quad_ratio_acc_front = baseline_acc_front / (baseline_acc_front + ours_acc_front)
    quad_ratio_acc_back = baseline_acc_back / (baseline_acc_back + ours_acc_back)
    sorted_quad_ratio_acc = np.sort(
        np.concatenate((quad_ratio_acc_front, quad_ratio_acc_back))
    )

    # scatter plots
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    HUGE_SIZE = 16

    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # score
    plt.scatter(
        baseline_scores_front / 20,
        ours_scores_front / 20,
        label="Front Palm",
        color="blue",
        alpha=0.8,
    )
    plt.scatter(
        baseline_scores_back / 20,
        ours_scores_back / 20,
        label="Back Palm",
        color="orange",
        alpha=0.8,
    )
    plt.scatter(
        (baseline_scores_front / 20).mean(),
        (ours_scores_front / 20).mean(),
        color="blue",
        marker="x",
        s=80,
    )
    plt.annotate(
        "Mean",
        xy=((baseline_scores_front / 20).mean(), (ours_scores_front / 20).mean()),
        color="k",
        xytext=(5, -10),
        textcoords="offset points",
    )
    plt.scatter(
        (baseline_scores_back / 20).mean(),
        (ours_scores_back / 20).mean(),
        color="orange",
        marker="x",
        s=80,
    )
    plt.annotate(
        "Mean",
        xy=((baseline_scores_back.mean() / 20), (ours_scores_back / 20).mean()),
        color="k",
        xytext=(5, -15),
        textcoords="offset points",
    )
    _, right = plt.xlim()
    _, top = plt.ylim()
    plt.xlim(0, max(right, top))
    plt.ylim(0, max(right, top))
    # plot red line for equal scores from x to xlim
    plt.plot(
        [0, max(right, top)],
        [0, max(right, top)],
        color="r",
        label="Equal Time",
    )
    plt.xlabel("Baseline Average Round Time [s]")
    plt.ylabel("Ours Average Round Time [s]")
    # set x and y limits
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(dst_path / "guess_char_scores_scatter.pdf"))
    plt.cla()
    plt.clf()
    # q1
    # plt.scatter(baseline_q1, ours_q1)
    # # plot red line for equal scores from x to xlim
    # plt.plot(
    #     baseline_q1,
    #     baseline_q1,
    #     color="r",
    #     label="Equal Difficulty",
    # )
    # plt.xlabel("Baseline Q1 [1-5]")
    # plt.ylabel("Ours Q1 [1-5]")
    # plt.title("How difficult was the task? [1 - easy, 5 - hard]")
    # plt.legend()
    # plt.savefig(str(dst_path / "guess_char_q1_scatter.pdf"))
    # plt.cla()
    # plt.clf()

    # heatmap plot
    # first sum the baselineq1 and ours q1 responses into a matrix
    response_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            response_matrix[i, j] = np.count_nonzero(
                (baseline_q1 == j + 1) & (ours_q1 == i + 1)
            )
    # normalize
    # response_matrix = response_matrix / response_matrix.sum()
    for i in range(5):
        for j in range(5):
            text = plt.text(
                j, i, int(response_matrix[i, j]), ha="center", va="center", color="w"
            )
    plt.imshow(response_matrix, cmap="coolwarm", interpolation="nearest")
    # plt.colorbar()
    plt.xlabel("Baseline")
    plt.ylabel("Ours")
    plt.xticks(np.arange(5), np.arange(1, 6))
    plt.yticks(np.arange(5), np.arange(1, 6))
    plt.title("How difficult was the task? [1 - easy, 5 - hard]")
    plt.savefig(str(dst_path / "guess_char_q1_heatmap.pdf"))

    # bar plots
    plt.stairs(
        sorted_quad_ratio_scores,
        np.linspace(0, 1, len(sorted_quad_ratio_scores) + 1),
        orientation="horizontal",
        hatch="//",
        label=r"$\frac{Baseline}{Baseline + Ours}$",
    )
    plt.stairs(
        np.linspace(1, 1, len(sorted_quad_ratio_scores)),
        np.linspace(0, 1, len(sorted_quad_ratio_scores) + 1),
        baseline=sorted_quad_ratio_scores,
        orientation="horizontal",
        hatch="//",
        label=r"$\frac{Ours}{Baseline + Ours}$",
    )
    plt.legend()
    plt.xlabel("Normalized Total Session Time")
    plt.ylabel("Sessions")
    plt.tight_layout()
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=True, bottom=True
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.vlines(x=0.5, ymin=0, ymax=1, color="r")
    plt.text(0.47, 0.7, "Equal Session Time", rotation=90, verticalalignment="center")
    plt.savefig(str(dst_path / "guess_char_scores.pdf"))
    plt.cla()
    plt.clf()
    width = 10.0
    categories = (0.0, width)
    weight_counts = {
        "1": np.array(
            [np.count_nonzero(ours_q1 == 1), np.count_nonzero(baseline_q1 == 1)]
        ),
        "2": np.array(
            [np.count_nonzero(ours_q1 == 2), np.count_nonzero(baseline_q1 == 2)]
        ),
        "3": np.array(
            [np.count_nonzero(ours_q1 == 3), np.count_nonzero(baseline_q1 == 3)]
        ),
        "4": np.array(
            [np.count_nonzero(ours_q1 == 4), np.count_nonzero(baseline_q1 == 4)]
        ),
        "5": np.array(
            [np.count_nonzero(ours_q1 == 5), np.count_nonzero(baseline_q1 == 5)]
        ),
    }

    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=HUGE_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=HUGE_SIZE)  # fontsize of the x and y labels
    # plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=HUGE_SIZE)  # fontsize of the figure title
    plt.rc("xtick", labelsize=HUGE_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=HUGE_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_aspect("equal")
    bottom = np.zeros(2)
    category_colors = plt.colormaps["coolwarm"](
        np.linspace(0.15, 0.85, len(weight_counts))
    )
    for i, (boolean, weight_count) in enumerate(weight_counts.items()):
        p = ax.barh(
            categories,
            weight_count,
            width,
            label=boolean,
            left=bottom,
            color=category_colors[i],
        )
        bottom += weight_count
        barlabels = [str(x) for x in weight_count]
        for j in range(len(barlabels)):
            if barlabels[j] == "0":
                barlabels[j] = ""
        ax.bar_label(
            p,
            labels=barlabels,
            label_type="center",
            color="k",
        )
    ax.set_yticks([0.0, width], labels=["Ours", "Baseline"])
    # ax.set_xticklabels([])
    # ax.set_ylabel("Sessions")
    ax.set_title("How difficult was the task?\n (1 - easy, 5 - hard)")
    # ax.legend(
    #     # ncol=5,
    #     # loc="upper center",
    #     # bbox_to_anchor=(0.5, 1.1),
    #     fancybox=True,
    #     shadow=True,
    # )
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # handles, labels = ax.get_legend_handles_labels()
    # see https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    # see https://stackoverflow.com/questions/34576059/reverse-the-order-of-a-legend
    # see https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
    # see https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio
    ax.legend(
        # handles[::-1],
        # labels[::-1],
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, -0.2),
        loc="lower center",
        ncols=len(weight_counts.items()),
    )
    # remove y ticks
    # ax.yaxis.set_visible(False)
    # remove x ticks
    ax.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(str(dst_path / "guess_char_q1.pdf"))


def func_latency(latency, u, tck, data):
    gt = splev(u + latency, tck)
    dist = np.mean(np.linalg.norm(np.array(gt).T[100:490] - data[100:490], axis=1))
    return dist


def ndc_to_pixel(data):
    screen_width = 1024
    screen_height = 768
    data[..., 0] = (data[..., 0] + 1) * 0.5 * (screen_width - 1)
    data[..., 1] = (data[..., 1] + 1) * 0.5 * (screen_height - 1)
    return data


def sim_plot(root_path, dst_path):
    for dir in root_path.glob("*"):
        data_gt = np.load(Path(dir, "simdata_gt.npy"))
        data_baseline = np.load(Path(dir, "simdata_baseline.npy"))
        data_naive = np.load(Path(dir, "simdata_naive.npy"))
        data_ours = np.load(Path(dir, "simdata_ours.npy"))
        data_kalman = np.load(Path(dir, "simdata_kalman.npy"))

        data_gt = ndc_to_pixel(data_gt)
        data_baseline = ndc_to_pixel(data_baseline)
        data_naive = ndc_to_pixel(data_naive)
        data_ours = ndc_to_pixel(data_ours)
        data_kalman = ndc_to_pixel(data_kalman)

        t_gt = np.load(Path(dir, "simtime_gt.npy"))
        t_bl = np.load(Path(dir, "simtime_baseline.npy"))
        t_naive = np.load(Path(dir, "simtime_naive.npy"))
        t_ours = np.load(Path(dir, "simtime_ours.npy"))
        t_kalman = np.load(Path(dir, "simtime_kalman.npy"))
        # find a parametric B-spline curve for gt data
        naive_distances = []
        ours_distances = []
        baseline_distances = []
        kalman_distances = []
        for i in range(data_gt.shape[1]):
            landmark_id = i
            x = data_gt[:, landmark_id, 0]
            y = data_gt[:, landmark_id, 1]
            tck, u_out = splprep([x, y], u=t_gt, s=0)
            # given the data timestamps, find what should be the gt data
            gt = splev(t_gt, tck)
            baseline_gt = splev(t_bl, tck)
            naive_gt = splev(t_naive, tck)
            ours_gt = splev(t_ours, tck)
            kalman_gt = splev(t_kalman, tck)
            # min_lat_naive = fmin(
            #     func_latency,
            #     -0.1,
            #     args=(
            #         t_naive,
            #         tck,
            #         data_naive[:, landmark_id],
            #     ),
            #     disp=0,
            # )
            # min_lat_bl = fmin(
            #     func_latency,
            #     -0.1,
            #     args=(
            #         t_bl,
            #         tck,
            #         data_baseline[:, landmark_id],
            #     ),
            #     disp=0,
            # )
            # min_lat_ours = fmin(
            #     func_latency,
            #     -0.1,
            #     args=(
            #         t_ours,
            #         tck,
            #         data_ours[:, landmark_id],
            #     ),
            #     disp=0,
            # )
            # min_lat_kalman = fmin(
            #     func_latency,
            #     -0.1,
            #     args=(
            #         t_kalman,
            #         tck,
            #         data_kalman[:, landmark_id],
            #     ),
            #     disp=0,
            # )
            # print("naive latency: {}".format(min_lat_naive))
            # print("baseline latency: {}".format(min_lat_bl))
            # print("ours latency: {}".format(min_lat_ours))
            # print("kalman latency: {}".format(min_lat_kalman))
            # t_ours_optimal = splev(t_ours + min_lat_ours, tck)
            # t_naive_optimal = splev(t_naive + min_lat_naive, tck)
            # t_kalman_optimal = splev(t_kalman + min_lat_kalman, tck)
            # t_bl_optimal = splev(t_bl + min_lat_bl, tck)
            # stdev_ours = np.std(
            #     np.linalg.norm(
            #         np.array(t_ours_optimal).T[100:490]
            #         - data_ours[100:490, landmark_id],
            #         axis=1,
            #     )
            # )
            # print("ours jitter: {}".format(stdev_ours))
            # stdev_naive = np.std(
            #     np.linalg.norm(
            #         np.array(t_naive_optimal).T[100:490]
            #         - data_naive[100:490, landmark_id],
            #         axis=1,
            #     )
            # )
            # print("naive jitter: {}".format(stdev_naive))
            # stdev_bl = np.std(
            #     np.linalg.norm(
            #         np.array(t_bl_optimal).T[100:490]
            #         - data_baseline[100:490, landmark_id],
            #         axis=1,
            #     )
            # )
            # print("baseline jitter: {}".format(stdev_bl))
            # stdev_kalman = np.std(
            #     np.linalg.norm(
            #         np.array(t_kalman_optimal).T[100:490]
            #         - data_kalman[100:490, landmark_id],
            #         axis=1,
            #     )
            # )
            # print("kalman jitter: {}".format(stdev_kalman))

            naive_dist = np.linalg.norm(
                np.array(naive_gt).T - data_naive[:, landmark_id], axis=1
            )
            ours_dist = np.linalg.norm(
                np.array(ours_gt).T - data_ours[:, landmark_id], axis=1
            )
            baseline_dist = np.linalg.norm(
                np.array(baseline_gt).T - data_baseline[:, landmark_id], axis=1
            )
            kalman_dist = np.linalg.norm(
                np.array(kalman_gt).T - data_kalman[:, landmark_id], axis=1
            )
            naive_distances.append(naive_dist)
            ours_distances.append(ours_dist)
            baseline_distances.append(baseline_dist)
            kalman_distances.append(kalman_dist)
            # plt.plot(t_ours, data_ours[:, landmark_id, 0], label="Ours")
            # plt.plot(t_kalman, data_kalman[:, landmark_id, 0], label="Kalman")
            # plt.plot(t_naive, data_naive[:, landmark_id, 0], label="Naive")
            # plt.plot(t_bl, data_baseline[:, landmark_id, 0], label="Basline")
            # plt.plot(t_ours, np.array(ours_gt).T[:, 0], label="GT")
            # plt.legend()

        naive_distances = np.array(naive_distances).mean(axis=0)
        ours_distances = np.array(ours_distances).mean(axis=0)
        baseline_distances = np.array(baseline_distances).mean(axis=0)
        kalman_distances = np.array(kalman_distances).mean(axis=0)
        # delayed_gt = splev(t_delayed, tck)

        # lets do a time plot of the distance from the signal as a func of time
        plt.plot(
            t_bl,
            baseline_distances,
            # s=1,
            label="Baseline",
            color="green",
            alpha=0.8,
            linewidth=0.5,
        )
        plt.plot(
            t_naive,
            naive_distances,
            # s=1,
            label="Naive",
            color="orange",
            alpha=0.8,
            linewidth=0.5,
        )
        plt.plot(
            t_kalman,
            kalman_distances,
            # s=1,
            label="Kalman Filter",
            color="red",
            alpha=0.8,
            linewidth=0.5,
        )
        plt.plot(
            t_ours,
            ours_distances,
            # s=1,
            label="Ours",
            color="blue",
            alpha=0.8,
            linewidth=0.5,
        )
        plt.xlabel("Time [ms]")
        plt.ylabel("Distance to Ideal [pixel]")
        plt.legend()
        # first we compute the "jitter" of the data, i.e. for each datapoint, its minimum distance to the gt curve

        # second we compute the latency of the data

        # plt.scatter(new_points[0], new_points[1], s=1)
        # plt.scatter(x, y, s=1)
        plt.tight_layout()
        plt.savefig(Path(dst_path, "{}_simdata.pdf".format(dir.stem)))
        plt.cla()
        plt.clf()


if __name__ == "__main__":
    # src_path = Path(
    #     "C:/Users/sens/Desktop/ahand/images/jnd_results/ours"
    #     # "C:/Users/sens/Desktop/ahand/images/jnd_results/baseline"
    # )
    # mask_path = Path("C:/src/augmented_hands/resource/images/mask.png")
    # dst_path = Path("C:/Users/sens/Desktop/ahand/images/jnd_processed")
    # jnd_process_images(src_path, mask_path, dst_path)

    # jnd_plot(Path("C:/Users/sens/Desktop/ahand/images/jnd_processed"))
    # guesschar_plot(
    #     Path("C:/Users/sens/Desktop/ahand/guess_char_results.csv"),
    #     Path("C:/Users/sens/Desktop/ahand/images/"),
    # )
    sim_plot(
        Path("C:/src/casper/debug/sim_data"),
        Path("C:/Users/sens/Desktop/casper_materials/images"),
    )
