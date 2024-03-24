import gsoup
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator
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
            sid, _, method, palm, score, accuracy, q1 = line
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
    # fig, axs = plt.subplots(3, 1, figsize=(7, 15))
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

    categories = (0.0, 5.0)
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
    width = 5.0
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_aspect("equal")
    bottom = np.zeros(2)
    category_colors = plt.colormaps["inferno"](
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
            color="white",
        )
    ax.set_yticks([0.0, 5.0], labels=["Ours", "Baseline"])
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
    # plt.tight_layout()
    plt.savefig(str(dst_path / "guess_char_q1.pdf"))


if __name__ == "__main__":
    # src_path = Path(
    #     "C:/Users/sens/Desktop/ahand/images/jnd_results/ours"
    #     # "C:/Users/sens/Desktop/ahand/images/jnd_results/baseline"
    # )
    # mask_path = Path("C:/src/augmented_hands/resource/images/mask.png")
    # dst_path = Path("C:/Users/sens/Desktop/ahand/images/jnd_processed")
    # jnd_process_images(src_path, mask_path, dst_path)

    # jnd_plot(Path("C:/Users/sens/Desktop/ahand/images/jnd_processed"))
    guesschar_plot(
        Path("C:/Users/sens/Desktop/ahand/guess_char_results.csv"),
        Path("C:/Users/sens/Desktop/ahand/images/"),
    )
