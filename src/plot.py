import gsoup
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator
from scipy import ndimage
import cv2
from PIL import Image, ImageDraw


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
    import matplotlib.pyplot as plt

    # set bar plots
    n_subjects = len(baseline)
    ind = np.arange(n_subjects)
    width = 0.3
    plt.bar(ind, baseline, width, label="baseline", color="blue", alpha=0.8)
    plt.bar(ind + width, ours, width, label="ours", color="green", alpha=0.8)
    plt.xticks(ind + width / 2, tuple(str(i) for i in range(1, n_subjects + 1)))
    # set means as horizontal lines
    plt.axhline(baseline.mean(), color="blue", linewidth=2, linestyle="--")
    plt.axhline(ours.mean(), color="green", linewidth=2, linestyle="--")
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
    plt.savefig(str(dst_path / "jnd.png"))


def guesschar_plot(dst_path):
    baseline_score_f = [55.6936, 62.7488, 83.6083, 66.7138, 65.8102, 55.8145]
    baseline_score_b = [91.0068, 60.4948, 63.1906, 63.9748, 63.2502, 42.1859]
    baseline_acc = [0.6, 0.9, 0.95, 0.9, 1, 0.95, 0.8, 0.85, 0.9, 0.85, 0.9, 0.95]
    baseline_q1 = [3, 5, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3]
    ours_score_f = [48.1361, 55.4559, 46.1222, 40.5935, 45.5417, 35.9801]
    ours_score_b = [48.0332, 62.7061, 35.8121, 28.9236, 36.9414, 34.0126]
    ours_acc = [0.9, 0.75, 0.95, 1, 0.75, 0.95, 0.85, 0.95, 0.85, 1, 0.75, 1]
    ours_q1 = [4, 2, 3, 3, 1, 2, 1, 2, 2, 2, 2, 2]

    baseline = np.array(baseline_score_f + baseline_score_b)
    ours = np.array(ours_score_f + ours_score_b)
    # report mean and stds
    print("baseline: {} ± {}".format(baseline.mean(), baseline.std()))
    print("baseline acc: {} ± {}".format(np.mean(baseline_acc), np.std(baseline_acc)))
    print("baseline q1: {} ± {}".format(np.mean(baseline_q1), np.std(baseline_q1)))
    print("ours: {} ± {}".format(ours.mean(), ours.std()))
    print("ours acc: {} ± {}".format(np.mean(ours_acc), np.std(ours_acc)))
    print("ours q1: {} ± {}".format(np.mean(ours_q1), np.std(ours_q1)))


if __name__ == "__main__":
    # src_path = Path(
    #     "C:/Users/sens/Desktop/images/jnd_results/ours"
    #     # "C:/Users/sens/Desktop/images/jnd_results/baseline"
    # )
    # mask_path = Path("C:/src/augmented_hands/resource/images/mask.png")
    # dst_path = Path("C:/Users/sens/Desktop/images/jnd_processed")
    # jnd_process_images(src_path, mask_path, dst_path)

    # jnd_plot(Path("C:/Users/sens/Desktop/images/jnd_processed"))
    guesschar_plot(Path("C:/Users/sens/Desktop/images/"))
