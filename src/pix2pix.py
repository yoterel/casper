import numpy as np
import gsoup
from pathlib import Path
import basler
import dynaflash

root_path = Path("C:/src/augmented_hands")
resource_path = Path(root_path, "resource")
test_pattern = gsoup.load_image(
    Path(resource_path, "XGA_colorbars.png"))
proj_wh = (1024, 768)
gray = gsoup.GrayCode()
patterns = gray.encode(proj_wh)
proj = dynaflash.projector()
proj.init()
# proj.project(test_pattern)
proj.project_white()

# gray_code_pattern = np.repeat(patterns[6], 3, axis=-1)
# proj.project(gray_code_pattern)
# proj.kill()

cam = basler.camera()
cam.init(7400.0)
cam.balance_white()
for i in range(10):
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
gsoup.save_images(captures, Path(root_path, "debug", "captures"))
forward, fg = gray.decode(captures, proj_wh, output_dir="debug", debug=True)
composed = forward * fg[..., None]
composed_normalized = (
    composed / np.array([proj_wh[1], proj_wh[0]])).astype(np.float32)
composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
# composed_normalized = np.concatenate((composed_normalized, np.zeros_like(composed_normalized[..., :1])), axis=-1)
gsoup.save_image(composed_normalized[..., 0:1], Path(
    "debug", "texturex.tiff"), extension="tiff")
gsoup.save_image(composed_normalized[..., 1:2], Path(
    "debug", "texturey.tiff"), extension="tiff")
