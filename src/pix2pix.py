import numpy as np
import gsoup
from pathlib import Path
import basler
import dynaflash

proj_wh = (1024, 768)
gray = gsoup.GrayCode()
patterns = gray.encode(proj_wh)
proj = dynaflash.projector(proj_wh[0], proj_wh[1])
proj.init()
proj.project_white()
cam  = basler.camera()
cam.init()
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
gsoup.save_images(captures, Path("debug", "captures"))
forward, fg = gray.decode(captures, proj_wh, output_dir="debug", debug=True)
composed = forward * fg[..., None]
composed_normalized = (composed / np.array([proj_wh[1], proj_wh[0]])).astype(np.float32)
composed_normalized[..., [0, 1]] = composed_normalized[..., [1, 0]]
# composed_normalized = np.concatenate((composed_normalized, np.zeros_like(composed_normalized[..., :1])), axis=-1)
gsoup.save_image(composed_normalized[..., 0:1], Path("debug", "texturex.tiff"), extension="tiff")
gsoup.save_image(composed_normalized[..., 1:2], Path("debug", "texturey.tiff"), extension="tiff")
