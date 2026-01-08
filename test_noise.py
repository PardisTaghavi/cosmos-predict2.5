#Dit noise injection
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

in_img = "/Users/ptgh/Downloads/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"

# noise
# t in [0,1]
# N(0,I)
# x_t = t * x_1 + (1−t) * x_0
# where x_1 is pure noise, x_0 is original image

# load image and normalize to [0,1]
img = Image.open(in_img)
x_0 = np.array(img).astype(np.float32) / 255.0

# sample pure noise x_1 ~ N(0,I)
x_1 = np.random.randn(*x_0.shape).astype(np.float32)

# noise schedule - different t values
t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(1, len(t_values), figsize=(20, 4))
for i, t in enumerate(t_values):
    # x_t = t * x_1 + (1-t) * x_0
    x_t = t * x_1 + (1 - t) * x_0
    # clip to valid range for display
    x_t_display = np.clip(x_t, 0, 1)
    axes[i].imshow(x_t_display)
    axes[i].set_title(f't = {t}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('noise_progression.png', dpi=150)
plt.show()


#add code below:
