import scipy.io
import matplotlib.pyplot as plt

rd = scipy.io.loadmat('rd.mat')
hn = scipy.io.loadmat('hn.mat')

# print(node1mobility)
import numpy as np

video_idx = 6
tile_idx = 6

for video_idx in range(15):
    video_bit_rate = rd['video_bitrate_data'][0, video_idx]  # 7x64xN: for (QP, tile index, N=frame index)
    video_ymse_data = rd['video_ymse_data'][0, video_idx]

    x = video_bit_rate[:, tile_idx, :].mean(axis=1)/ 2**20
    y = video_ymse_data[:, tile_idx, :].mean(axis=1)

    plt.plot(x, label=str(video_idx))
# plt.scatter(x, 10 * np.log10((255*255) / y))
plt.show()

print(rd)
print(hn)
