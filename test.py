import torch
import numpy as np
import matplotlib.pyplot as plt

# 載入生成器模型

# 檢查是否有可用的GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


# 可選：載入訓練好的生成器權重
G=torch.load('Generator_epoch_11.pth').to(device)

# 生成噪音
noise = torch.rand(16, 128).to(device)

# 通過生成器生成圖像
fake_image = G(noise)

# 將生成的圖像轉換為NumPy數組
imgs_numpy = (fake_image.data.cpu().numpy() + 1.0) / 2.0

# 顯示生成的圖像
plt.rcParams['figure.figsize'] = (10.0, 8.0)

def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.transpose(1, 2, 0))

show_images(imgs_numpy)
plt.show()
