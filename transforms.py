from torchvision import transforms as tr

import numpy as np

from sklearn.preprocessing import minmax_scale

# Функция для получения train и val transform-ов, а так же функции для де-нормализации изображения
def get_transforms(mean, std):
    val_transform = tr.Compose([
        tr.CenterCrop(224),
        tr.Resize(256),
        tr.ToTensor(),
        tr.Normalize(mean=mean, std=std),
    ])

    def de_normalize(img, normalized=True):
        if not normalized:
            return img.transpose(1, 2, 0)
        return minmax_scale((img.cpu().numpy().reshape(3, -1) * std[:, None] + mean[:, None]),
                            feature_range=(0., 1.),
                            axis=1).reshape(*img.shape).transpose(1, 2, 0)

    return val_transform, de_normalize

means_stds = np.array([0.5, 0.5, 0.5])

val_transform_a, de_normalize_a = get_transforms(means_stds, means_stds)
val_transform_b, de_normalize_b = get_transforms(means_stds, means_stds)
