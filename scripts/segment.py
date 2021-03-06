import torch
import numpy as np
import cv2


from autoballs.network.dataloader import get_preprocessing


def get_mask(img, model, pre_fn, device='cuda'):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.asarray(img)[:,:,:3]

    image = pre_fn(image=img)['image'][:1,:,:]
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pr_mask = model.predict(x_tensor)

    return (pr_mask.squeeze().cpu().numpy().round())