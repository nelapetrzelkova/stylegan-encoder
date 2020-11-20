import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def affine(coords: torch.Tensor) -> torch.Tensor:
    r"""Computes transformation matrix M which transforms point in homogeneous coordinates
        from canonical coordinate system into image

    Return:
        torch.Tensor: affine transformation matrix

    Shape:
        - Input :math:`(B, 3, 2)`
        - Output: :math:`(B, 3, 3)`

    """

    B = coords.size(0)
    coords[:, 1, :] += coords[:, 0, :]
    coords[:, 2, :] += coords[:, 0, :]
    M = coords.transpose(1, 2)
    # N = inverse of [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
    N = torch.tensor([[-1., -1., 1.], [1., 0., 0.], [0., 1., 0.]]).double().unsqueeze(0).repeat(B, 1, 1)
    out = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    prod = torch.bmm(M, N)
    out[:, 0:2, :] = prod
    return out


def insert_faces_to_image(input: torch.Tensor,
                          A: torch.Tensor,
                          gen_img: torch.Tensor):
    """Extract patches defined by affine transformations A from image tensor X.

    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        gen_img: (torch.Tensor)  images, :math: `(N, CH, PS, PS)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        --- so far just for one image


    Returns:
        edited_images: (torch.Tensor) :math:`(B, CH, H, W)`
    """

    b, ch, h, w = input.size()
    num_patches = A.size(0)
    PS = gen_img.size(2)
    x = torch.linspace(-1.0, +1.0, PS)  # create grid
    meshx, meshy = torch.meshgrid([x, x])
    x = meshx.reshape(1, PS * PS).repeat(num_patches, 1)
    y = meshy.reshape(1, PS * PS).repeat(num_patches, 1)
    gr = torch.ones([num_patches, 3, PS * PS])
    gr[:, 0, :] = x
    gr[:, 1, :] = y
    transformed = torch.bmm(A, gr)  # apply transformation
    transformed_x = transformed[:, 0, :].reshape(num_patches, PS, PS)
    transformed_y = transformed[:, 1, :].reshape(num_patches, PS, PS)
    grid = torch.zeros([num_patches, PS, PS, 2])
    grid[:, :, :, 0] = transformed_x
    grid[:, :, :, 1] = transformed_y
    x_coords = grid[:, :, :, 0].int().numpy()
    y_coords = grid[:, :, :, 1].int().numpy()
    for i in range(num_patches):
        input[:, :, y_coords[i, :, :], x_coords[i, :, :]] = gen_img[i, :, :, :]
    return input


