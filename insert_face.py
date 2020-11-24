import numpy as np
import torch

def affine(coords: torch.Tensor) -> torch.Tensor:
    r"""Computes transformation matrix which transforms point in homogeneous coordinates
        from canonical coordinate system into image
    Args:
        coords: coorinated of center, y direction and y direciton, :math:`(B, 3, 2)`

    Return:
        torch.Tensor: affine transformation matrix, :math:`(B, 3, 3)`
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


def upsample(image: torch.Tensor, ratio: int = 2):
    """
    Simple image upsampling by copying the pixel values
    Args:
        image: image to upsample, :math:`(B, C, H, W)`
        ratio: how many times each pixel should be copied, int

    Returns:
        Upsampled image
    """
    new_img = np.repeat(image, ratio, axis=2)
    new_img = np.repeat(new_img, ratio, axis=3)
    return new_img


def upsampling_needed(x, y):
    """
    Check if the generated image is smaller than the original faces in the photo.
    If yes, then upsampling will be performed.
    Args:
        x: x coordinates of the transformed image with face in the original photo
        y: y coordinates of the transformed image with face in the original photo

    Returns:
        True if image should be upsamples, False otherwise
    """
    res = x.shape[1]
    fst_idx = np.argmin(x)
    snd_idx = np.argmin(y)
    y1 = int(np.floor(fst_idx // res))
    x1 = fst_idx % res
    y2 = int(np.floor(snd_idx // res))
    x2 = snd_idx % res
    a = np.array([y[0, y1, x1], x[0, y1, x1]])
    b = np.array([y[0, y2, x2], x[0, y2, x2]])
    dist = np.linalg.norm(a-b)
    return True if dist > res else False


def create_grid(A: torch.Tensor, num_patches: int, PS: int):
    """
    Given a affine transformation A, return the positions of the pixels
    in the original photo.
    Args:
        A: matrix with the affine transformation, :math:`(B, 3, 3)`
        num_patches: numbers of images (=faces)
        PS: size of the generated image with face

    Returns:
        coordinates in the original photo
    """
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
    return x_coords, y_coords


def insert_faces_to_image(input_img: torch.Tensor,
                          A: torch.Tensor,
                          gen_imgs: torch.Tensor):
    """Given a affine transformation A, put the modified
        faces into the original image.

    Args:
        input_img: (torch.Tensor) images, :math:`(1, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        gen_imgs: (torch.Tensor)  images, :math: `(N, CH, PS, PS)`

    Returns:
        edited_image: (torch.Tensor) :math:`(1, CH, H, W)`
    """
    num_patches = A.size(0)
    PS = gen_imgs.size(2)
    x_coords, y_coords = create_grid(A, num_patches, PS)
    while upsampling_needed(x_coords,
                            y_coords):  # when the face in the original image has bigger resolution then generated image
        gen_imgs = upsample(gen_imgs)
        PS *= 2
        x_coords, y_coords = create_grid(A, num_patches, PS)
    # TODO: solve when the square with face is not whole in the photo
    for i in range(num_patches):
        input_img[:, :, y_coords[i, :, :], x_coords[i, :, :]] = gen_imgs[i, :, :, :]
    return input_img

