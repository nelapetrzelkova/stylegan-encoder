import numpy as np
import torch
import argparse
import os
import kornia
import cv2
import matplotlib.pyplot as plt
import torchvision
import PIL
from imutils import face_utils
import cv2
import dlib
import PIL.Image
from keras.utils import get_file
import bz2
import numpy as np
from PIL import ImageFilter


def imshow_torch(tensor, *kwargs):
    plt.figure(figsize=(10,15))
    plt.imshow(kornia.tensor_to_image(tensor), *kwargs)
    return


def load_img(path):
    img1 = cv2.imread(path)
    timg1 = kornia.image_to_tensor(img1, False)
    timg1 = kornia.color.bgr_to_rgb(timg1)
    return timg1


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
    y1 = int(np.floor(fst_idx // res))-1
    x1 = fst_idx % res
    y2 = int(np.floor(snd_idx // res))-1
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


def padding_needed(input_img, x, y):
    pad_val = 0
    b, c, h, w = input_img.shape
    min_y, max_y, min_x, max_x = np.min(y), np.max(y), np.min(x), np.max(x)
    if min_y < 0:
        pad_val = -min_y
    if max_y > h and max_y-h > pad_val:
      pad_val = max_y-h
    if min_x < 0 and -min_x > pad_val:
        pad_val = -min_x
    if max_x > w and max_x-w > pad_val:
      pad_val = max_x-w
    return pad_val


def pad_img(input_img, val):
    b, c, h, w = input_img.shape
    padded_img = np.zeros((b, c, h+val, w+val))
    padded_img[:, :, :h, :w] = input_img
    return padded_img


def blend(inner, outer, transition=150):
    t = transition
    inner = np.transpose(inner, (1, 2, 0)).numpy()
    outer = np.transpose(outer, (1, 2, 0)).numpy()
    coeffs = np.linspace(0, 1, t)[np.newaxis, :, np.newaxis]
    l = np.tile(coeffs, (inner.shape[1], 1, 3))
    r = np.flip(l, 1)
    u = np.transpose(l, (1, 0, 2))
    d = np.flip(u, 0)
    inner[:t, :, :] = inner[:t, :, :]*u + outer[:t, :, :]*(1-u)
    inner[-t:, :, :] = inner[-t:, :, :]*d + outer[-t:, :, :]*(1-d)
    inner[:, :t, :] = inner[:, :t, :]*l + outer[:, :t, :]*(1-l)
    inner[:, -t:, :] = inner[:, -t:, :]*r + outer[:, -t:, :]*(1-r)
    return np.transpose(inner, (2, 0, 1))

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
    b, c, h, w = input_img.shape
    x_coords, y_coords = create_grid(A, num_patches, PS)
    while upsampling_needed(x_coords,
                            y_coords):  # when the face in the original image has bigger resolution then generated image
        gen_imgs = upsample(gen_imgs)
        PS *= 2
        x_coords, y_coords = create_grid(A, num_patches, PS)
    pad_val = padding_needed(input_img, x_coords, y_coords)
    if pad_val > 0:
        input = pad_img(input_img, pad_val)
    for i in range(num_patches):
        blended = blend(gen_imgs[i, :, :, :], input_img[0, :, y_coords[i, :, :], x_coords[i, :, :]])
        input_img[0, :, y_coords[i, :, :], x_coords[i, :, :]] = torch.from_numpy(blended)
    if pad_val > 0:
        output = input_img[:, :, :h, :w]
    else:
        output = input_img
    return output


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def generate_face_mask(im, use_grabcut=True, scale_mask=1.4):
    detector = dlib.get_frontal_face_detector()
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    predictor = dlib.shape_predictor(landmarks_model_path)
    im = np.array(im)
    rects = detector(im, 1)
    # loop over the face detections
    for (j, rect) in enumerate(rects):
        """
        Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        """
        shape = predictor(im, rect)
        shape = face_utils.shape_to_np(shape)

        # we extract the face
        vertices = cv2.convexHull(shape)
        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, vertices, 1)
        if use_grabcut:
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (0, 0, im.shape[1], im.shape[2])
            (x, y), radius = cv2.minEnclosingCircle(vertices)
            center = (int(x), int(y))
            radius = int(radius * scale_mask)
            mask = cv2.circle(mask, center, radius, cv2.GC_PR_FGD, -1)
            cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
            cv2.grabCut(im, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask = np.where((mask == 2) | (mask == 0), 0, 1)
        imask = (255 * mask).astype('uint8')
        imask = PIL.Image.fromarray(imask, 'L')
        # imask.save(dest, 'PNG')
        return imask


def apply_masks(gen_imgs, background, composite_blur=8):
    new_imgs = np.zeros_like(gen_imgs)
    for i, gen_img in enumerate(gen_imgs):
        img = PIL.Image.fromarray(gen_img)
        mask = generate_face_mask(img, use_grabcut=False)
        mask = mask.filter(ImageFilter.GaussianBlur(composite_blur))
        mask = np.array(mask)/255
        mask = np.expand_dims(mask, axis=-1)
        img_array = mask*np.array(gen_img) + (1.0-mask)*np.array(background)
        img_array = img_array.astype(np.uint8)
        new_imgs[i, :, :, :] = img_array
    return new_imgs

def get_background(input, A, PS):
      x_coords, y_coords = create_grid(A, 1, PS)
      square = input[0, :, y_coords, x_coords].squeeze()
      return np.transpose(square, (1, 2, 0)).numpy()
