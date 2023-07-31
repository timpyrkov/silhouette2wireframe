#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import gif
import itertools
import numpy as np
import pylab as plt
from PIL import Image
from PIL import ImageColor
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from pythonperlin import perlin, extend2d
import time


def remove_margins():
    """
    Removes figure margins, keeps only plot area 

    """
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    return


def load_grayscale_image(fname):
    """
    Loads an image and converts to grayscale
    
    Parameters
    ----------
    fname : str
        Path and file name of an image

    Returns
    -------
    img : ndarray
        Array of grayscale pixels

    """
    img = Image.open(os.path.expanduser(fname))
    alpha = np.array(img)[::-1]
    if alpha.shape[-1] % 2 == 0:
        alpha = alpha[...,-1] < 250
    else:
        alpha = np.zeros_like(alpha[...,0]).astype(bool)
    img = np.array(img.convert("L"))[::-1] / 255
    img[alpha] = 1
    return img


def load_blackwhite_image(fname, threshhold=0.9):
    """
    Loads an image and converts to black & white
    
    Parameters
    ----------
    fname : str
        Path and file name of an image
    threshold : float, default 0.9
        Threshold in range (0,1) to split grayscale to b&w

    Returns
    -------
    img : ndarray
        Array of black & white pixels

    """
    img = load_grayscale_image(fname)
    img = (img > threshhold).astype(float)
    return img


def fit_grid_shape_to_output_image(size):
    """
    Selects shape of coarse-grain grid to fit the desired size of output image
    
    Parameters
    ----------
    size : tuple
        Desired output image size [pixels]

    Returns
    -------
    shape : tuple
        Shape of grid

    """
    aspect = size[0] / size[1]
    shape = np.array([[35, 25], [25, 25], [25, 35]])
    diff = shape[:,0] / shape[:,1] - aspect
    i = np.argmin(np.abs(diff))
    shape = shape[i]
    return shape


def fit_input_image_to_grid_shape(img, shape):
    """
    Selects coarse-grain grid shape to fit output image size
    
    Parameters
    ----------
    img : ndarray
        Input image, array of black & white pixels
    shape : tuple
        Shape of grid

    Returns
    -------
    img : ndarray
        Resized image to fit the grid

    """
    size = img.shape[::-1]
    aspect = size[0] / size[1]
    
    if aspect >= shape[0] / shape[1]:
        # if input_aspect is greater - fit x and expand y
        n = int(size[0] / shape[0])
        nx, ny = n * shape[0], n * shape[1]
        dx, dy = size[0] - nx, ny - size[1]
        img = img[:,dx//2:][:,:nx]
        if dy > 0:
            pad = np.stack([img[-1]] * dy)
            img = np.vstack([img, pad])
    else:
        # else - fit y and expand x
        n = int(size[1] / shape[1])
        nx, ny = n * shape[0], n * shape[1]
        dx, dy = nx - size[0], size[1] - ny
        img = img[dy//2:][:ny]
        if dx > 0:
            pad0 = np.stack([img[:,0]] * (dx // 2)).T
            pad1 = np.stack([img[:,-1]] * (dx - dx // 2)).T
            img = np.hstack([pad0, img, pad1])
    return img
    

def to_prime_factors(n):
    """
    Splits output image size into prime number factors
    
    Parameters
    ----------
    n : int
        Width or height

    Returns
    -------
    factors : list
        Prime number factors

    """
    factors = []
    i = 2
    while i * i < n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    factors.append(n)
    factors.append(1)
    return factors


def size_to_size_and_dpi(size):
    """
    Splits output image size into size (~5 inch) and dpi
    
    Parameters
    ----------
    size : tuple
        Width and height

    Returns
    -------
    size : tuple
        Output image size [inch]
    dpi : int
        Output image resolution [dots per inch]

    """
    dpi = np.gcd(*size)
    size = np.asarray(size) // dpi
    dpi = to_prime_factors(dpi)
    while size.min() < 5 and max(dpi) > 1:
        size = dpi.pop(0) * size
    dpi = np.prod(dpi)
    return size, dpi


def extend_time(arr):
    """
    Extennds 3D array along axis = 0 (Time)

    Parameters
    ----------
    arr : ndarray
        3D image array

    Returns
    -------
    arr : ndarray
        Extended 3D image array

    """
    arr = np.repeat(arr, 2, axis=0)
    ext = np.vstack([arr, np.expand_dims(arr[0], axis=0)])
    ext = 0.5 * (ext[1:] + ext[:-1])
    arr[1::2] = ext[1::2]
    return arr


def split_array_into_blocks(arr, shape):
    """
    Splits array into nx * ny blocks

    Parameters
    ----------
    arr : ndarray
        2D image array, dimensions: 0 - Y, 1 - X
    shape : tuple
        Number of blocks (nx, ny)

    Returns
    -------
    blocks : list of lists
        List of lists, lengths: nx * ny

    """
    nx, ny = shape
    blocks = np.hsplit(arr, nx)
    blocks = np.vstack(blocks)
    blocks = np.split(blocks, nx * ny)
    blocks = [blocks[i:i + ny] for i in range(0, nx * ny, ny)]
    return blocks


def com2d(arr):
    """
    Calculates center of mass (COM) of points in a 2D array
    
    Parameters
    ----------
    arr : ndarray
        2D array of mass weights; indices are treated as coords

    Returns
    -------
    com : float
        Center of mass

    """
    assert arr.ndim == 2
    com = np.zeros((2))
    if np.std(arr):
        summ = np.sum(arr)
        for axis in range(2):
            n = arr.shape[axis]
            idx = (np.arange(n) + .5) / n - .5
            if axis:
                com[axis] = np.sum(np.dot(arr, idx), axis=0) / summ
            else:
                com[axis] = np.sum(np.dot(arr.T, idx), axis=0) / summ
    return com


def get_grid_grads(blocks):
    """
    Calculates gradients (X and Y displacements) based on edge detection
    
    Parameters
    ----------
    blocks : list of lists
        List of lists, lengths: nx * ny

    Returns
    -------
    grad : tuple
        X and Y displacements of grid nodes

    """
    nx, ny = len(blocks), len(blocks[0])
    blocks = list(itertools.chain(*blocks))
    grad = np.array([com2d(1 - b) - com2d(b) for b in blocks])[:,::-1].T
    grad = (grad[0].reshape(nx, ny), grad[1].reshape(nx, ny))
    return grad


def get_grid_values(blocks):
    """
    Calculates values of black pixel fraction in grid blocks
    
    Parameters
    ----------
    blocks : list of lists
        List of lists, lengths: nx * ny

    Returns
    -------
    values : ndarray
        Values of black pixel fractions

    """
    nx, ny = len(blocks), len(blocks[0])
    blocks = list(itertools.chain(*blocks))
    values = np.array([np.mean(1 - b) for b in blocks])
    values = values.reshape(nx, ny)
    return values


def get_grid_coords(shape):
    """
    Generates coordinates of grid nodes
    
    Parameters
    ----------
    shape : tuple
        Number of nodes (nx, ny)

    Returns
    -------
    x : ndarray
        2D-array of shape = (nx, ny)
    y : ndarray
        2D-array of shape = (nx, ny)

    """
    nx, ny = shape
    x, y = np.arange(nx), np.arange(ny)
    x, y = np.meshgrid(x, y, indexing="ij")
    return x, y


def blur_gradients(grad):
    """
    Blures closeness to edges to all distand grid nodes

    Parameters
    ----------
    grad : tuple
        X and Y displacementss of grid nodes

    Returns
    -------
    levels : ndarray
        Closeness of nodes to image edges (in range 0,1)

    """
    dx, dy = grad
    levels = (np.abs(dx) > 0) | (np.abs(dy) > 0)
    levels = levels.astype(float)
    level = np.max(levels) + 1
    start = True
    while start:
        start = False
        l_next = np.copy(levels)
        for i in [-1, 1]:
            for j in [-1, 1]:
                l = np.pad(levels, (1,1))
                l = np.roll(l, i, 0)
                l = np.roll(l, j, 1)
                l = l[1:-1][:,1:-1]
                l_next[l > 0] = 1
        mask = (levels == 0) & (l_next > 0)
        if np.any(mask):
            start = True
            levels[mask] = level
            level += 1
    levels = 1 / levels
    return levels


def image_to_grid(fname, size=(1080,1350), threshhold=0.9):
    """
    Loadsinput image and converts it to a coarse-grained grid

    Parameters
    ----------
    fname : str
        Path and file name of an image
    size : tuple, default (1080,1350)
        Desired output image size [pixels]
    threshold : float, default 0.9
        Threshold in range (0,1) to split grayscale to b&w

    Returns
    -------
    img : ndarray
        Array of black & white pixels
    grad : tuple
        Tuple of six 2D-arrays representing a property of each grid node:
        XY-coords, values, XY-displacements, closeness to edges

    """    
    # Load B&W silhouette image as 2D array
    img = load_blackwhite_image(fname, threshhold)
    # Fit grid shape and input image to the output size
    shape = fit_grid_shape_to_output_image(size)
    img = fit_input_image_to_grid_shape(img, shape)
    # Split image into grid blocks
    blocks = split_array_into_blocks(img, shape)
    # Calculate grid coords, values, and gradients
    x, y = get_grid_coords(shape)
    value = get_grid_values(blocks)
    grad = get_grid_grads(blocks)
    # Blur edges (abs of gradients)
    level = blur_gradients(grad)
    # Combine coords, values, gradients, levels to tuple
    grid = x, y, value, grad[0], grad[1], level
    return img, grid


def show_grid(grid):
    """
    Shows grid as scatterplots of values, grads and closeness to image edges

    Parameters
    ----------
    grid : tuple
        Grid, an output of image_to_grid()

    """    
    x, y, value, dx, dy, level = grid
    plt.figure(figsize=(24,4), facecolor="w")
    plt.subplot(141)
    plt.title("Value")
    plt.scatter(x, y, c=value, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(142)
    plt.title("Grad_x")
    plt.scatter(x, y, c=dx, cmap="coolwarm", vmin=-.5, vmax=.5)
    plt.colorbar()
    plt.subplot(143)
    plt.title("Grad_y")
    plt.scatter(x, y, c=dy, cmap="coolwarm", vmin=-.5, vmax=.5)
    plt.colorbar()
    plt.subplot(144)
    plt.title("Blurred edges")
    plt.scatter(x, y, c=level, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    return


def colorize_grid(grid, top, bottom):
    """
    Assigned color to each grid node by gradient from top to bottom

    Parameters
    ----------
    grid : tuple
        Grid, an output of image_to_grid()
    top : str or matplotlib.colors.Color object
        Top color
    bottom : str or matplotlib.colors.Color object
        Bottom color

    Returns
    -------
    colors : ndarray
        3D-array of shape = (nx, ny, n_channels), channels = RGBA

    """    
    x, y, value, dx, dy, level = grid
    z = y / y.max() - 0.15 * (level >= 1)
    z = extend2d(z, n=4, axis=0)
    z = np.clip(z, 0, 1)
    cmap = LinearSegmentedColormap.from_list("cmap", [bottom, top])
    colors = np.stack([cmap(z_) for z_ in z])
    return colors



class ImageToWireframe():
    """
    This class loads image, generates 3D Perlin noise array (Time, X, Y), 
    draws and animates wireframes based on image.

    Parameters
    ----------
    fname : str
        Path and file name of an input image
    size : tuple, default (1080,1350)
        Desired output image size [pixels]
    threshold : float, default 0.9
        Threshold in range (0,1) to split grayscale to b&w
    seed : int, default 0
        Random seed for Perlin noise generation
    top : str or matplotlib.colors.Color object, default "#000022"
        Background color
    top : str or matplotlib.colors.Color object, default "#FF0000"
        Top color
    bottom : str or matplotlib.colors.Color object, default "#FFFF00"
        Bottom color

    """

    def __init__(self, fname, size=(1080,1350), threshold=0.9, seed=0, 
                 bg="#000022", top="#FF0000", bottom="#FFFF00"):
        img, grid = image_to_grid(fname, size, threshold)
        self.cdict = {"bg": bg, "top": top, "bottom": bottom}
        size, dpi = size_to_size_and_dpi(size)
        self.shape = grid[0].shape
        self.size = size
        self.dpi = dpi
        self.seed = seed
        self.grid = grid
        self.img = img
        self.generate_perlin_noise(seed=seed)
        #self.w = 2 * np.pi / (self.nz * self.dens)
        print("P", self.px.shape)


    def generate_perlin_noise(self, dens=20, seed=0):
        """
        Generates Perlin noise
        
        Parameters
        ----------
        dens : int, default 20
            Defines the number frames = 10 * dens, must be a multiple of 4
        seed : int, default 0
            Random seed for Perlin noise generation

        Returns
        -------
        p : ndarray
            3D Perlin noise array (Time, X, Y)

        """
        t0 = time.time()
        assert dens % 4  == 0
        print("Start generating perlin noise...")
        shape = (10,) + self.shape
        p = perlin(shape, dens=dens//2, seed=seed)
        d, d2 = dens // 4, dens // 2
        p = p[:,d:][:,:,d:]
        p = p[:,::d2][:,:,::d2]
        p = extend_time(p)
        z = np.exp(2j * np.pi * p)
        self.px = z.real
        self.py = z.imag        
        dt = time.time() - t0
        nmin, nsec = dt // 60, dt % 60
        print(f"Done. Time: {nmin:.0f} min {nsec:.1f} sec")
        return p


    def show_grid(self):
        """
        Shows grid as scatterplots of values, grads and closeness to image edges

        """    
        show_grid(self.grid)
        return


    def show_image(self):
        """
        Shows input image fitted to match the desired aspect ratio of the output

        """    
        plt.imshow(self.img)
        plt.ylim(0, len(self.img))
        plt.show()
        return


    def draw_frame(self, i, verbose=False, cdict=None):
        """
        Cretes a wireframe image
        
        Parameters
        ----------
        i : int
            Frame number
        verbose : bool, default False
            If True prints number for each tenth frame to track progress
        cdict : dict or None, default None
            Color dict, must include keys: 'bg', 'top', 'bottom'

        Returns
        -------
        fig : matplotlib.figure.Figure
            Wireframe image

        """
        if verbose and (i + 1) % 10 == 0:
            print(f"Frame {i + 1}")
        x, y, value, dx, dy, level = self.grid

        # Seed noisy x and y
        p = 1.5
        xn = x + p * self.px[i]
        yn = y + p * self.py[i]
        xn = extend2d(xn[:,::2][::2])
        yn = extend2d(yn[:,::2][::2])

        # Seed displaced x and y
        d = 1.5
        xd = x + d * dx
        yd = y + d * dy

        # Mix noisy and displaced x and y periodically
        phi = np.pi + (i - x - y) * 2 * np.pi / len(self.px)
        per = 0.9 * np.power((1 + np.cos(phi)) / 2, 2)
        f = 0.6 * ((level > 0.3) | (value > 0)).astype(float)
        mask = f > 0
        per[~mask] = 0
        f = np.max(np.stack([f, per]), axis=0)
        x = f * xd + (1 - f) * xn
        y = f * yd + (1 - f) * yn

        # Populate wires along x axis
        x = extend2d(x, n=4, axis=0)
        y = extend2d(y, n=4, axis=0)
        nx, ny = x.shape

        # Draw frame with aspect ratio 0.8 (1080 x 1350)
        cdict = cdict if cdict is not None else self.cdict
        color = colorize_grid(self.grid, cdict["top"], cdict["bottom"])
        fig = plt.figure(figsize=self.size, facecolor=cdict["bg"])
        #fig.set_dpi(self.dpi)
        for j in range(nx):
            for k in range(ny-1):
                c = color[j][k]
                xs = [x[j,k], x[j,k+1]]
                ys = [y[j,k], y[j,k+1] - 0.05]
                plt.plot(xs, ys, color=c)
        plt.xlim(-3, self.shape[0] + 3)
        plt.ylim(-1, self.shape[1] + 1)
        plt.tight_layout()
        remove_margins()
        return fig


    def draw_animation(self, output="wireframe.gif", duration=120, cdict=None):
        """
        Cretes a wireframe animation
        
        Parameters
        ----------
        output : str
            Path and file name to an output gif animated image
        duration: int, default 120
            Delay between gif frames [milliseconds]
        cdict : dict or None, default None
            Color dict, must include keys: 'bg', 'top', 'bottom'

        """
        # Set the dots per inch resolution
        gif.options.matplotlib["dpi"] = self.dpi

        # Decorate a plot function with @gif.frame
        @gif.frame
        def plot(i):
            self.draw_frame(i, verbose=True, cdict=cdict)

        # Construct "frames"
        n = self.px.shape[0]
        frames = [plot(i) for i in range(0, n, 1)]

        # Save "frames" to gif with a specified duration (milliseconds) between each frame
        gif.save(frames, output, duration=duration)
        return


