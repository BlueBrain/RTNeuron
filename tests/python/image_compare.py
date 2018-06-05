## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Juan Hernando <juan.hernando@epfl.ch>
##
## This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
##
## This library is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License version 3.0 as published
## by the Free Software Foundation.
##
## This library is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along
## with this library; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Requires Python Image Library
import rtneuron
from PIL import ImageChops, Image
import numpy
import math
import setup
import os

_KEEP_IMAGES = 'KEEP_IMAGES' in os.environ

# Computes the root mean square error in each of the given pair of images
def _per_channel_RMS(img1, img2):
    diff = ImageChops.difference(img1, img2)
    hist = diff.histogram()
    chunk = [0, 0]
    rms = []

    # Only 8-bit per channel images are supported by this implementation
    squared_errors = numpy.array(range(0, 256))
    squared_errors **= 2

    for band in diff.getbands():
        if band not in ['R', 'G', 'B', 'A', 'L', 'C', 'M', 'Y', 'K']:
            raise ValueError("Unsupported image format for comparison")
        # Range for the values of the next channel in hist. the interval is
        # open on the right
        chunk = [chunk[1], chunk[1] + 256]
        # Summation of the component-wise multiplication of the histogram counts
        # by their squared errors values. This gives the absolute error
        squared_abs_error = (
            numpy.array(hist[chunk[0]:chunk[1]]) * squared_errors).sum()
        rms.append(math.sqrt(squared_abs_error /
                             float(img1.size[0] * img1.size[1])))
    return rms

def _get_tmp_file_name(suffix = '.png'):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete = False, suffix = suffix)
    name = tmp.name
    # The file is supposed to be overwritten
    tmp.close()
    return name

def compare(img1, img2, channel_threshold = 1.75):
    # 'with' used because PIL is not closing the files properly
    with open(img1, "rb") as imgfile1:
        with open(img2, "rb") as imgfile2:
            img1 = Image.open(imgfile1)
            img2 = Image.open(imgfile2)
            assert(img1.size == img2.size)
            assert(img1.getbands() == img2.getbands())
            rms = _per_channel_RMS(img1, img2)
            if not all([x < channel_threshold for x in rms]):
                raise AssertionError("comparing images: %s > %f" % (
                    str(rms), channel_threshold))

def capture_and_compare(view, sample_name, threshold = 1.75,
                        snapshot = rtneuron.View.snapshot,
                        prepend_sample_path = True):
    if _KEEP_IMAGES:
        name = sample_name
        if not prepend_sample_path:
            name = name[:-3] + "bis.png"
    else:
        name = _get_tmp_file_name()

    snapshot(view, name)
    try:
        sample_path = sample_name
        if prepend_sample_path:
            paths = []
            for path in setup.golden_sample_paths:
                sample_path = path + sample_name
                if os.path.isfile(sample_path):
                    break
        compare(name, sample_path, threshold)

    finally:
        if not _KEEP_IMAGES:
            os.remove(name)

def compare_tiled_image(ref_image, tiled_image, ref_size, tile_size):
    composition = Image.new('RGB', ref_size)

    for i in range(0, ref_size[1], tile_size[1]):
        for j in range(0, ref_size[0], tile_size[0]):
            suffix = "%01d-%01d.png" % (j/tile_size[0], i/tile_size[1])
            with open(tiled_image + suffix, "rb") as tile_name:
                tile = Image.open(tile_name)
                composition.paste(tile, (i,j))

    composition_name = "snapshot_tiled_composition.png"
    composition.save(composition_name)
    try:
        compare(ref_image, composition_name)
    finally:
        if not _KEEP_IMAGES:
            os.remove(ref_image)
            os.remove(composition_name)

def capture_temporary(view):
    name = _get_tmp_file_name()
    view.snapshot(name)
    return name
