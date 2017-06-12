import numpy as np
import struct
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread

MAXCOLS = 60
ncols = 0
colorwheel = None


def read_flow_file(base_path, scene, n):
    return read_flow_file_with_path(base_path + "/flow/" + scene + "/frame_%0.4i" % n + ".flo")


def read_flow_file_with_path(path):
    with open(path, "rb") as flow_file:
        raw_data = flow_file.read()
        assert struct.unpack_from("f", raw_data, 0)[0] == 202021.25  # check to make sure the file is being read correctly
        width, height = struct.unpack_from("ii", raw_data, 4)

        data = np.zeros([width, height, 2], dtype=np.float32)
        for j in range(height):
            for i in range(width):
                data[i, j, 0] = struct.unpack_from("f", raw_data, 8 * (i + j * width) + 12)[0]
                data[i, j, 1] = struct.unpack_from("f", raw_data, 8 * (i + j * width) + 12 + 4)[0]
        return data


def setcols(r, g, b, k):
    colorwheel[k][0] = r
    colorwheel[k][1] = g
    colorwheel[k][2] = b


def makecolorwheel():
    # relative lengths of color transitions:
    # these are chosen based on perceptual similarity
    # (e.g. one can distinguish more shades between red and yellow
    #  than between yellow and green)
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    global ncols, colorwheel
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3], dtype=np.float32)
    print "ncols = %d\n" % ncols
    if (ncols > MAXCOLS):
        raise EnvironmentError("something went wrong?")
    k = 0
    for i in range(RY):
        setcols(1,	   1.0*float(i)/RY,	 0,	       k)
        k += 1
    for i in range(YG):
        setcols(1.0-float(i)/YG, 1,		 0,	       k)
        k += 1
    for i in range(GC):
        setcols(0,		   1,		 float(i)/GC,     k)
        k += 1
    for i in range(CB):
        setcols(0,		   1-float(i)/CB, 1,	       k)
        k += 1
    for i in range(BM):
        setcols(float(i)/BM,	   0,		 1,	       k)
        k += 1
    for i in range(MR):
        setcols(1,	   0,		 1-float(i)/MR, k)
        k += 1
makecolorwheel()


def sintel_compute_color(data_interlaced):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    data_u_in, data_v_in = np.split(data_interlaced, 2, axis=2)
    data_u_in = np.squeeze(data_u_in)
    data_v_in = np.squeeze(data_v_in)
    # pre-normalize (for some reason?)
    max_rad = np.max(np.sqrt(np.power(data_u_in, 2) + np.power(data_v_in, 2)))
    fx = data_u_in / max_rad
    fy = data_v_in / max_rad

    # now do the stuff done in computeColor()
    rad = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    a = np.arctan2(-fy, -fx) / np.pi
    fk = (a + 1.0) / 2.0 * (ncols-1)
    k0 = fk.astype(np.int32)
    k1 = ((k0 + 1) % ncols).astype(np.int32)
    f = fk - k0
    h, w = k0.shape
    col0 = colorwheel[k0.reshape(-1)].reshape([h, w, 3])
    col1 = colorwheel[k1.reshape(-1)].reshape([h, w, 3])
    col = (1 - f[..., np.newaxis]) * col0 + f[..., np.newaxis] * col1
    # col = col0

    col = 1 - rad[..., np.newaxis] * (1 - col)  # increase saturation with radius
    return col


if __name__ == "__main__":
    frame_number = 1
    training_set = "alley_1"
    data = read_flow_file("data/sintel/training", training_set, frame_number)
    rgb_new = sintel_compute_color(data)
    # plt.imshow(np.dstack([np.transpose(rgb_new, [1, 0, 2]), np.ones([rgb_new.shape[1], rgb_new.shape[0], 1])]))
    image_ref = mpimg.imread("data/sintel/training/flow_viz/" + training_set + ("/frame_%0.4i.png" % frame_number))
    # image_ref = imread("data/sintel/flow_code/C" + ("/frame_%0.4i_w.ppm" % frame_number))
    image_ref = image_ref.astype(np.float32) / 255.0

    print "average error:", np.average(rgb_new - image_ref.transpose([1,0,2]))

    ax1 = plt.subplot(221)
    ax1.imshow(rgb_new.transpose([1, 0, 2]))
    ax1.set_title("calculated image")

    ax2 = plt.subplot(222)
    ax2.imshow(image_ref)
    ax2.set_title("reference image (from given code)")

    ax3 = plt.subplot(223)
    ax3.imshow(np.abs(rgb_new.transpose([1,0,2]) - image_ref))
    ax3.set_title("error from calculated and reference")

    # rgb_old = old_compute_color(np.dstack([data_u, data_v]))
    # ax4 = plt.subplot(224)
    # ax4.imshow(rgb_new.transpose([1, 0, 2]))
    # ax4.set_title("non-vectorized image")

    # plt.gray()
    plt.tight_layout()
    plt.show()
    pass
