import time
import sys
import numpy as np
import matplotlib.pyplot as plt

class ProcessBar:
    __max__ = 0
    __start_time__ = time.time()

    def reset(self, max_iter):
        self.__max__ = max_iter
        self.__start_time__ = time.time()

    def show(self, i, msg = ""):
        percents = int(100 * i / self.__max__)
        empty_percents = 100 - percents
        line_str = "[{}>{}], {}/{}, {}%, {:.2f}s {}".format(percents // 2 * "=",
            empty_percents // 2 * " ", i, self.__max__, percents, time.time() - self.__start_time__, msg)
        sys.stdout.write("\r" + line_str)
    
    def summary(self, epoch, msg):
        print()
        print("EPOCH: {} {}".format(epoch, msg))

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    cp_matrix = np.copy(matrix)
    cp_matrix[cp_matrix < 1] = np.nan
    mins = np.nanpercentile(cp_matrix, 1, axis=0)
    maxs = np.nanpercentile(cp_matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

def visual_image(im_2015, im_2017, mask_1, mask_2, 
                index_X = 512, index_Y = 512,
                width = 512, height = 512, is_scale = True):
    plt.rcParams["figure.figsize"] = (20, 20)
    row_start = index_X
    row_end = index_X + height
    col_start = index_Y
    col_end = index_Y + width

    if is_scale:
        show_2015 = scale_percentile(im_2015[index_X: index_X + height, index_Y: index_Y + width, :3])
        show_2017 = scale_percentile(im_2017[index_X: index_X + height, index_Y: index_Y + width, :3])
    else:
        show_2015 = im_2015[index_X: index_X + height, index_Y: index_Y + width, :3]
        show_2017 = im_2017[index_X: index_X + height, index_Y: index_Y + width, :3]

    if len(mask_1.shape) == 3:
        mask_1_show = mask_1[row_start: row_end, col_start: col_end, 0]
    else:
        mask_1_show = mask_1[row_start: row_end, col_start: col_end]

    if len(mask_2.shape) == 3:
        mask_2_show = mask_2[row_start: row_end, col_start: col_end, 0]
    else:
        mask_2_show = mask_2[row_start: row_end, col_start: col_end]

    plt.subplot(2, 2, 1)
    plt.imshow(show_2015)
    plt.title("2015")
    plt.subplot(2, 2, 2)
    plt.imshow(mask_1_show)
    plt.title("Mask 1")
    plt.subplot(2, 2, 3)
    plt.imshow(show_2017)
    plt.title("2017")
    plt.subplot(2, 2, 4)
    plt.imshow(mask_2_show)
    plt.title("Mask 2")

def pad_images_list(X_image, y_image, stride, width, channel=8):
    pad_height = X_image.shape[0] // stride * stride + stride
    pad_width  = X_image.shape[1] // stride * stride + stride
    X_train_pad = np.zeros([pad_height, pad_width, channel])
    y_train_pad = np.zeros([pad_height, pad_width, 1])
    top_height = (pad_height - X_image.shape[0]) // 2
    left_width = (pad_width - X_image.shape[1]) // 2
    
    X_train_pad[top_height: -top_height, left_width: -left_width] = X_image
    y_train_pad[top_height: -top_height, left_width: -left_width] = y_image
    X_images = np.zeros([pad_height // stride - 1, pad_width // stride - 1, width, width, channel])
    y_images = np.zeros([pad_height // stride - 1, pad_width // stride - 1, width, width, 1])
    pb = ProcessBar()
    pb.reset(max_iter = pad_height * pad_width // stride // stride)
    for i in range(0, X_train_pad.shape[0] - stride, stride):
        for j in range(0, y_train_pad.shape[1] - stride, stride):
            X_images[i // stride, j // stride] = X_train_pad[i: i + width, j: j + width]
            y_images[i // stride, j // stride] = y_train_pad[i: i + width, j: j + width]
            pb.show(i * pad_width // stride // stride + j // stride)
    return X_images, y_images, X_train_pad, y_train_pad


def pad_images_list_different(X_image, y_image, stride_x, stride_y, height, width):
    pad_height = X_image.shape[0] // stride_x * stride_x + stride_x
    pad_width  = X_image.shape[1] // stride_y * stride_y + stride_y
    X_train_pad = np.zeros([pad_height, pad_width, 8])
    y_train_pad = np.zeros([pad_height, pad_width, 1])
    top_height = (pad_height - X_image.shape[0]) // 2
    left_width = (pad_width - X_image.shape[1]) // 2
    
    X_train_pad[top_height: -top_height, left_width: -left_width] = X_image
    y_train_pad[top_height: -top_height, left_width: -left_width] = y_image
    X_images = np.zeros([pad_height // stride_x - 1, pad_width // stride_x - 1, height, height, 8])
    y_images = np.zeros([pad_height // stride_y - 1, pad_width // stride_y - 1, width, width, 1])
    pb = ProcessBar()
    pb.reset(max_iter = pad_height * pad_width // stride_x // stride_y)
    for i in range(0, X_train_pad.shape[0] - stride_x, stride_x):
        for j in range(0, y_train_pad.shape[1] - stride_y, stride_y):
            X_images[i // stride_x, j // stride_y] = X_train_pad[i: i + height, j: j + width]
            y_images[i // stride_x, j // stride_y] = y_train_pad[i: i + height, j: j + width]
            pb.show(i * pad_width // stride_x // stride_y + j // stride_y)
    return X_images, y_images, X_train_pad, y_train_pad


def make_metric(y_pred, y_true):
    y_pred = y_pred.reshape(y_pred.shape[:2])
    y_true = y_true.reshape(y_true.shape[:2])
    answer_cp = np.copy(y_true)
    answer_cp[y_true == 2] = 0
    answer_cp[y_true == 1] = 1
    tp = np.sum(answer_cp * y_pred)
    recall = tp / np.sum(answer_cp)
    answer_cp[y_true == 2] = 1
    answer_cp[y_true == 1] = 1
    precision = tp / np.sum(y_pred * answer_cp)
    return recall, precision, 2 / (1 / recall + 1 / precision)