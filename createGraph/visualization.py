import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import spectral as spy

def normalize_maxmin(Mx, axis=2):
    '''
    Normalize the matrix Mx by max-min normalization.
    axis=0: normalize each row
    axis=1: normalize each column
    axis=2: normalize the whole matrix
    '''
    Mx_min = Mx.min()
    if Mx_min < 0:
        Mx +=abs(Mx_min)
        Mx_min = Mx.min()

    if axis == 1:
        M_min = np.amin(Mx, axis=1)
        M_max = np.amax(Mx, axis=1)
        for i in range(Mx.shape[1]):
            Mx[:, i] = (Mx[:, i] - M_min) / (M_max - M_min)
    elif axis == 0:
        M_min = np.amin(Mx, axis=0)
        M_max = np.amax(Mx, axis=0)
        for i in range(Mx.shape[0]):
            Mx[i, :] = (Mx[i, :] - M_min) / (M_max - M_min)
    elif axis == 2:
        M_min = np.amin(Mx)
        M_max = np.amax(Mx)
        Mx = (Mx - M_min) / (M_max - M_min)
    else:
        print('Error')
        return None
    return Mx
def visualize_hyperspectral_image(image):
    # 定义类别颜色映射
    color_map = {
        0: (0, 0, 0),       # 背景颜色为黑色
        1: (70, 130, 180),  # 类别1的颜色（SteelBlue）
        2: (70, 70, 70),    # 类别2的颜色（DarkGray）
        3: (128, 0, 128),   # 类别3的颜色（Purple）
        4: (0, 128, 0),     # 类别4的颜色（Green）
        5: (220, 20, 60),   # 类别5的颜色（Crimson）
        6: (0, 0, 255),     # 类别6的颜色（Blue）
        7: (128, 128, 128), # 类别7的颜色（Gray）
        8: (255, 69, 0),    # 类别8的颜色（OrangeRed）
        9: (255, 255, 0),   # 类别9的颜色（Yellow）
        10: (0, 255, 0),    # 类别10的颜色（Lime）
        11: (255, 0, 255),  # 类别11的颜色（Magenta）
        12: (128, 0, 0),    # 类别12的颜色（Maroon）
        13: (0, 255, 255),  # 类别13的颜色（Cyan）
        14: (255, 255, 255),  # 类别14的颜色（White）
        15: (128, 255, 255),
        16: (128, 255, 128)
    }

    # 创建RGB图像，初始为全黑
    h, w = image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 逐像素填充颜色
    for i in range(h):
        for j in range(w):
            label = image[i, j]
            rgb_image[i, j] = color_map[label]

    # 显示图像
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

def ShowSlicSegments(image,segments,path):
    plt.imshow(mark_boundaries(image,segments,outline_color=(128,0,0),mode="subpixel"))
    # plt.savefig(path, dpi=300,format="svg")
    plt.show()
def Draw_Classification_MapWHU(label, name: str, scale: float = 4.0, dpi: int = 400):
    # WHU
    colors = np.array([
        [0, 0, 0],  # 背景色（黑色）
        [222, 0, 0],  # 类别 1 - 印度红松（Red Pine）
        [205, 133, 63],  # 类别 2
        [64, 128, 33],  # 类别 3
        [255, 255, 0],  # 类别 4
        [255, 0, 255],  # 类别 5
        [0, 128, 255],  # 类别 6
        [128, 0, 0],  # 类别 7
        [0, 188, 0],  # 类别 8
        [255, 192, 203],  # 类别 9
        [192, 192, 192],  # 类别 10
        [139, 69, 19],  # 类别 11
        [0, 100, 0],  # 类别 12
        [128, 128, 0],  # 类别 13
        [128, 0, 128],  # 类别 14#[128, 0, 128],
        [0, 0, 128],  # 类别 15
        [128, 128, 128],  # 类别 16
        [155, 205, 0],  # 类别
        [55, 90, 250],  # 类别
        [60, 128, 25],
    ])

    visualization = colors[label]
    plt.imshow(visualization)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name + '.png', format='png',  dpi=400, bbox_inches='tight', pad_inches=0)

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)

    #WHU
    # 
    color_map = np.array([
        [0, 0, 0],  # 背景色（黑色）
        [222, 0, 0],  # 类别 1 - 印度红松（Red Pine）
        [205, 133, 63],  # 类别 2
        [64, 128, 33],  # 类别 3
        [255, 255, 0],  # 类别 4
        [255, 0, 255],  # 类别 5
        [0, 128, 255],  # 类别 6
        [128, 0, 0],  # 类别 7
        [0, 188, 0],  # 类别 8
        [255, 192, 203],  # 类别 9
        [192, 192, 192],  # 类别 10
        [139, 69, 19],  # 类别 11
        [0, 100, 0],  # 类别 12
        [128, 128, 0],  # 类别 13
        [128, 0, 128],  # 类别 14
        [0, 0, 128],  # 类别 15
        [128, 128, 128],  # 类别 16
    ])

    # XuZhou
    # color_map = np.array([
    #     [0, 0, 0],  # 背景色（黑色）
    #     [222, 0, 0],  # 类别 1 - 红色
    #     (205, 133, 63),  # 类别 2 - 绿色
    #     [64, 128, 33],  # 类别 3 - 蓝色
    #     [255, 255, 0],  # 类别 4 - 黄色
    #     [0, 255, 128],  # 类别 5 - 洋红
    #     [0, 128, 255],  # 类别 6 - 青色
    #     [128, 0, 0],  # 类别 7 - 深红
    #     [0, 188, 0],  # 类别 8 - 深绿
    #     [255, 192, 203],  # 类别 9 - 深蓝
    # ])
    # paviaU
    # color_map = np.array([
    #     [0, 0, 0],  # 背景色（黑色）
    #     [255, 128, 0],  # 类别 1 - 橙色
    #     [0, 128, 255],  # 类别 2 - 天蓝色
    #     [255, 0, 255],  # 类别 3 - 紫色
    #     [255, 255, 0],  # 类别 4 - 黄色
    #     [0, 255, 128],  # 类别 5 - 青绿色
    #     [128, 0, 128],  # 类别 6 - 紫红色
    #     [0, 0, 255],  # 类别 7 - 深蓝色
    #     [0, 128, 0],  # 类别 8 - 深绿色
    #     [255, 192, 203],  # 类别 7 - 深红
    # ])
    #WHUL
    # color_map = np.array([
    #     [0, 0, 0],  # 背景色（黑色）
    #     [222, 0, 0],  # 类别 1 - 红色
    #     (205, 133, 63),  # 类别 2 - 绿色
    #     [64, 128, 33],  # 类别 3 - 蓝色
    #     [255, 255, 0],  # 类别 4 - 黄色
    #     [255, 0, 255],  # 类别 5 - 洋红
    #     [0, 128, 255],  # 类别 6 - 青色
    #     [128, 0, 0],  # 类别 7 - 深红
    #     [0, 188, 0],  # 类别 8 - 深绿
    #     [255, 192, 203],  # 类别 9 - 深蓝
    # ])

    # Create a mask for pixels with labels
    labeled_pixels_mask = numlabel >= 0

    # Set pixels without labels to a background color (e.g., white)
    numlabel[~labeled_pixels_mask] = -1  # Set to a value that represents the background

    v = spy.imshow(classes=numlabel.astype(np.int16),colors=color_map, fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass