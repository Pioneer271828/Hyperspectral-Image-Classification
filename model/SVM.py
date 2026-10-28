import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as scio
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import spectral as spy
# 假设您的高光谱数据存储在X中，每行代表一个样本，每列代表一个特征
# 假设标签存储在y中，每个元素表示相应样本的类别
def split_data(gt_reshape, class_num, train_ratio, val_ratio, train_num, val_num, samples_type):
    train_index = []
    test_index = []
    val_index = []
    # np.random.seed(4)
    if samples_type == 'ratio':
        # class_num = 16 类
        for i in range(class_num):

            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            # print("Class ",i,":", samplesCount)
            train_num = np.ceil(samplesCount * train_ratio).astype('int32')
            val_num = np.ceil(samplesCount * val_ratio).astype('int32')
            np.random.shuffle(idx)
            train_index.append(idx[:train_num])
            val_index.append(idx[train_num:train_num+val_num])
            test_index.append(idx[train_num+val_num:])

    else:
        sample_num = train_num
        # class_num = 16 类
        for i in range(class_num):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            # print("Class ",i,":", samplesCount)  # 每一类的个数

            max_index = np.max(samplesCount) + 1
            np.random.shuffle(idx)
            if sample_num > max_index:
                sample_num = 15
            else:
                sample_num = train_num

            # 取出每个类别选择出的训练集
            train_index.append(idx[: sample_num])
            val_index.append(idx[sample_num : sample_num+sample_num])
            # test_index.append(idx[sample_num+class_num : ])
            test_index.append(idx)

    train_index = np.concatenate(train_index, axis=0)
    val_index = np.concatenate(val_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)
    return train_index, val_index, test_index

def get_label(gt_reshape, train_index, val_index, test_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_reshape[train_index[i]]

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_reshape[test_index[i]]

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_index)):
        val_samples_gt[val_index[i]] = gt_reshape[val_index[i]]

    return train_samples_gt, test_samples_gt, val_samples_gt
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

    # #paviaU
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

    v = spy.imshow(classes=numlabel.astype(np.int16), colors=color_map,fignum=fig.number)
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
data = scio.loadmat(r'D:\dataset\Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
data_gt = scio.loadmat(r'D:\dataset\Indian/Indian_pines_gt.mat')['indian_pines_gt']

#
# data = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\Pavia\PaviaU.mat')["paviaU"]
# data_gt = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\Pavia\PaviaU_gt.mat')["Data_gt"]
# samples_type="number"
samples_type="ratio"
class_num = np.max(data_gt)
height, width, bands = data.shape
gt_reshape = np.reshape(data_gt, [-1])
data_reshape=np.reshape(data, [height*width,bands])
train_num=30
val_num=30

train_ratio=0.01
val_ratio=0.01

train_index, val_index, test_index = split_data(gt_reshape,
                    class_num, train_ratio, val_ratio, train_num, val_num, samples_type)

X_train=data_reshape[train_index]
Y_train=gt_reshape[train_index]
X_test=data_reshape[test_index]
Y_test=gt_reshape[test_index]


# 创建SVM分类器
# clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel='linear')

# 训练模型
clf.fit(X_train, Y_train)

# 使用训练后的模型进行预测
y_pred = clf.predict(X_test)

# 计算准确度
accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 计算混淆矩阵
conf_matrix = confusion_matrix(Y_test, y_pred)

# 计算OA
oa = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

# 计算AA
class_count = np.sum(conf_matrix, axis=1)
aa = np.sum(np.diag(conf_matrix) / class_count) / len(class_count)

# 计算kappa系数
total = np.sum(conf_matrix)
pa = np.trace(conf_matrix) / total
pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total * total)
kappa = (pa - pe) / (1 - pe)

print(f'Overall Accuracy (OA): {oa}')
print(f'Average Accuracy (AA): {aa}')
print(f'Kappa Coefficient: {kappa}')

imag_pre=np.zeros_like(gt_reshape)
for i in range(len(y_pred)):
    imag_pre[test_index[i]]=y_pred[i]
imag_pre=np.reshape(imag_pre,[height,width])
name="SVM_IP_0.01_V2"
# name="SVM_pavia_lin_0.005_V2"
Draw_Classification_Map(imag_pre,name)
# Draw_Classification_Map(data_gt, "True_pavia")
# print(imag_pre.shape)
# visualize_hyperspectral_image(imag_pre)


print("新计算指标方式")
classification = classification_report(Y_test, y_pred, digits=4)
kappa = cohen_kappa_score(Y_test, y_pred)
print(classification,kappa)

