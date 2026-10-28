from skimage.segmentation import slic
# import scipy.io as scio
from . import visualization
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Z-score 归一化
def z_score_normalize(X):
    mean_value = np.mean(X)
    std_value = np.std(X)
    normalized_X = (X - mean_value) / std_value
    return normalized_X

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

def SILC_Image(image,label,num_segments ):
    print("######正在进行SLIC分割######")
    is_save = 0
    is_show_SILC_fenge = 0
    H, W, C = image.shape
    # 定义超像素分割的分块数
    print("SLIC分割前维度：",image.shape)
    # image=visualization.normalize_maxmin(image)
    # 进行SLIC超像素分割
    compactness=1
    segments = slic(image, n_segments=num_segments,max_num_iter=20,start_label=0,convert2lab=False, compactness=0.01)

    # np.save("./pavia/pavia_SLIC_Segments.npy",segments.astype(np.int32))
    # np.save("./hyper/hyper_SLIC_Segments3.npy",segments.astype(np.int32))
    # np.save("./multi/LaoYuHe_SLIC_Segments_2000_2.npy",segments.astype(np.int32))

    if is_show_SILC_fenge==1:
        # idx=[0,1,2]
        # image2=image[:,:,idx]
        # image2= visualization.normalize_maxmin(image)
        idx=[10,30,60]
        # image2=image2[:,:,idx]
        # plt.imshow(image2)
        # plt.show()
        # label2= visualization.normalize_maxmin(label)
        # plt.imshow(label2)
        # plt.show()
        # path="./multi/LaoYuHe_SLIC_multi_2000_22.svg"
        path="./result/Indian/Indian_SLIC"
        visualization.ShowSlicSegments(image[:, :, [0, 1, 2]], segments, path)

    n_segments = np.max(segments) + 1
    listA=np.arange(n_segments).reshape(-1,1)
    # 计算每个超像素块的特征
    segment_features = []
    for segment_label in np.unique(segments):
        # 获取当前超像素块的位置索引
        segment_indices = np.where(segments == segment_label)
        segment_indices_row, segment_indices_col = segment_indices

        # 计算超像素块的光谱特征，这里以像素点值的平均值作为光谱特征
        segment_spectrum = np.mean(image[segment_indices_row, segment_indices_col], axis=0)

        # 计算超像素块的位置特征，这里以中心点的位置坐标作为位置特征
        segment_center_row = np.mean(segment_indices_row)
        segment_center_col = np.mean(segment_indices_col)
        segment_center = segment_center_row, segment_center_col

        # 获取超像素块的真实标签，采用最大值投票法
        segment_label = np.argmax(np.bincount(label[segment_indices_row, segment_indices_col].flatten()))

        # 将光谱特征、位置特征和真实标签合并成一个特征向量
        segment_feature = np.concatenate((segment_spectrum, segment_center,[segment_label]))

        # 将超像素块的特征向量添加到列表中
        segment_features.append(segment_feature)

    # 将特征向量列表转换为数组，得到最终的分割结果
    segment_features = np.array(segment_features)
    n_segments = np.max(segments) + 1
    listA=np.arange(n_segments).reshape(-1,1)
    segment_features=np.hstack((segment_features,listA))


    if is_save==1:
        path1="./result/Indian/Indian_SLIC_features.npy"
        np.save(path1,segment_features.astype(np.int32))
        path2="./result/Indian/Indian_SLIC_segments.npy"
        np.save(path2, segments.astype(np.int32))
    # np.save("./hyper/LaoYUHe_SLIC_features_Hyper_UsingMultiSe.npy",segment_features.astype(np.int32))
    # np.save("./pavia/pavia_SLIC_features.npy",segment_features.astype(np.int32))
    # 输出分割结果，形状为10000*(C+2+1)
    print("SLIC分割前维度：", segment_features.shape)

    print("######SLIC分割结束######")
    return segment_features,segments

def Compute_Distance(x):
    start_time_of_CaDis = time.time()
    # Calculate pairwise Euclidean distances
    """
      Calculate the distance among each row of x
      :param x: N X D
                  N: the object number
                  D: Dimension of the feature
      :return: N X N distance matrix
      """
    x = np.asmatrix(x)  # 构建矩阵
    aa = np.sum(np.multiply(x, x), 1)  # 哈达玛乘积
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    # dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    end_time_of_CaDis = time.time()
    print("计算DIS用时{}s".format(end_time_of_CaDis - start_time_of_CaDis))

    # 保存最近邻居为txt

    # np.savetxt('./data_Indian/NP_Distance_4_17_Indian_spa.txt', dist_mat, fmt='%.4f')
    return dist_mat

def Select_Nearest_Neighbors(X,n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    knn_indices = nbrs.kneighbors(X, return_distance=False)
    return knn_indices

def CreateSimpleGraph(n,m):
    # Build a simple graph using pixel neighborhoods
    graph = np.zeros((n * m, n * m))
    for i in range(n):
        for j in range(m):
            center = i * m + j
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * m + j)  # Top
            if i < n - 1:
                neighbors.append((i + 1) * m + j)  # Bottom
            if j > 0:
                neighbors.append(i * m + (j - 1))  # Left
            if j < m - 1:
                neighbors.append(i * m + (j + 1))  # Right
            if i > 0 and j > 0:
                neighbors.append((i - 1) * m + (j - 1))  # Top-left
            if i > 0 and j < m - 1:
                neighbors.append((i - 1) * m + (j + 1))  # Top-right
            if i < n - 1 and j > 0:
                neighbors.append((i + 1) * m + (j - 1))  # Bottom-left
            if i < n - 1 and j < m - 1:
                neighbors.append((i + 1) * m + (j + 1))  # Bottom-right
            graph[center, neighbors] = 1
    np.fill_diagonal(graph, 1)
    # Visualize the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(graph[:500, :500], cmap='gray')
    return graph
def Generate_H_By_RandWork_distance2(X,idx_rand,p = 1.0,q=0.5,walk_length = 10,n_neighbors=20,):

    num_walks = 1
    walks = []
    distances = np.asarray(Compute_Distance(X))
    graph=CreateSimpleGraph(145,145)
    idx_rand=np.squeeze(idx_rand)
    graph=graph[idx_rand, :]
    graph=graph[:,idx_rand]
    for start_node in range(X.shape[0]):
        for walk_iter in range(num_walks):
            walk = [start_node]
            current_node = start_node
            for _ in range(walk_length - 1):
                neighbors = np.where(graph[current_node])[0]
                if len(walk) == 1:
                    # Bias towards immediately connected nodes
                    next_node = np.random.choice(neighbors)
                else:
                    # Bias towards farther away nodes
                    distances_to_current = distances[current_node]
                    distances_to_prev = distances[walk[-2]]
                    transition_probs = np.exp(-q * distances_to_current) * (distances_to_prev ** p)
                    transition_probs /= np.sum(transition_probs)
                    next_node = np.random.choice(neighbors,p=transition_probs[neighbors] / np.sum(transition_probs[neighbors]))
                walk.append(next_node)
                current_node = next_node
            if (start_node % 10 == 0):
                print("这是第{}个节点开始的随机游走序列：{}\n".format(start_node, walk))
            walks.append(walk)

    # Combine walks into hyperedges to create a hypergraph
    hypergraph = []
    for walk in walks:
        hyperedge = set()
        for i in range(walk_length):
            hyperedge.add(walk[i])
        # if (hyperedge.__len__() != 10):
        #     print("{}大小不同".format(walk))
        hypergraph.append(hyperedge)
    H = np.zeros((X.shape[0], X.shape[0]), dtype=int)
    # 根据listA中的集合元素在H数组中对应位置上的值设置为1
    for i, s in enumerate(hypergraph):
        for j in s:
            H[j][i] = 1
    # 保存H为txt

    # np.savetxt('./data_Indian/H_4_17_Indian_spa.txt',H, fmt='%d')
    return H

def MatrixDistance(Mx):
    '''
    Calculate the distance matrix.
    '''
    DisMatrix=np.zeros((Mx.shape[0],Mx.shape[0]))
    for i in range(Mx.shape[1]):
        col=Mx[:,[i]]
        len=Mx.shape[0]
        a=col**2
        A = a.repeat(len,axis=-1)
        B=col*col.T
        c=a.T
        C=c.repeat(len,axis=0)
        D=A+C-2*B
        DisMatrix+=D
    return DisMatrix

def takemaxinrow(mx,n):
    """
    take max n value in each row
    """
    temp=np.sort(mx,axis=1)
    limit=temp[:,-n].T
    # zero=np.zeros((1,mx.shape[1]))
    for i in range(mx.shape[0]):
        mx[i]=np.where(mx[i]>=limit[i],mx[i],0)
    return mx
def MultikernelMatrix(Mx,sigmalist):
    '''
    Multikernel Matrix
    '''
    print('Building Multikernel Matrix')
    Spatial=Mx
    SpatialDistance=MatrixDistance(Spatial)
    MultikernelMatrix=np.zeros((Mx.shape[0],Mx.shape[0],len(sigmalist)))
    for i in range (len(sigmalist)):
        SpatialGaussAdj=np.exp(-SpatialDistance/sigmalist[i])-np.eye(SpatialDistance.shape[0])
        SpatialGaussAdj=normalize_maxmin(SpatialGaussAdj)
        ADJMatrix=SpatialGaussAdj
        MultikernelMatrix[:,:,i]=((ADJMatrix)+(ADJMatrix).T)/2
    print('Multikernel Matrix Done')
    return MultikernelMatrix
def normalize(mx):
    """
    row-normalize matrix torch
    """
    rowsum = torch.sum(mx,1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv.mm(mx)
    return mx
def normalize_adj(adj):
    '''
    Nomalize the adjacency matrix
    '''
    sumrow=np.sum(adj,axis=1)
    normalizedadj=adj/sumrow[:,np.newaxis]
    return normalizedadj

def get_top_eight_indices(matrix):
    sorted_indices = np.argsort(matrix, axis=1)
    top_eight_indices = sorted_indices[:, -8:]
    return top_eight_indices

def Find_Nearest_Neighbors(distance, n_neighbors):
    # 获取距离矩阵的行数，即节点数量
    num_nodes = distance.shape[0]

    # 初始化一个二维数组用于存储每个节点的最近邻节点索引
    nearest_neighbors_indices = np.zeros((num_nodes, n_neighbors), dtype=int)

    for i in range(num_nodes):
        # 对于每个节点，获取其与所有其他节点的距离并按升序排序
        dist_to_other_nodes = distance[i, :]
        sorted_indices = np.argsort(dist_to_other_nodes)

        # 获取该节点的最近邻节点索引，并存储到nearest_neighbors_indices中
        nearest_neighbors_indices[i] = sorted_indices[1:n_neighbors+1]  # 排除自身，取最近的n_neighbors个邻居

    return nearest_neighbors_indices

def Generate_H_By_RandWork_distance(X,distances,p = 1,q=0.5,walk_length = 20,n_neighbors=100):#p = 1.0,q=0.5
    knn_indices=Find_Nearest_Neighbors(distances,n_neighbors)
    num_walks=1
    walks = []
    disnode=0
    start_time_of_CaDis = time.time()
    for start_node in range(X.shape[0]):
        for walk_iter in range(num_walks):
            walk = [start_node]
            current_node = start_node
            for _ in range(walk_length - 1):
                if len(walk) == 1:
                    # Bias towards immediately connected nodes
                    next_node = np.random.choice(knn_indices[current_node])
                    disnode=abs(start_node-next_node)
                else:
                    # Bias towards farther away nodes
                    distances_to_current = distances[current_node]
                    distances_to_prev = distances[walk[-2]]
                    #有问题
                    transition_probs = np.exp(-q * distances_to_current) * (distances_to_prev ** p)
                    transition_probs /= np.sum(transition_probs)
                    next_node = np.random.choice(X.shape[0], p=transition_probs)
                    disnode = abs(start_node - next_node)
                if disnode<1000:
                    walk.append(next_node)
                    current_node = next_node
                else:
                    walk.append(current_node)

        if(start_node%10==0):
            print("这是第{}个节点开始的随机游走序列：{}\n".format(start_node, walk))
        walks.append(walk)
    end_time_of_CaDis = time.time()
    print("生成随机序列消耗时间{}s".format(end_time_of_CaDis - start_time_of_CaDis))
    # Combine walks into hyperedges to create a hypergraph
    hypergraph = []
    for walk in walks:
        hyperedge = set()
        for i in range(walk_length):
            hyperedge.add(walk[i])
        # if (hyperedge.__len__() != 10):
        #     print("{}大小不同".format(walk))
        hypergraph.append(hyperedge)
    print(hyperedge)
    H=np.zeros((X.shape[0],X.shape[0]),dtype=int)
    # 根据listA中的集合元素在H数组中对应位置上的值设置为1
    for i, s in enumerate(hypergraph):
        for j in s:
            H[j][i] = 1
    #保存H为txt

    # np.savetxt('./data_Indian/H_4_17_Indian_spa.txt',H, fmt='%d')
    return H

def Generate_H_By_RandWork_distance3(A,p = 1,q=0.5,walk_length = 10,n_neighbors=20,num_walks=1):#p = 1.0,q=0.5

    num_nodes = A.shape[0]
    walks = []

    for start_node in range(num_nodes):
        for _ in range(num_walks):
            walk = [start_node]
            current_node = start_node
            prev_node = None
            for _ in range(walk_length - 1):
                neighbors = [neighbor for neighbor, weight in enumerate(A[current_node]) if weight > 0]
                if not neighbors:
                    break
                probabilities = []
                if len(walk) == 1:
                    # Bias towards immediately connected nodes
                    next_node = np.random.choice(neighbors)
                else:
                    for neighbor in neighbors:
                        if neighbor == prev_node:
                            probabilities.append(1.0 / p)
                        elif A[neighbor][prev_node] > 0:
                            probabilities.append(1.0)
                        else:
                            probabilities.append(1.0 / q)
                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                prev_node = current_node
                current_node = next_node
            walk.append(next_node)
        if (start_node % 10 == 0):
            print("这是第{}个节点开始的随机游走序列：{}\n".format(start_node, walk))
        walks.append(walk)
    # Combine walks into hyperedges to create a hypergraph
    hypergraph = []
    for walk in walks:
        hyperedge = set()
        for i in range(walk_length):
            hyperedge.add(walk[i])
        # if (hyperedge.__len__() != 10):
        #     print("{}大小不同".format(walk))
        hypergraph.append(hyperedge)
    print(hyperedge)
    H=np.zeros((A.shape[0],A.shape[0]),dtype=int)
    # 根据listA中的集合元素在H数组中对应位置上的值设置为1
    for i, s in enumerate(hypergraph):
        for j in s:
            H[j][i] = 1
    #保存H为txt

    # np.savetxt('./data_Indian/H_4_17_Indian_spa.txt',H, fmt='%d')
    return H


def Generate_H_By_RandWork(image_spe,image_spa,s2D_whole_spe, s2D_whole_spa):
    X_spe = image_spe
    X_spa = image_spa
    # X_spe = z_score_normalize(X_spe)
    # X_spe = normalize_maxmin(X_spe,axis=2)
    # X_spa = normalize_maxmin(X_spa)
    s2D_whole_spe=np.array(s2D_whole_spe)
    s2D_whole_spa=np.array(s2D_whole_spa)
    # H_spe = Generate_H_By_RandWork_distance2(X_spe,idx_rand)
    H_spe = Generate_H_By_RandWork_distance(X_spe, s2D_whole_spe)
    H_spa = Generate_H_By_RandWork_distance3(X_spa,s2D_whole_spa)




    return H_spe,H_spa


def lda_dimensionality_reduction(images, labels, n_components):
    """
    使用线性判别分析（LDA）进行图像数据的降维

    参数：
    images: 图像数据，形状为 (样本数, 图像高度, 图像宽度, 通道数)
    labels: 图像标签，形状为 (样本数,)
    n_components: 降维后的特征数

    返回值：
    降维后的数据，形状为 (样本数, n_components)
    """
    print("正在进行LDA降维")
    print("LDA前维度",images.shape)

    # 将图像数据转换为二维形式，每个样本是一个一维向量表示的图像像素
    image_height, image_width, num_channels = images.shape
    flattened_images = np.reshape(images, [image_height * image_width, num_channels])
    # 执行LDA降维
    flattened_labels=np.reshape(labels,[-1])
    idx = np.where(flattened_labels != 0)[0]
    x=flattened_images[idx]
    y=flattened_labels[idx]
    lda = LinearDiscriminantAnalysis()
    lda.fit(x, y - 1)
    X_lda =lda.transform(flattened_images)
    X_lda=np.reshape(X_lda,[image_height,image_width,-1])
    print("LDA降维结束")
    print("LDA后维度", X_lda.shape)
    return X_lda
