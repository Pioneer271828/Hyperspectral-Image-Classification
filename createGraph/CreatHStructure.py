import numpy as np
from . import utils2
import random
import networkx as nx
from sklearn.metrics.pairwise import pairwise_kernels
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
def MultikernelMatrix(Mx,sigmalist,weights):
    '''
    Multikernel Matrix
    '''
    print('Building Multikernel Matrix')
    Spatial=Mx
    SpatialDistance=MatrixDistance(Spatial)
    MultikernelMatrix=np.zeros((Mx.shape[0],Mx.shape[0]))
    for i in range (len(sigmalist)):
        # SpatialGaussAdj=np.exp(-SpatialDistance/sigmalist[i])-np.eye(SpatialDistance.shape[0])
        SpatialGaussAdj=np.exp(-SpatialDistance/sigmalist[i])

        # SpatialGaussAdj=normalize_maxmin(SpatialGaussAdj)

        ADJMatrix=((SpatialGaussAdj)+(SpatialGaussAdj).T)/2

        MultikernelMatrix+=weights[i] *ADJMatrix
    print('Multikernel Matrix Done')



    return MultikernelMatrix
def Mat_dis(x):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.asmatrix(x)   #构建矩阵
    aa = np.sum(np.multiply(x, x), 1)   #哈达玛乘积
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    #dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)

    return dist_mat

def Mat_dis_s2(sp):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    dist= Mat_dis(sp) / sp.shape[1]
    return dist

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    A = np.mean(dis_mat)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 1.0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        #avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                # H[node_idx, center_idx] = np.exp(- sig * dis_vec[0, node_idx] / A )
                #print(H[node_idx, center_idx])
                H[node_idx, center_idx] = dis_vec[0, node_idx]
            else:
                H[node_idx, center_idx] = 1.0
    return H

def Generate_H_By_RandWork_distance3(A,Top_nodes,p,q):
    walk_length = 10
    num_walks = 2
    num_nodes = A.shape[0]
    walks = []
    for start_node in Top_nodes:
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

                walk.append(next_node)
                prev_node = current_node
                current_node = next_node

            walks.append(walk)

    # Combine walks into hyperedges to create a hypergraph
    hypergraph = []
    for walk in walks:
        hyperedge = set(walk)
        hypergraph.append(hyperedge)

    H = np.zeros((num_nodes, len(hypergraph)), dtype=int)

    for i, s in enumerate(hypergraph):
        for j in s:
            H[j][i] = 1

    return H

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

def Generate_H_By_RandWork_distance_spa(X,distances,p = 0.5,q=1,walk_length =10,n_neighbors=10):#p = 1.0,q=0.5
    distances=np.array(distances)
    knn_indices=Find_Nearest_Neighbors(distances,10)
    num_walks=2
    walks = []
    disnode=0

    for start_node in range(X.shape[0]):
        for walk_iter in range(num_walks):
            walk = [start_node]
            current_node = start_node
            for _ in range(walk_length - 1):
                if len(walk) == 1:
                    # Bias towards immediately connected nodes
                    next_node = np.random.choice(knn_indices[current_node])
                else:
                    # Bias towards farther away nodes
                    distances_to_current = distances[current_node]
                    distances_to_prev = distances[walk[-2]]
                    #有问题
                    transition_probs = np.exp(-q * distances_to_current) * (distances_to_prev ** p)
                    transition_probs /= np.sum(transition_probs)
                    next_node = np.random.choice(X.shape[0], p=transition_probs)


                walk.append(next_node)
                current_node = next_node

        # if (start_node % 500 == 0 ):
        #     print("这是第{}个节点开始的随机游走序列：{}\n".format(start_node, walk))
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
    H=np.zeros((X.shape[0],X.shape[0]),dtype=int)
    # 根据listA中的集合元素在H数组中对应位置上的值设置为1
    for i, s in enumerate(hypergraph):
        for j in s:
            H[j][i] = 1
    #保存H为txt

    # np.savetxt('./data_Indian/H_4_17_Indian_spa.txt',H, fmt='%d')
    return H

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H, dtype=float)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    print(np.where(DE <= 0))
    invDE = np.asmatrix(np.diag(np.power(DE, -1)))
    DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))
    H = np.asmatrix(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        W = np.asmatrix(np.diag(W))
        G = DV2 * H * W * invDE * HT * DV2
        return G

def generate_Graph(H):
    """
    generate graph G with H
    :param H:
    :return:
    """
    H = np.maximum(H,H.T)
    D = np.sum(H, axis=1)
    D2 = np.asmatrix(np.diag(np.power(D, -0.5)))
    G = D2 * H * D2

    return G
def construct_similarity_to_adjacency(similarity_scores, K):
    num_nodes = similarity_scores.shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    for i in range(num_nodes):
        # 对于每个节点，找到与之最相关的K个节点的索引
        top_k_indices = np.argsort(similarity_scores[i])[-K:]

        for j in top_k_indices:
            # 使用相似性分数作为边的权重
            adjacency_matrix[i, j] = 1

    return adjacency_matrix
def construct_s2d_to_adjacency(similarity_scores, K):
    num_nodes = similarity_scores.shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    similarity_scores=np.array(similarity_scores)
    for i in range(num_nodes):
        # 对于每个节点，找到与之最相关的K个节点的索引
        top_k_indices = np.argsort(similarity_scores[i])[:K]

        for j in top_k_indices:
            # 使用相似性分数作为边的权重
            adjacency_matrix[i, j] = 1

    return adjacency_matrix
def createH(image_spe,image_spa,segments):

    print("#####正在构建超图#####")
    # visualize_hyperspectral_image(gt)
     # 1:KNN 2:randwork
    randwork=1
    gammas = [0.1, 0.5, 1, 1.5, 2]
    weights = [0.1, 0.35, 0.35, 0.1, 0.1]
    using_multikern =1
    s2D=0
    image_spe = normalize_maxmin(image_spe)
    image_spa = normalize_maxmin(image_spa)
    image_spe=np.concatenate((image_spe,image_spa),axis=1)
    if using_multikern == 1:
        # 计算高斯核矩阵
        gaussian_kernel_matrix1 = pairwise_kernels(image_spe, metric='rbf', gamma=1.0 / (2 * gammas[0] ** 2))
        gaussian_kernel_matrix2 = pairwise_kernels(image_spe, metric='rbf', gamma=1.0 / (2 * gammas[1] ** 2))
        gaussian_kernel_matrix3 = pairwise_kernels(image_spe, metric='rbf', gamma=1.0 / (2 * gammas[2] ** 2))
        gaussian_kernel_matrix4 = pairwise_kernels(image_spe, metric='rbf', gamma=1.0 / (2 * gammas[3] ** 2))
        gaussian_kernel_matrix5 = pairwise_kernels(image_spe, metric='rbf', gamma=1.0 / (2 * gammas[4] ** 2))

        multikernel_matrix = weights[0] * gaussian_kernel_matrix1 + weights[1] * gaussian_kernel_matrix2+weights[2] * gaussian_kernel_matrix3+\
        weights[3] * gaussian_kernel_matrix4+weights[4] * gaussian_kernel_matrix5

        adjacency=construct_similarity_to_adjacency(multikernel_matrix,K=10)
        # s2D_whole_spa = Mat_dis_s2(image_spa)
    if s2D==1:
        s2D_whole_spe = Mat_dis_s2(image_spe)
        H_whole_spe = construct_H_with_KNN_from_distance(s2D_whole_spe,10)


    if randwork == 1:
        #randwork
        G = nx.Graph(adjacency)
        K=int(np.ceil(0.1*adjacency.shape[0]))
        # 计算度中心性
        degree_centrality = nx.degree_centrality(G)
        # 选择度中心性最高的K个节点
        sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        top_k1_nodes = [node for node, _ in sorted_degree_centrality[:K]]
        # Identify the K1 nodes with the highest degree centrality

        # Step 1: Calculate betweenness centrality for each node

        betweenness_centrality = nx.betweenness_centrality(G)

        # Step 2: Select the top K2 nodes with the highest betweenness centrality
        top_K2_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
        top_K2_nodes=top_K2_nodes[:K]


        # H_C= Generate_H_By_RandWork_distance3(adjacency,top_k1_nodes,p=0.8,q=1.22)
        # H_S= Generate_H_By_RandWork_distance3(adjacency,top_K2_nodes,p=1.2,q=0.8)

        H_C = Generate_H_By_RandWork_distance3(adjacency, top_k1_nodes, p=0.5, q=2)
        H_S = Generate_H_By_RandWork_distance3(adjacency, top_K2_nodes, p=2, q=0.5)
        # H_whole_spa = Generate_H_By_RandWork_distance_spa(image_spa,s2D_whole_spa)


    print("#####构建超图结束#####")
    print("#####准备训练#####")

    #PV
    # H_whole=0.7*H_whole_spa+0.3*H_whole_spe
    #only HGCN
    # H_whole=H_whole_spe
    H_whole=adjacency.T
    H_whole = np.concatenate((H_whole, 0.5*H_C,0.5*H_S), axis=1)
    AHy_whole = np.array(_generate_G_from_H(H_whole))
    return AHy_whole

