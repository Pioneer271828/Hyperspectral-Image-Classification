import torch
import numpy as np
from sklearn import metrics


def compute_loss(
        network_output: torch.Tensor,
        train_samples_gt_onehot: torch.Tensor,
        train_label_mask: torch.Tensor,
        class_balance: bool = False,
        eps: float = 1e-8):
    real_labels = train_samples_gt_onehot
    probs = network_output.clamp(min=eps, max=1.0)
    we = -torch.mul(real_labels, torch.log(probs))

    if class_balance:
        class_counts = torch.sum(torch.mul(real_labels, train_label_mask), dim=0)
        valid_classes = class_counts > 0
        valid_class_count = valid_classes.float().sum().clamp_min(1.0)
        total_count = class_counts.sum().clamp_min(1.0)
        class_weights = torch.zeros_like(class_counts)
        class_weights[valid_classes] = total_count / (
            valid_class_count * class_counts[valid_classes].clamp_min(1.0)
        )
        we = we * class_weights.view(1, -1)

    we = torch.mul(we, train_label_mask)
    if train_label_mask.dim() == 2:
        valid_count = train_label_mask[:, 0].sum().clamp_min(1.0)
    else:
        valid_count = train_label_mask.sum().clamp_min(1.0)
    pool_cross_entropy = torch.sum(we) / valid_count
    return pool_cross_entropy

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()          # 有效标签的个数
        correct_prediction = torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA= correct_prediction.cpu() / available_label_count
        return OA

def evaluate_performance2(network_output,train_samples_gt,train_samples_gt_onehot,zeros, class_count,m,n,require_AA_KPP=True,printFlag=True):
    if False==require_AA_KPP:
        with torch.no_grad():
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count

            return OA
    else:
        with torch.no_grad():
            #计算OA
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            OA=OA.cpu().numpy()

            # 计算AA
            zero_vector = np.zeros([class_count])
            output_data=network_output.cpu().numpy()
            train_samples_gt=train_samples_gt.cpu().numpy()
            train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()

            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            # idx = idx + train_samples_gt
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)
            Test_GT = np.reshape(train_samples_gt, [m, n])  # 测试样本图
            # 计算KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # 输出
            if 1:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)
            return OA,test_AA,test_kpp,test_AC_list

