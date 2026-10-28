import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import torch
import time
from model import SAHGCN,GCN,F2HGNN,VCASAHGCN,utils
from model.interaction_config import add_interaction_arguments, interaction_kwargs

from loadData import split_data,data_read,data_reader
from createGraph import rdSLIC, create_graph
from createGraph.visualization import Draw_Classification_Map,Draw_Classification_MapWHU
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def make_progress(iterable, enabled=True, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def log_message(message, progress_enabled=True):
    if progress_enabled and tqdm is not None:
        tqdm.write(str(message))
    else:
        print(message)


def format_float_tag(value):
    return ("{:.4g}".format(value)).replace("-", "m").replace(".", "p")


parser = argparse.ArgumentParser(description='CSGFNet_main')
add_interaction_arguments(parser, prefix="vca-")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--no-wavelength-encoder', action='store_true')
parser.add_argument('--drop-band-ratio', type=float, default=0.0)
parser.add_argument('--drop-band-seed', type=int, default=2026)
parser.add_argument('--train-drop-band-ratio', type=float, default=0.0)
parser.add_argument('--train-drop-band-seed', type=int, default=2026)
parser.add_argument('--train-band-mixup-ratio', type=float, default=0.0)
parser.add_argument('--train-band-mixup-prob', type=float, default=1.0)
parser.add_argument('--train-band-mixup-alpha', type=float, default=0.4)
parser.add_argument('--train-band-mixup-neighbors', type=int, default=1)
parser.add_argument('--use-vca', action='store_true')
parser.add_argument('--vca-latent-channels', type=int, default=128)
parser.add_argument('--vca-embed-dim', type=int, default=64)
parser.add_argument('--vca-locality-sigma', type=float, default=0.08)
parser.add_argument('--vca-locality-window', type=float, default=0.16)
parser.add_argument('--vca-adapter-attention-mode', type=str, default='wavelength',
                    choices=['wavelength', 'index', 'uniform'])
parser.add_argument('--vca-no-wavelength-attention', action='store_true')
parser.add_argument('--vca-wavelength-min', type=float, default=400.0)
parser.add_argument('--vca-wavelength-max', type=float, default=2500.0)
parser.add_argument('--vca-graph-mode', type=str, default='static', choices=['static', 'learned'])
parser.add_argument('--vca-learned-assignment-dim', type=int, default=32)
parser.add_argument('--vca-learned-assignment-radius', type=int, default=1)
parser.add_argument('--vca-learned-graph-radius', type=int, default=1)
parser.add_argument('--vca-learned-assignment-temperature', type=float, default=1.0)
parser.add_argument('--vca-learned-assignment-spatial-sigma', type=float, default=1.0)
parser.add_argument('--vca-learned-graph-spatial-sigma', type=float, default=1.5)
parser.add_argument('--vca-graph-compactness-weight', type=float, default=0.01)
parser.add_argument('--vca-graph-entropy-weight', type=float, default=0.001)
parser.add_argument('--vca-graph-balance-weight', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--train-runs', type=int, default=3)
parser.add_argument('--seeds', type=str, default=None)
parser.add_argument('--samples-type', type=str, default='ratio', choices=['ratio', 'number'])
parser.add_argument('--train-num', type=int, default=30)
parser.add_argument('--val-num', type=int, default=10)
parser.add_argument('--train-ratio', type=float, default=0.01)
parser.add_argument('--val-ratio', type=float, default=0.001)
parser.add_argument('--superpixel-scale', type=int, default=None)
parser.add_argument('--no-progress', action='store_true')
parser.add_argument('--no-save-maps', action='store_true')
parser.add_argument('--metrics-json', type=str, default=None)
parser.add_argument('--experiment-name', type=str, default=None)
args = parser.parse_args()
vca_interaction_options = interaction_kwargs(args, prefix="vca_")
if not args.use_vca and (vca_interaction_options["interaction_mode"] != "none" or vca_interaction_options["graph_update_mode"] != "none"):
    parser.error("Interaction and graph update modules require --use-vca")
if args.vca_no_wavelength_attention:
    args.vca_adapter_attention_mode = 'index'
if args.vca_learned_assignment_dim <= 0:
    raise ValueError('--vca-learned-assignment-dim must be positive')
if args.vca_learned_assignment_radius < 0 or args.vca_learned_graph_radius < 0:
    raise ValueError('learned graph radii must be non-negative')
if args.vca_learned_assignment_temperature <= 0:
    raise ValueError('--vca-learned-assignment-temperature must be positive')
if args.vca_learned_assignment_spatial_sigma <= 0 or args.vca_learned_graph_spatial_sigma <= 0:
    raise ValueError('learned graph spatial sigmas must be positive')
if min(args.vca_graph_compactness_weight, args.vca_graph_entropy_weight, args.vca_graph_balance_weight) < 0:
    raise ValueError('graph regularization weights must be non-negative')
# 指定使用顯卡5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)  # 输出 PyTorch 版本
print(torch.version.cuda)  # 输出 CUDA 版本
print("當前使用的顯卡編號：")
# print(device)
print(torch.cuda.device_count())  
print(torch.cuda.is_available()) 
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


def make_band_mixup_input(
        base_input,
        train_samples_gt,
        height,
        width,
        spectral_bands,
        wavelengths,
        mixup_ratio,
        mixup_alpha,
        neighbor_count,
        mixup_prob=1.0,
        available_band_indices=None):
    if mixup_ratio <= 0 or mixup_prob <= 0 or spectral_bands < 2:
        return base_input

    device = base_input.device
    if mixup_prob < 1.0 and float(torch.rand((), device=device).item()) >= mixup_prob:
        return base_input

    if available_band_indices is None:
        available = torch.arange(spectral_bands, device=device, dtype=torch.long)
    else:
        available = torch.as_tensor(available_band_indices, device=device, dtype=torch.long).reshape(-1)
        available = available[(available >= 0) & (available < spectral_bands)]
        available = torch.unique(available, sorted=True)
    if available.numel() < 2:
        return base_input

    mix_count = int(np.ceil(float(available.numel()) * mixup_ratio))
    mix_count = min(max(mix_count, 1), int(available.numel()))
    neighbor_count = min(max(int(neighbor_count), 1), int(available.numel()) - 1)

    if wavelengths is None:
        band_position = torch.arange(spectral_bands, device=device, dtype=base_input.dtype)
    else:
        band_position = torch.as_tensor(wavelengths, device=device, dtype=base_input.dtype).reshape(-1)
        if band_position.numel() != spectral_bands:
            band_position = torch.arange(spectral_bands, device=device, dtype=base_input.dtype)

    available_position = band_position[available]
    distance = torch.abs(available_position[:, None] - available_position[None, :])
    distance.fill_diagonal_(float("inf"))
    nearest_positions = torch.topk(distance, k=neighbor_count, largest=False).indices

    selected_positions = torch.randperm(int(available.numel()), device=device)[:mix_count]
    target_bands = available[selected_positions]
    neighbor_candidates = nearest_positions[selected_positions]
    neighbor_choices = torch.randint(0, neighbor_count, (mix_count,), device=device)
    source_positions = neighbor_candidates[torch.arange(mix_count, device=device), neighbor_choices]
    source_bands = available[source_positions]

    train_pixel_idx = (train_samples_gt != 0).nonzero(as_tuple=True)[0]
    if train_pixel_idx.numel() == 0:
        return base_input

    x = base_input.clone()
    x_flat = x.reshape(height * width, -1)
    beta = torch.distributions.Beta(
        torch.tensor(mixup_alpha, device=device, dtype=base_input.dtype),
        torch.tensor(mixup_alpha, device=device, dtype=base_input.dtype),
    )
    lam = beta.sample([mix_count]).to(device=device, dtype=base_input.dtype)
    lam = torch.maximum(lam, 1.0 - lam)

    target_values = x_flat[train_pixel_idx[:, None], target_bands[None, :]].clone()
    source_values = x_flat[train_pixel_idx[:, None], source_bands[None, :]]
    x_flat[train_pixel_idx[:, None], target_bands[None, :]] = (
        lam.view(1, -1) * target_values + (1.0 - lam).view(1, -1) * source_values
    )
    return x
print("CSFGNet runing")

# data_set_name="Indian"
data_set_name="paviaU"
# data_set_name="HoustonU"
# data_set_name="DC"
# data_set_name="xuzhou"
#data_set_name="WHU"
if args.dataset is not None:
    data_set_name = args.dataset
print(data_set_name)

# load data
print("loda_data now")
data,data_gt=data_read.readData(data_set_name)
print(data.shape)
if args.drop_band_ratio > 0 and args.train_drop_band_ratio > 0:
    raise ValueError("Use either --drop-band-ratio or --train-drop-band-ratio, not both.")
if not 0 <= args.train_band_mixup_ratio <= 1:
    raise ValueError("--train-band-mixup-ratio must be in [0, 1]")
if not 0 <= args.train_band_mixup_prob <= 1:
    raise ValueError("--train-band-mixup-prob must be in [0, 1]")
if args.train_band_mixup_ratio > 0 and args.train_band_mixup_alpha <= 0:
    raise ValueError("--train-band-mixup-alpha must be positive when band mixup is enabled")
if args.train_band_mixup_neighbors < 1:
    raise ValueError("--train-band-mixup-neighbors must be positive")
original_bands = data.shape[2]
original_wavelengths, original_fwhm = data_read.readWavelengths(data_set_name, original_bands)
kept_band_indices = np.arange(original_bands)
dropped_band_indices = np.array([], dtype=np.int64)
if args.drop_band_ratio > 0:
    if not 0 < args.drop_band_ratio < 1:
        raise ValueError("--drop-band-ratio must be in (0, 1)")
    rng = np.random.default_rng(args.drop_band_seed)
    drop_count = int(np.floor(original_bands * args.drop_band_ratio))
    drop_count = min(max(drop_count, 1), original_bands - 1)
    dropped_band_indices = np.sort(rng.choice(original_bands, size=drop_count, replace=False))
    kept_band_indices = np.setdiff1d(kept_band_indices, dropped_band_indices)
    data = data[:, :, kept_band_indices].copy()
    original_wavelengths = original_wavelengths[kept_band_indices]
    if original_fwhm is not None:
        original_fwhm = original_fwhm[kept_band_indices]
    print("Band-drop experiment is enabled.")
    print("Original bands:", original_bands)
    print("Dropped bands:", drop_count)
    print("Kept bands:", data.shape[2])
    print("Drop seed:", args.drop_band_seed)
#获取空間信息特征
get_spa=1
if get_spa==1:
    img=data
    gt=data_gt
    h, w = gt.shape[0], gt.shape[1]
    c = img.shape[2]
    idx = np.ones((img.shape[0], img.shape[1]))
    idx = np.where(idx == 1)
    idx_x = np.resize(idx[0], (img.shape[0], img.shape[1], 1))
    idx_y = np.resize(idx[1], (img.shape[0], img.shape[1], 1))
    img_idx = np.concatenate((idx_x, idx_y), axis=2)
    img_idx=normalize_maxmin(img_idx)

#設置采樣策略
samples_type=args.samples_type #ratio
# samples_type="number" #number

# 選擇模型
mode_name = "CSGFNet"
# mode_name="GCN"
# mode_name="F2HGNN"
partname="GCN_and_HGCNall"
if args.use_vca:
    partname += "_VCAK{}".format(args.vca_latent_channels)
    if args.vca_adapter_attention_mode != "wavelength":
        partname += "_attn{}".format(args.vca_adapter_attention_mode)
if args.no_wavelength_encoder:
    partname += "_noWave"
if args.drop_band_ratio > 0:
    partname += "_drop{}p_seed{}".format(int(round(args.drop_band_ratio * 100)), args.drop_band_seed)
if args.train_drop_band_ratio > 0:
    partname += "_trainDrop{}p_seed{}".format(int(round(args.train_drop_band_ratio * 100)), args.train_drop_band_seed)
if args.train_band_mixup_ratio > 0:
    partname += "_bandMix{}p_a{}_k{}".format(
        int(round(args.train_band_mixup_ratio * 100)),
        format_float_tag(args.train_band_mixup_alpha),
        args.train_band_mixup_neighbors,
    )
if args.superpixel_scale is not None:
    superpixel_scale = args.superpixel_scale
elif data_set_name == "paviaU":
    superpixel_scale = 50
else:
    superpixel_scale = 100
train_num=args.train_num
val_num=args.val_num
train_ratio=args.train_ratio
val_ratio = args.val_ratio
max_epoch=args.epochs
Train_num=args.train_runs
#WHU 5
#PaviaU 3

if args.seeds is None:
    seed_list=[1,3,3]
else:
    seed_list=[int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
if len(seed_list) == 0:
    raise ValueError("--seeds must contain at least one integer seed")
while len(seed_list) < Train_num:
    seed_list.append(seed_list[-1] + 1)
seed_list = seed_list[:Train_num]
#seed_list=[2,3,5]
learning_rate=0.001
weight_decay=0.001
lb_smooth=0.01

# Save checkpoints and result maps inside this project.
path_weight = os.path.join(os.getcwd(), "weights") + os.sep
path_result = os.path.join(os.getcwd(), "results", data_set_name, mode_name) + os.sep
os.makedirs(path_weight, exist_ok=True)
os.makedirs(path_result, exist_ok=True)
checkpoint_filename = (
    "model_learned_graph.pt"
    if args.use_vca and args.vca_graph_mode == "learned"
    else "model.pt"
)
if args.use_vca and (vca_interaction_options["interaction_mode"] != "none" or vca_interaction_options["graph_update_mode"] != "none"):
    checkpoint_filename = "model_{}_{}_{}_{}_{}.pt".format(
        args.vca_graph_mode, args.vca_alignment_mode, args.vca_interaction_mode,
        args.vca_interaction_direction, args.vca_graph_update_mode,
    )

class_num = np.max(data_gt)
height, width, bands = data.shape
wavelengths, fwhm = original_wavelengths, original_fwhm
use_wavelength_encoder = not args.no_wavelength_encoder
train_dropped_band_indices = np.array([], dtype=np.int64)
if args.train_drop_band_ratio > 0:
    if not 0 < args.train_drop_band_ratio < 1:
        raise ValueError("--train-drop-band-ratio must be in (0, 1)")
    rng = np.random.default_rng(args.train_drop_band_seed)
    train_drop_count = int(np.floor(bands * args.train_drop_band_ratio))
    train_drop_count = min(max(train_drop_count, 1), bands - 1)
    train_dropped_band_indices = np.sort(rng.choice(bands, size=train_drop_count, replace=False))
    print("Train-only band masking is enabled.")
    print("Train dropped bands:", train_drop_count)
    print("Train kept bands:", bands - train_drop_count)
    print("Train drop seed:", args.train_drop_band_seed)
band_mixup_available_indices = np.setdiff1d(np.arange(bands), train_dropped_band_indices)
train_band_mixup_enabled = args.train_band_mixup_ratio > 0 and args.train_band_mixup_prob > 0
if train_band_mixup_enabled:
    print("Train-only close-band mixup is enabled.")
    print("Band mixup ratio:", args.train_band_mixup_ratio)
    print("Band mixup probability:", args.train_band_mixup_prob)
    print("Band mixup alpha:", args.train_band_mixup_alpha)
    print("Band mixup neighbor candidates:", args.train_band_mixup_neighbors)
    print("Band mixup available bands:", band_mixup_available_indices.size)
    if band_mixup_available_indices.size < 2:
        print("Band mixup will be skipped because fewer than two spectral bands are available.")
gt_reshape = np.reshape(data_gt, [-1])

torch.cuda.empty_cache()
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL = []
Test_Time_ALL = []
RUN_METRICS = []
#训练K次取平均值和方差
for i in range(Train_num):
    np.random.seed(seed_list[i])
    torch.manual_seed(int(seed_list[i]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed_list[i]))
    # split datasets
    train_index, val_index, test_index = split_data.split_data(gt_reshape,
                    class_num, train_ratio, val_ratio, train_num, val_num, samples_type)


    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                     train_index, val_index, test_index)


    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt,
                                                test_samples_gt, val_samples_gt, data_gt, class_num)

    # label transfer to one-hot encode
    train_gt = np.reshape(train_samples_gt,[height,width])
    test_gt = np.reshape(test_samples_gt,[height,width])
    val_gt = np.reshape(val_samples_gt,[height,width])
    print_data_info=1
    if print_data_info==1:
        data_reader.data_info(train_gt, val_gt, test_gt)

    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

    if args.use_vca and args.vca_graph_mode == 'learned':
        print("Learned graph mode: skip label-guided LDA-SLIC")
        tic0 = time.time()
        toc0 = tic0
        LDA_SLIC_Time = 0.0
        Q = None
        A = None
    else:
        print("LDA-SLIC Operation is Processing ")
        ls = rdSLIC.LDA_SLIC(data, train_gt, class_num-1)
        tic0=time.time()
        Q, S ,A, Seg= ls.simple_superpixel(scale=superpixel_scale)
        toc0 = time.time()
        LDA_SLIC_Time=toc0-tic0
        Q=VCASAHGCN.dense_assignment_to_sparse_tensor(Q, args.device)
        A=torch.from_numpy(A).to(args.device)

    train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
    test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
    val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)

    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
    val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)

    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)

    #聯合光譜-空間信息
    net_input = np.array(data, np.float32)
    net_input=normalize_maxmin(net_input)
    net_input=net_input = np.concatenate((net_input, img_idx), axis=2)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(args.device)
    train_net_input = net_input
    if train_dropped_band_indices.size > 0:
        train_net_input = net_input.clone()
        train_band_idx = torch.from_numpy(train_dropped_band_indices).long().to(args.device)
        train_pixel_idx = (train_samples_gt != 0).nonzero(as_tuple=True)[0]
        train_net_input_flat = train_net_input.reshape(height * width, -1)
        train_net_input_flat[train_pixel_idx[:, None], train_band_idx[None, :]] = 0.0

    # model PV 0.8 XZ 0.6 WHU 0.4
    if mode_name=="CSGFNet":
        if args.use_vca:
            net=VCASAHGCN.VCASAHGCN(
                height, width, bands+2, class_num, Q, A, alph=0.8,
                spectral_bands=bands,
                wavelengths=wavelengths,
                fwhm=fwhm,
                latent_channels=args.vca_latent_channels,
                adapter_embed_dim=args.vca_embed_dim,
                wavelength_min=args.vca_wavelength_min,
                wavelength_max=args.vca_wavelength_max,
                locality_sigma=args.vca_locality_sigma,
                locality_window=args.vca_locality_window,
                adapter_attention_mode=args.vca_adapter_attention_mode,
                graph_mode=args.vca_graph_mode,
                learned_assignment_dim=args.vca_learned_assignment_dim,
                learned_assignment_radius=args.vca_learned_assignment_radius,
                learned_graph_radius=args.vca_learned_graph_radius,
                learned_assignment_temperature=args.vca_learned_assignment_temperature,
                learned_assignment_spatial_sigma=args.vca_learned_assignment_spatial_sigma,
                learned_graph_spatial_sigma=args.vca_learned_graph_spatial_sigma,
                superpixel_scale=superpixel_scale,
                **vca_interaction_options,
            ).to(args.device)
        else:
            net=SAHGCN.SAHGCN(
                height, width, bands+2, class_num, Q, A, alph=0.8,
                spectral_bands=bands,
                wavelengths=wavelengths,
                fwhm=fwhm,
                use_wavelength_encoder=use_wavelength_encoder,
            ).to(args.device)
    elif mode_name=="GCN":
        net=GCN.GCN(bands,128,class_num,Q,A).to(args.device)
    elif mode_name=="F2HGNN":
        net=F2HGNN.HGNN_weight(net_input.shape[-1],128,class_num,Q,A,dropout=0.2).to(args.device)

    # 统计参数
    totalPar = sum([param.nelement() for param in net.parameters()])
    print('Number of parameter: % .2fK' % (totalPar / 1e3))
    # train
    print("\n\n==================== train ====================\n")
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate, weight_decay=weight_decay) #, weight_decay=0.0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    zeros = torch.zeros([height * width]
                        ).to(args.device).float()
    best_loss=99999
    net.train()
    tic1 = time.time()
    progress_enabled = not args.no_progress
    if progress_enabled and tqdm is None:
        print("tqdm is not installed; falling back to plain training logs.")
    epoch_iter = make_progress(
        range(max_epoch + 1),
        enabled=progress_enabled,
        desc="epochs",
        dynamic_ncols=True,
    )
    for i in epoch_iter:
        optimizer.zero_grad()  # zero the gradient buffers
        current_train_input = train_net_input
        if train_band_mixup_enabled:
            current_train_input = make_band_mixup_input(
                train_net_input,
                train_samples_gt,
                height,
                width,
                bands,
                wavelengths,
                args.train_band_mixup_ratio,
                args.train_band_mixup_alpha,
                args.train_band_mixup_neighbors,
                mixup_prob=args.train_band_mixup_prob,
                available_band_indices=band_mixup_available_indices,
            )

        #onlyCNN
        # output= net(net_input)
        # loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)

        if args.use_vca and args.vca_graph_mode == 'learned':
            output,outputHGNN,outputCNN,graph_info = net(
                current_train_input,
                return_graph_info=True,
            )
        else:
            output,outputHGNN,outputCNN= net(current_train_input)
        lossHGNN = utils.compute_loss(outputHGNN, train_gt_onehot, train_label_mask)
        lossCNN = utils.compute_loss(outputCNN, train_gt_onehot, train_label_mask)
        loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
        loss=loss+lossCNN+lossHGNN
        if args.use_vca and args.vca_graph_mode == 'learned':
            loss = (
                loss
                + args.vca_graph_compactness_weight * graph_info['compactness_loss']
                + args.vca_graph_entropy_weight * graph_info['assignment_entropy_loss']
                + args.vca_graph_balance_weight * graph_info['assignment_balance_loss']
            )


        loss.backward()
        optimizer.step()  # Does the update

        # if i%10==0:
        with torch.no_grad():
            net.eval()

            #onlyCNN
            # output= net(net_input)
            # trainloss = utils.compute_loss(output, train_gt_onehot, train_label_mask)

            output_train,outputHGNN,outputCNN= net(current_train_input)
            trainloss = utils.compute_loss(output_train, train_gt_onehot, train_label_mask)

            trainOA = utils.evaluate_performance(output_train, train_samples_gt, train_gt_onehot, zeros)
            output_val = output_train
            if train_dropped_band_indices.size > 0 or train_band_mixup_enabled:
                output_val,_,_= net(net_input)
            valloss = utils.compute_loss(output_val, val_gt_onehot, val_label_mask)
            valOA = utils.evaluate_performance(output_val, val_samples_gt, val_gt_onehot, zeros)
            # print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

            if valloss < best_loss :
                best_loss = valloss
                torch.save(net.state_dict(), path_weight + checkpoint_filename)
                log_message('save model...', progress_enabled)
        scheduler.step(valloss)
        torch.cuda.empty_cache()
        net.train()

        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix({
                "train": "{:.4f}".format(float(trainloss.detach().cpu())),
                "val": "{:.4f}".format(float(valloss.detach().cpu())),
                "trainOA": "{:.4f}".format(float(trainOA)),
                "valOA": "{:.4f}".format(float(valOA)),
            })

        if i%10==0:
            log_message("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA), progress_enabled)
    toc1 = time.time()

    print("\n\n====================training done. starting evaluation...========================\n")

    # test
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load(path_weight + checkpoint_filename))
        net.eval()
        tic2 = time.time()

        #OnlyCNN
        # output= net(net_input)
        
        output,_,_ = net(net_input)

        torch.save(output, 'output_all_PaviaU.pt')

        toc2 = time.time()
        testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
        print(zeros.shape)
        print(class_num)
        print(height)
        print(width)
        testOA,testAA,testKppa,test_AC_list = utils.evaluate_performance2(output, test_samples_gt, test_gt_onehot, zeros,np.array(class_num),np.array(height),np.array(width))
        print("{}\ttest loss={:.4f}\t test OA={:.4f}\t test AA={:.4f}\t test kppa={:.4f}".format(str(i + 1), testloss, testOA,testAA,testKppa))
        OA_ALL.append(testOA)
        AA_ALL.append(testAA)
        KPP_ALL.append(testKppa)
        AVG_ALL.append(test_AC_list)
        RUN_METRICS.append({
            "run": len(RUN_METRICS),
            "seed": int(seed_list[len(RUN_METRICS)]),
            "loss": float(testloss.detach().cpu()),
            "oa": float(testOA),
            "aa": float(testAA),
            "kappa": float(testKppa),
            "acc_per_class": np.asarray(test_AC_list).astype(float).tolist(),
        })
        totalPar = sum([param.nelement() for param in net.parameters()])
        print('Number of parameter: % .2fK' % (totalPar  / 1e3))
    torch.cuda.empty_cache()
    del net

    LDA_SLIC_Time=toc0-tic0
    print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
    training_time = toc1 - tic1 + LDA_SLIC_Time
    testing_time = toc2 - tic2 + LDA_SLIC_Time
    Train_Time_ALL.append(training_time)
    Test_Time_ALL.append(testing_time)
    print("train costs time: {}".format(training_time - LDA_SLIC_Time))
    print("train costs time(on-): {}".format(training_time))
    print('Number of parameter: % .2fK' % (totalPar / 1e3))

    # classification report
    test_label_mask_cpu = test_label_mask.cpu().numpy()[:,0].astype('bool')
    test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
    predict = torch.argmax(output, 1).cpu().numpy()

    classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
    mask=data_gt>0
    masked_predicted_labels = np.copy(classification_map)
    masked_predicted_labels[~mask]=0
    # visualize_hyperspectral_image(masked_predicted_labels)
    if not args.no_save_maps:
        Draw_Classification_Map(classification_map, path_result+mode_name+"-"+ data_set_name + str(testOA)+str(train_ratio)+"-BigImage-"+partname)
        Draw_Classification_Map(masked_predicted_labels, path_result+ mode_name+"-"+data_set_name + str(testOA)+str(train_ratio)+"-smallImage-"+partname)

    classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                        predict[test_label_mask_cpu]+1, digits=4)
    kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu]+1)
    show_results=1
    if show_results==1:
        print(classification, kappa)

    # store results
    save_results=1
    if save_results==1:
        print("save results")
        run_date = time.strftime('%Y%m%d-%H%M-',time.localtime(time.time()))
        f = open(path_result + run_date +mode_name+"-"+data_set_name +"CSFGNet_all"+partname+ '.txt', 'a+')
        str_results = '\n ======================' \
                    + '\nrun data = ' + run_date \
                    + "\nlearning rate = " + str(learning_rate) \
                    + "\nepochs = " + str(max_epoch) \
                    + "\nsamples_type = " + str(samples_type) \
                    + "\nuse vca = " + str(args.use_vca) \
                    + "\nvca latent channels = " + str(args.vca_latent_channels) \
                    + "\nvca embed dim = " + str(args.vca_embed_dim) \
                    + "\nvca locality sigma = " + str(args.vca_locality_sigma) \
                    + "\nvca locality window = " + str(args.vca_locality_window) \
                    + "\nvca adapter attention mode = " + str(args.vca_adapter_attention_mode) \
                    + "\nvca wavelength min = " + str(args.vca_wavelength_min) \
                    + "\nvca wavelength max = " + str(args.vca_wavelength_max) \
                    + "\nuse wavelength encoder = " + str(use_wavelength_encoder) \
                    + "\ndrop band ratio = " + str(args.drop_band_ratio) \
                    + "\ndrop band seed = " + str(args.drop_band_seed) \
                    + "\noriginal bands = " + str(original_bands) \
                    + "\nkept bands = " + str(bands) \
                    + "\ndropped band indices = " + str(dropped_band_indices.tolist()) \
                    + "\ntrain drop band ratio = " + str(args.train_drop_band_ratio) \
                    + "\ntrain drop band seed = " + str(args.train_drop_band_seed) \
                    + "\ntrain dropped band indices = " + str(train_dropped_band_indices.tolist()) \
                    + "\ntrain band mixup ratio = " + str(args.train_band_mixup_ratio) \
                    + "\ntrain band mixup prob = " + str(args.train_band_mixup_prob) \
                    + "\ntrain band mixup alpha = " + str(args.train_band_mixup_alpha) \
                    + "\ntrain band mixup neighbors = " + str(args.train_band_mixup_neighbors) \
                    + "\ntrain band mixup available indices = " + str(band_mixup_available_indices.tolist()) \
                    + "\ntrain ratio = " + str(train_ratio) \
                    + "\nval ratio = " + str(val_ratio) \
                    + "\ntrain num = " + str(train_num) \
                    + "\nval num = " + str(val_num) \
                    + "\nseed list = " + str(seed_list) \
                    + '\ntrain time = ' + str(training_time) \
                    + '\ntest time = ' + str(testing_time) \
                    + '\n' + classification \
                    + "kappa = " + str(kappa) \
                    + '\n'
        f.write(str_results)
        f.close()

run_date2 = time.strftime('%Y%m%d-%H%M-',time.localtime(time.time()))
f = open(path_result + run_date +partname+"-"+data_set_name +"mean_result"+ '.txt', 'a+')
str_results = '\n =====***************Mean result************=========' \
                + '\nList of OA:' + str(list(OA_ALL)) \
                + '\nList of AA:' + str(list(AA_ALL)) \
                + '\nList of KPP:' + str(list(KPP_ALL)) \
                + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
    round(np.std(KPP_ALL) * 100, 2)) \
                + '\nAcc per class=\n' + str(np.round(np.mean(AVG_ALL, 0) * 100, 2)) + '+-' + str(
                np.round(np.std(AVG_ALL, 0) * 100, 2))\
                + '\n'
f.write(str_results)
f.close()



OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL=np.array(Train_Time_ALL)
Test_Time_ALL=np.array(Test_Time_ALL)
print("")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))

if args.metrics_json is not None:
    metrics_dir = os.path.dirname(os.path.abspath(args.metrics_json))
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    payload = {
        "experiment": args.experiment_name,
        "script": "main.py",
        "dataset": data_set_name,
        "model": "VCASAHGCN" if args.use_vca else "SAHGCN",
        "config": {
            **{"vca_" + key: value for key, value in vca_interaction_options.items()},
            "epochs": max_epoch,
            "train_runs": Train_num,
            "seeds": [int(seed) for seed in seed_list],
            "samples_type": samples_type,
            "train_num": train_num,
            "val_num": val_num,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "superpixel_scale": superpixel_scale,
            "use_vca": args.use_vca,
            "vca_latent_channels": args.vca_latent_channels,
            "vca_embed_dim": args.vca_embed_dim,
            "vca_adapter_attention_mode": args.vca_adapter_attention_mode,
            "vca_graph_mode": args.vca_graph_mode,
            "vca_learned_assignment_dim": args.vca_learned_assignment_dim,
            "vca_learned_assignment_radius": args.vca_learned_assignment_radius,
            "vca_learned_graph_radius": args.vca_learned_graph_radius,
            "vca_graph_compactness_weight": args.vca_graph_compactness_weight,
            "vca_graph_entropy_weight": args.vca_graph_entropy_weight,
            "vca_graph_balance_weight": args.vca_graph_balance_weight,
            "use_wavelength_encoder": use_wavelength_encoder,
            "drop_band_ratio": args.drop_band_ratio,
            "train_drop_band_ratio": args.train_drop_band_ratio,
            "train_band_mixup_ratio": args.train_band_mixup_ratio,
        },
        "runs": RUN_METRICS,
        "summary": {
            "oa_mean": float(np.mean(OA_ALL)),
            "oa_std": float(np.std(OA_ALL)),
            "aa_mean": float(np.mean(AA_ALL)),
            "aa_std": float(np.std(AA_ALL)),
            "kappa_mean": float(np.mean(KPP_ALL)),
            "kappa_std": float(np.std(KPP_ALL)),
            "acc_per_class_mean": np.asarray(np.mean(AVG_ALL, 0)).astype(float).tolist(),
            "acc_per_class_std": np.asarray(np.std(AVG_ALL, 0)).astype(float).tolist(),
            "train_time_mean": float(np.mean(Train_Time_ALL)) if Train_Time_ALL.size else None,
            "test_time_mean": float(np.mean(Test_Time_ALL)) if Test_Time_ALL.size else None,
        },
    }
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("write metrics json:", args.metrics_json)
