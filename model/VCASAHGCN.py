import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SAHGCN import HGNN_conv
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.variable_channel_adapter import VariableChannelAdapter
from model.region_pixel_alignment import RegionPixelAlignment, pool_regions, broadcast_regions
from model.cross_branch_interaction import BidirectionalSelectiveInteraction
from model.graph_edge_update import GraphEdgeUpdater
from model.interaction_config import InteractionConfig, interaction_kwargs


def dense_assignment_to_sparse_tensor(Q, device):
    rows, cols = Q.nonzero()
    indices = torch.from_numpy(np.vstack((rows, cols)).astype("int64", copy=False)).to(device)
    values = torch.from_numpy(Q[rows, cols].astype("float32", copy=False)).to(device)
    return torch.sparse_coo_tensor(indices, values, Q.shape, device=device).coalesce()


def _safe_dataset_names(dataset_names):
    if dataset_names is None or len(dataset_names) == 0:
        return ["default"]
    return list(dataset_names)


def _group_count(num_channels, max_groups=8):
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class DomainSpecificBatchNorm1d(nn.Module):
    def __init__(self, num_features, dataset_names):
        super(DomainSpecificBatchNorm1d, self).__init__()
        self.norms = nn.ModuleDict({
            name: nn.BatchNorm1d(num_features)
            for name in _safe_dataset_names(dataset_names)
        })

    def add_dataset(self, dataset_name):
        if dataset_name not in self.norms:
            self.norms[dataset_name] = nn.BatchNorm1d(next(iter(self.norms.values())).num_features)

    def forward(self, x, dataset_name=None):
        if dataset_name is None:
            dataset_name = next(iter(self.norms.keys()))
        return self.norms[dataset_name](x)


class DomainSpecificBatchNorm2d(nn.Module):
    def __init__(self, num_features, dataset_names):
        super(DomainSpecificBatchNorm2d, self).__init__()
        self.norms = nn.ModuleDict({
            name: nn.BatchNorm2d(num_features)
            for name in _safe_dataset_names(dataset_names)
        })

    def add_dataset(self, dataset_name):
        if dataset_name not in self.norms:
            self.norms[dataset_name] = nn.BatchNorm2d(next(iter(self.norms.values())).num_features)

    def forward(self, x, dataset_name=None):
        if dataset_name is None:
            dataset_name = next(iter(self.norms.keys()))
        return self.norms[dataset_name](x)


class DatasetAwareNorm1d(nn.Module):
    def __init__(self, num_features, norm_type="batch_norm", dataset_names=None):
        super(DatasetAwareNorm1d, self).__init__()
        self.norm_type = norm_type
        if norm_type == "domain_bn":
            self.norm = DomainSpecificBatchNorm1d(num_features, dataset_names)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(num_features)
        elif norm_type == "group_norm":
            self.norm = nn.LayerNorm(num_features)
        else:
            raise ValueError("Unsupported norm_type: {}".format(norm_type))

    def add_dataset(self, dataset_name):
        if hasattr(self.norm, "add_dataset"):
            self.norm.add_dataset(dataset_name)

    def forward(self, x, dataset_name=None):
        if self.norm_type == "domain_bn":
            return self.norm(x, dataset_name)
        return self.norm(x)


class DatasetAwareNorm2d(nn.Module):
    def __init__(self, num_features, norm_type="batch_norm", dataset_names=None):
        super(DatasetAwareNorm2d, self).__init__()
        self.norm_type = norm_type
        if norm_type == "domain_bn":
            self.norm = DomainSpecificBatchNorm2d(num_features, dataset_names)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(num_features)
        elif norm_type == "group_norm":
            self.norm = nn.GroupNorm(_group_count(num_features), num_features)
        else:
            raise ValueError("Unsupported norm_type: {}".format(norm_type))

    def add_dataset(self, dataset_name):
        if hasattr(self.norm, "add_dataset"):
            self.norm.add_dataset(dataset_name)

    def forward(self, x, dataset_name=None):
        if self.norm_type == "domain_bn":
            return self.norm(x, dataset_name)
        return self.norm(x)


class DenoiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch_norm", dataset_names=None):
        super(DenoiseBlock, self).__init__()
        self.norm = DatasetAwareNorm2d(in_channels, norm_type=norm_type, dataset_names=dataset_names)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.act = nn.LeakyReLU()

    def forward(self, x, dataset_name=None):
        x = self.norm(x, dataset_name)
        x = self.conv(x)
        return self.act(x)


class DatasetAwareSSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, norm_type="batch_norm", dataset_names=None):
        super(DatasetAwareSSConv, self).__init__()
        self.norm = DatasetAwareNorm2d(in_ch, norm_type=norm_type, dataset_names=dataset_names)
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

    def forward(self, x, dataset_name=None):
        out = self.point_conv(self.norm(x, dataset_name))
        out = self.act1(out)
        out = self.depth_conv(out)
        out = self.act2(out)
        return out


class SharedHGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0,
                 norm_type="batch_norm", dataset_names=None):
        super(SharedHGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(input_dim, hidden_dim)
        self.hgc2 = HGNN_conv(hidden_dim, output_dim)
        self.batch_normalzation2 = DatasetAwareNorm1d(
            hidden_dim,
            norm_type=norm_type,
            dataset_names=dataset_names,
        )
        self._sparse_graph_cache = {}

    def add_dataset(self, dataset_name):
        self.batch_normalzation2.add_dataset(dataset_name)

    def clear_graph_cache(self):
        self._sparse_graph_cache = {}

    def _sparse_graph(self, A, dataset_name=None):
        if A.is_sparse:
            return A.coalesce()
        cache_key = (
            dataset_name or "default",
            str(A.device),
            str(A.dtype),
            tuple(A.shape),
            int(A.data_ptr()),
        )
        cached = self._sparse_graph_cache.get(cache_key)
        if cached is not None:
            return cached
        sparse_A = A.to(torch.float32).to_sparse().coalesce()
        self._sparse_graph_cache[cache_key] = sparse_A
        return sparse_A

    def _graph_conv(self, conv, x, A_sparse):
        x = x.matmul(conv.weight)
        if conv.bias is not None:
            x = x + conv.bias
        return torch.sparse.mm(A_sparse, x)

    def _project_to_superpixels(self, Q, x_flatten):
        return pool_regions(Q, x_flatten)

    def _project_to_pixels(self, Q, sup_x):
        return broadcast_regions(Q, sup_x)

    def forward_stage1(self, regions, A, dataset_name=None):
        regions = self._graph_conv(self.hgc1, regions, self._sparse_graph(A, dataset_name))
        regions = self.batch_normalzation2(regions, dataset_name)
        regions = F.relu(regions)
        return F.dropout(regions, self.dropout, training=self.training)

    def forward_stage2(self, regions, A, dataset_name=None):
        return self._graph_conv(self.hgc2, regions, self._sparse_graph(A, dataset_name))

    def forward(self, x, Q, A, dataset_name=None):
        height, width, _ = x.shape
        x_flatten = x.reshape([height * width, -1])
        sup_x = self._project_to_superpixels(Q, x_flatten)
        sup_x = self.forward_stage1(sup_x, A, dataset_name)
        sup_x = self.forward_stage2(sup_x, A, dataset_name)
        return self._project_to_pixels(Q, sup_x)


class VCASAHGCNBackbone(nn.Module):
    """
    Shared VCA + SAHGCN feature extractor.

    In static mode Q and A are passed per forward call. In learned mode a
    differentiable local assignment and dynamic graph are generated from the
    VCA-normalized feature map.
    """

    def __init__(
        self,
        latent_channels=128,
        extra_channels=2,
        alph=0.8,
        adapter_embed_dim=64,
        adapter_value_dim=1,
        adapter_num_frequencies=6,
        wavelength_min=400.0,
        wavelength_max=2500.0,
        locality_sigma=0.08,
        locality_window=0.16,
        adapter_attention_mode="wavelength",
        adapter_chunk_pixels=0,
        adapter_checkpoint_chunks=False,
        norm_type="batch_norm",
        dataset_names=None,
        graph_mode="static",
        learned_assignment_dim=32,
        learned_assignment_radius=1,
        learned_graph_radius=1,
        learned_assignment_temperature=1.0,
        learned_assignment_spatial_sigma=1.0,
        learned_graph_spatial_sigma=1.5,
        alignment_mode="region",
        interaction_mode="none",
        interaction_direction="both",
        interaction_strength=0.1,
        interaction_hidden_dim=32,
        interaction_chunk_pixels=16384,
        graph_update_mode="none",
        graph_update_dim=32,
        graph_update_mix=0.5,
    ):
        super(VCASAHGCNBackbone, self).__init__()
        self.extra_channels = extra_channels
        self.alph = alph
        self.norm_type = norm_type
        if graph_mode not in ["static", "learned"]:
            raise ValueError("graph_mode must be one of: static, learned")
        self.graph_mode = graph_mode
        self.interaction_config = InteractionConfig(**interaction_kwargs(locals()))
        self.dataset_names = _safe_dataset_names(dataset_names)
        self.adapter = VariableChannelAdapter(
            latent_channels=latent_channels,
            embed_dim=adapter_embed_dim,
            value_dim=adapter_value_dim,
            num_frequencies=adapter_num_frequencies,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            locality_sigma=locality_sigma,
            locality_window=locality_window,
            attention_mode=adapter_attention_mode,
            chunk_pixels=adapter_chunk_pixels,
            checkpoint_chunks=adapter_checkpoint_chunks,
        )
        self.channel = self.adapter.out_channels + extra_channels

        layers_count = 2
        self.CNN_denoise = nn.ModuleList()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.append(
                    DenoiseBlock(
                        self.channel,
                        64,
                        norm_type=norm_type,
                        dataset_names=self.dataset_names,
                    )
                )
            else:
                self.CNN_denoise.append(
                    DenoiseBlock(
                        64,
                        64,
                        norm_type=norm_type,
                        dataset_names=self.dataset_names,
                    )
                )

        self.CNN_Branch = nn.ModuleList()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.append(
                    DatasetAwareSSConv(
                        64,
                        128,
                        kernel_size=3,
                        norm_type=norm_type,
                        dataset_names=self.dataset_names,
                    )
                )
            else:
                self.CNN_Branch.append(
                    DatasetAwareSSConv(
                        128,
                        64,
                        kernel_size=5,
                        norm_type=norm_type,
                        dataset_names=self.dataset_names,
                    )
                )

        self.HGNN_Branch = SharedHGCN(
            64,
            128,
            64,
            dropout=0.0,
            norm_type=norm_type,
            dataset_names=self.dataset_names,
        )
        if self.graph_mode == "learned":
            self.learnable_graph = LearnableSuperpixelGraph(
                feature_dim=64,
                assignment_dim=learned_assignment_dim,
                assignment_radius=learned_assignment_radius,
                graph_radius=learned_graph_radius,
                assignment_temperature=learned_assignment_temperature,
                assignment_spatial_sigma=learned_assignment_spatial_sigma,
                graph_spatial_sigma=learned_graph_spatial_sigma,
            )
        else:
            self.learnable_graph = None

        self.alignment = RegionPixelAlignment(alignment_mode)
        # New ablations keep identical initialization of the existing backbone/heads.
        with torch.random.fork_rng(devices=[]):
            self.interaction = BidirectionalSelectiveInteraction(
                channels=128, mode=interaction_mode, direction=interaction_direction,
                hidden_dim=interaction_hidden_dim, strength=interaction_strength,
                chunk_size=interaction_chunk_pixels,
            ) if interaction_mode != "none" else None
            self.edge_updater = GraphEdgeUpdater(
                channels=128, mode=graph_update_mode, embed_dim=graph_update_dim, mix=graph_update_mix,
            ) if graph_update_mode != "none" else None

    def add_dataset(self, dataset_name):
        if dataset_name in self.dataset_names:
            return
        self.dataset_names.append(dataset_name)
        for module in self.modules():
            if module is not self and hasattr(module, "add_dataset"):
                module.add_dataset(dataset_name)

    def _split_spectral_extra(self, x, wavelengths):
        if wavelengths is not None:
            spectral_bands = int(torch.as_tensor(wavelengths).numel())
        else:
            spectral_bands = x.shape[-1] - self.extra_channels
        if spectral_bands <= 0:
            raise ValueError("spectral_bands must be positive")
        spectral_x = x[:, :, :spectral_bands]
        extra_x = x[:, :, spectral_bands:]
        if extra_x.shape[-1] != self.extra_channels:
            raise ValueError(
                "expected {} extra channels, got {}".format(self.extra_channels, extra_x.shape[-1])
            )
        return spectral_x, extra_x

    def forward_features(
        self,
        x,
        Q=None,
        A=None,
        wavelengths=None,
        fwhm=None,
        dataset_name=None,
        superpixel_scale=100,
        return_graph_info=False,
        return_diagnostics=False,
    ):
        if return_diagnostics and not return_graph_info:
            raise ValueError("return_diagnostics requires return_graph_info=True")
        height, width, _ = x.shape
        spectral_x, extra_x = self._split_spectral_extra(x, wavelengths)
        latent_x = self.adapter(spectral_x, wavelengths=wavelengths, fwhm=fwhm)
        x = torch.cat((latent_x, extra_x), dim=2)

        noise = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        for block in self.CNN_denoise:
            noise = block(noise, dataset_name)
        clean_x = torch.squeeze(noise, 0).permute([1, 2, 0])

        graph_info = {}
        if self.graph_mode == "learned":
            Q, A, graph_info = self.learnable_graph(clean_x, superpixel_scale)
        elif Q is None or A is None:
            raise ValueError("static graph mode requires Q and A")

        if self.interaction_config.active:
            cnn_result, hgnn_result, interaction_info = self._interactive_features(
                clean_x, Q, A, dataset_name, return_graph_info, return_diagnostics,
            )
            if return_graph_info:
                graph_info["interaction"] = interaction_info
        else:
            cnn_result = torch.unsqueeze(clean_x.permute([2, 0, 1]), 0)
            for block in self.CNN_Branch:
                cnn_result = block(cnn_result, dataset_name)
            cnn_result = torch.squeeze(cnn_result, 0).permute([1, 2, 0]).reshape([height * width, -1])
            hgnn_result = self.HGNN_Branch(clean_x, Q, A, dataset_name=dataset_name)
        fused = self.alph * cnn_result + (1.0 - self.alph) * hgnn_result
        if return_graph_info:
            return fused, hgnn_result, cnn_result, graph_info
        return fused, hgnn_result, cnn_result

    def _interactive_features(self, x, Q, A, dataset_name, collect_stats, return_diagnostics):
        height, width, channels = x.shape
        cnn = self.CNN_Branch[0](x.permute(2, 0, 1).unsqueeze(0), dataset_name)
        pixels = cnn.squeeze(0).permute(1, 2, 0).reshape(height * width, 128)
        regions = self.HGNN_Branch.forward_stage1(pool_regions(Q, x.reshape(-1, channels)), A, dataset_name)
        info = {}
        if self.interaction is not None:
            # Both messages use the same pre-interaction snapshot (no update-order bias).
            region_at_pixels = self.alignment.to_pixels(regions, Q) if self.interaction.graph_to_cnn is not None else None
            pixels_at_regions = self.alignment.to_regions(pixels, Q) if self.interaction.cnn_to_graph is not None else None
            pixels, regions, message_info = self.interaction(
                pixels, regions, region_at_pixels, pixels_at_regions,
                collect_stats=collect_stats, return_diagnostics=return_diagnostics,
            )
            info.update(message_info)
        if self.edge_updater is not None:
            aligned_pixels = self.alignment.to_regions(pixels, Q) if self.edge_updater.mode == "joint" else None
            A, edge_info = self.edge_updater(
                regions, A, aligned_pixels, collect_stats=collect_stats, return_diagnostics=return_diagnostics,
            )
            info.update(edge_info)
        if return_diagnostics and "cnn_to_graph_gate" in info:
            info["cnn_to_graph_gate_pixels"] = broadcast_regions(Q, info["cnn_to_graph_gate"]).detach()
        cnn = pixels.reshape(height, width, 128).permute(2, 0, 1).unsqueeze(0)
        cnn = self.CNN_Branch[1](cnn, dataset_name)
        cnn = cnn.squeeze(0).permute(1, 2, 0).reshape(height * width, 64)
        regions = self.HGNN_Branch.forward_stage2(regions, A, dataset_name)
        return cnn, broadcast_regions(Q, regions), info


class MultiDatasetVCASAHGCN(nn.Module):
    """
    Shared VCA/backbone with dataset-specific classifier heads.
    """

    def __init__(self, class_counts, **backbone_kwargs):
        super(MultiDatasetVCASAHGCN, self).__init__()
        backbone_kwargs.setdefault("dataset_names", list(class_counts.keys()))
        self.backbone_kwargs = {
            name: parameter.default
            for name, parameter in inspect.signature(VCASAHGCNBackbone.__init__).parameters.items()
            if name != "self"
        }
        self.backbone_kwargs.update(backbone_kwargs)
        self.backbone = VCASAHGCNBackbone(**backbone_kwargs)
        self.heads = nn.ModuleDict()
        self.hgnn_heads = nn.ModuleDict()
        self.cnn_heads = nn.ModuleDict()
        for dataset_name, class_count in class_counts.items():
            class_count = int(class_count)
            self.heads[dataset_name] = nn.Linear(64, class_count)
            self.hgnn_heads[dataset_name] = nn.Linear(64, class_count)
            self.cnn_heads[dataset_name] = nn.Linear(64, class_count)

    def export_config(self):
        kwargs = dict(self.backbone_kwargs)
        kwargs["dataset_names"] = list(self.heads.keys())
        return {"format_version": 1, "class_counts": {name: head.out_features for name, head in self.heads.items()},
                "backbone_kwargs": kwargs}

    def add_dataset_head(self, dataset_name, class_count):
        if dataset_name in self.heads:
            raise ValueError("{} already has a dataset head".format(dataset_name))
        class_count = int(class_count)
        self.backbone.add_dataset(dataset_name)
        self.heads[dataset_name] = nn.Linear(64, class_count)
        self.hgnn_heads[dataset_name] = nn.Linear(64, class_count)
        self.cnn_heads[dataset_name] = nn.Linear(64, class_count)

    def forward(
        self,
        dataset_name,
        x,
        Q=None,
        A=None,
        wavelengths=None,
        fwhm=None,
        superpixel_scale=100,
        return_graph_info=False,
        return_diagnostics=False,
    ):
        features = self.backbone.forward_features(
            x,
            Q,
            A,
            wavelengths=wavelengths,
            fwhm=fwhm,
            dataset_name=dataset_name,
            superpixel_scale=superpixel_scale,
            return_graph_info=return_graph_info,
            return_diagnostics=return_diagnostics,
        )
        if return_graph_info:
            fused, hgnn_result, cnn_result, graph_info = features
        else:
            fused, hgnn_result, cnn_result = features
        y = F.softmax(self.heads[dataset_name](fused), -1)
        y_hgnn = F.softmax(self.hgnn_heads[dataset_name](hgnn_result), -1)
        y_cnn = F.softmax(self.cnn_heads[dataset_name](cnn_result), -1)
        if return_graph_info:
            return y, y_hgnn, y_cnn, graph_info
        return y, y_hgnn, y_cnn


class VCASAHGCN(nn.Module):
    """
    Single-dataset compatibility wrapper for main.py.
    """

    def __init__(
        self,
        height,
        width,
        changel,
        class_count,
        Q,
        A,
        alph,
        spectral_bands=None,
        wavelengths=None,
        fwhm=None,
        latent_channels=128,
        extra_channels=2,
        superpixel_scale=100,
        **adapter_kwargs,
    ):
        super(VCASAHGCN, self).__init__()
        self.height = height
        self.width = width
        self.channel = changel
        self.class_count = class_count
        self.superpixel_scale = superpixel_scale
        self.spectral_bands = spectral_bands if spectral_bands is not None else changel - extra_channels
        if Q is None:
            self.Q = None
        else:
            self.register_buffer("Q", Q)
        if A is None:
            self.A = None
        else:
            self.register_buffer("A", A)

        if wavelengths is None:
            self.wavelengths = None
        else:
            self.register_buffer("wavelengths", torch.as_tensor(wavelengths, dtype=torch.float32).reshape(-1))
        if fwhm is None:
            self.fwhm = None
        else:
            self.register_buffer("fwhm", torch.as_tensor(fwhm, dtype=torch.float32).reshape(-1))

        self.model = MultiDatasetVCASAHGCN(
            {"default": class_count},
            latent_channels=latent_channels,
            extra_channels=extra_channels,
            alph=alph,
            **adapter_kwargs,
        )

    def forward(self, x, return_graph_info=False, return_diagnostics=False):
        wavelengths = self.wavelengths if hasattr(self, "wavelengths") else None
        fwhm = self.fwhm if hasattr(self, "fwhm") else None
        return self.model(
            "default",
            x,
            self.Q,
            self.A,
            wavelengths=wavelengths,
            fwhm=fwhm,
            superpixel_scale=self.superpixel_scale,
            return_graph_info=return_graph_info,
            return_diagnostics=return_diagnostics,
        )
