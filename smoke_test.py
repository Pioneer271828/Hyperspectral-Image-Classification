"""Synthetic end-to-end check for VCA and the CNN/HGCN dual-branch backbone."""

import argparse

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN


def identity_graph(height, width, device="cpu"):
    """Return a simple valid pixel-to-superpixel assignment and adjacency."""
    pixel_count = height * width
    assignment = torch.eye(pixel_count, dtype=torch.float32, device=device).to_sparse().coalesce()
    adjacency = torch.eye(pixel_count, dtype=torch.float32, device=device)
    return assignment, adjacency


def run_case(model, dataset_name, bands, classes):
    device = next(model.parameters()).device
    height, width = 4, 5
    pixel_count = height * width
    wavelengths = torch.linspace(450.0, 900.0, bands, device=device)
    spectral = torch.randn(height, width, bands, device=device)
    coordinates = torch.rand(height, width, 2, device=device)
    model_input = torch.cat((spectral, coordinates), dim=-1)
    assignment, adjacency = identity_graph(height, width, device)

    with torch.no_grad():
        latent = model.backbone.adapter(spectral, wavelengths=wavelengths)
        fused, hgnn, cnn = model.backbone.forward_features(
            model_input,
            assignment,
            adjacency,
            wavelengths=wavelengths,
            dataset_name=dataset_name,
        )
        output, output_hgnn, output_cnn = model(
            dataset_name,
            model_input,
            assignment,
            adjacency,
            wavelengths=wavelengths,
        )

    assert latent.shape == (height, width, 8)
    assert fused.shape == hgnn.shape == cnn.shape == (pixel_count, 64)
    assert output.shape == output_hgnn.shape == output_cnn.shape == (pixel_count, classes)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(pixel_count, device=device), atol=1e-5)
    print(
        f"{dataset_name}: input_bands={bands}, latent_channels={latent.shape[-1]}, "
        f"classes={output.shape[-1]}"
    )


def run_learned_graph_case(device="cpu"):
    height, width, bands, classes = 6, 7, 15, 5
    wavelengths = torch.linspace(450.0, 950.0, bands, device=device)
    spectral = torch.randn(height, width, bands, device=device)
    coordinates = torch.rand(height, width, 2, device=device)
    model_input = torch.cat((spectral, coordinates), dim=-1)

    model = MultiDatasetVCASAHGCN(
        {"learned": classes},
        latent_channels=8,
        adapter_embed_dim=16,
        extra_channels=2,
        norm_type="group_norm",
        locality_window=None,
        graph_mode="learned",
        learned_assignment_dim=12,
    ).to(device).train()
    assignment, graph, direct_info = model.backbone.learnable_graph(
        torch.randn(height, width, 64, device=device),
        superpixel_scale=9,
    )
    assignment_row_sum = torch.sparse.sum(assignment, dim=1).to_dense()
    graph_row_sum = torch.sparse.sum(graph, dim=1).to_dense()
    assert torch.allclose(assignment_row_sum, torch.ones_like(assignment_row_sum), atol=1e-5)
    assert torch.allclose(graph_row_sum, torch.ones_like(graph_row_sum), atol=1e-5)
    assert direct_info["token_count"] == 6

    outputs = model(
        "learned",
        model_input,
        wavelengths=wavelengths,
        superpixel_scale=9,
        return_graph_info=True,
    )
    output, output_hgnn, output_cnn, graph_info = outputs
    pixel_count = height * width

    assert output.shape == output_hgnn.shape == output_cnn.shape == (pixel_count, classes)
    assert graph_info["token_count"] == 6
    assert graph_info["assignment_nnz"] <= pixel_count * 9
    assert graph_info["graph_nnz"] <= graph_info["token_count"] * 9
    loss = (
        -torch.log(output[0, 0].clamp_min(1e-8))
        + 0.01 * graph_info["compactness_loss"]
        + 0.001 * graph_info["assignment_entropy_loss"]
        + 0.01 * graph_info["assignment_balance_loss"]
    )
    loss.backward()
    assignment_grad = model.backbone.learnable_graph.pixel_query.weight.grad
    graph_grad = model.backbone.learnable_graph.graph_query.weight.grad
    assert assignment_grad is not None and torch.isfinite(assignment_grad).all()
    assert graph_grad is not None and torch.isfinite(graph_grad).all()
    assert assignment_grad.abs().sum() > 0
    assert graph_grad.abs().sum() > 0
    print(
        "learned_graph: pixels={}, tokens={}, assignment_nnz={}, graph_nnz={}".format(
            pixel_count,
            graph_info["token_count"],
            graph_info["assignment_nnz"],
            graph_info["graph_nnz"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = MultiDatasetVCASAHGCN(
        {"demo_a": 4, "demo_b": 3},
        latent_channels=8,
        adapter_embed_dim=16,
        extra_channels=2,
        norm_type="domain_bn",
        locality_window=None,
    ).to(args.device).eval()

    run_case(model, "demo_a", bands=12, classes=4)
    run_case(model, "demo_b", bands=20, classes=3)
    run_learned_graph_case(args.device)
    print(f"SMOKE_TEST_OK: device={args.device}; static and learned graph checks passed.")


if __name__ == "__main__":
    main()
