"""FAST-style traffic models with legacy scan and official Mamba backends."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from data import STATE_CONTEXT_NAMES


class SpatialScan(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.parameters_for_step = nn.Linear(width, 3 * width)
        self.log_decay = nn.Parameter(torch.zeros(width))
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.project = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, segments: tuple[tuple[int, int], ...], reverse: bool) -> torch.Tensor:
        # Reverse inside each component, then scan all components together.
        order = None
        if reverse:
            order = torch.tensor([i for start, stop in segments for i in range(stop - 1, start - 1, -1)],
                                 device=x.device)
            x = x.index_select(2, order)
        count = x.shape[2]
        starts = torch.tensor([start for start, _ in segments], device=x.device)
        ends = torch.tensor([stop - 1 for _, stop in segments], device=x.device)
        left_mask = torch.ones(count, dtype=x.dtype, device=x.device)
        right_mask = torch.ones(count, dtype=x.dtype, device=x.device)
        left_mask[starts] = 0
        right_mask[ends] = 0

        # Equivalent to the depthwise width-3 convolution, with zeros at each
        # component boundary rather than borrowing a neighbor from another road.
        left = torch.cat((torch.zeros_like(x[:, :, :1]), x[:, :, :-1]), dim=2) * left_mask[None, None, :, None]
        right = torch.cat((x[:, :, 1:], torch.zeros_like(x[:, :, :1])), dim=2) * right_mask[None, None, :, None]
        weights = self.local.weight[:, 0]
        local = left * weights[:, 0] + x * weights[:, 1] + right * weights[:, 2] + self.local.bias

        delta, source, readout = self.parameters_for_step(x).chunk(3, -1)
        decay = torch.exp(-F.softplus(delta) * F.softplus(self.log_decay))
        emission = (1 - decay) * torch.tanh(source)
        decay = decay * left_mask[None, None, :, None]

        # Parallel prefix of affine recurrences: s_i = decay_i * s_(i-1) + emission_i.
        # A zero decay at each start resets the state between components.
        state = emission
        step = 1
        while step < count:
            next_state = state[:, :, step:] + decay[:, :, step:] * state[:, :, :-step]
            next_decay = decay[:, :, step:] * decay[:, :, :-step]
            state = torch.cat((state[:, :, :step], next_state), dim=2)
            decay = torch.cat((decay[:, :, :step], next_decay), dim=2)
            step *= 2
        scanned = state * torch.sigmoid(readout) * torch.sigmoid(local)
        if reverse:
            scanned = scanned.index_select(2, order)
        return self.project(scanned)


class FASTBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.time_a = nn.MultiheadAttention(width, heads, batch_first=True)
        self.time_b = nn.MultiheadAttention(width, heads, batch_first=True)
        self.forward_scan = SpatialScan(width)
        self.reverse_scan = SpatialScan(width)
        self.norm_a = nn.LayerNorm(width)
        self.norm_s = nn.LayerNorm(width)
        self.norm_b = nn.LayerNorm(width)

    @staticmethod
    def attend(x: torch.Tensor, attention: nn.MultiheadAttention) -> torch.Tensor:
        batch, steps, nodes, width = x.shape
        sequence = x.transpose(1, 2).reshape(batch * nodes, steps, width)
        attended, _ = attention(sequence, sequence, sequence, need_weights=False)
        return attended.reshape(batch, nodes, steps, width).transpose(1, 2)

    def forward(self, x: torch.Tensor, segments: tuple[tuple[int, int], ...], bidirectional: bool) -> torch.Tensor:
        x = self.norm_a(x + self.attend(x, self.time_a))
        spatial = self.forward_scan(x, segments, False)
        if bidirectional:
            spatial = 0.5 * (spatial + self.reverse_scan(x, segments, True))
        x = self.norm_s(x + spatial)
        return self.norm_b(x + self.attend(x, self.time_b))


class FASTEncoder(nn.Module):
    def __init__(self, features: int = 9, width: int = 64, blocks: int = 2, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.blocks = nn.ModuleList(FASTBlock(width, heads) for _ in range(blocks))

    def forward(self, x: torch.Tensor, segments: tuple[tuple[int, int], ...], bidirectional: bool = True) -> torch.Tensor:
        hidden = self.input(x)
        outputs = []
        for block in self.blocks:
            hidden = block(hidden, segments, bidirectional)
            outputs.append(hidden)
        return torch.stack(outputs).mean(0)


class StateModel(nn.Module):
    def __init__(self, width: int = 64, blocks: int = 2):
        super().__init__()
        self.encoder = FASTEncoder(width=width, blocks=blocks)
        self.head = nn.Linear(width + 3, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, links: torch.Tensor, station_context: torch.Tensor,
                station_to_link: torch.Tensor, segments: tuple[tuple[int, int], ...],
                bidirectional: bool = True) -> torch.Tensor:
        hidden = self.encoder(links, segments, bidirectional)[:, :, station_to_link]
        correction = self.head(torch.cat((hidden, station_context), dim=-1))
        return F.relu(station_context[..., :2] + correction)


class QueueModel(nn.Module):
    def __init__(self, width: int = 64, blocks: int = 2):
        super().__init__()
        self.encoder = FASTEncoder(width=width, blocks=blocks)
        self.head = nn.Linear(width, 12)

    def forward(self, history: torch.Tensor, segments: tuple[tuple[int, int], ...],
                bidirectional: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        last = self.encoder(history, segments, bidirectional)[:, -1]
        output = self.head(last).reshape(last.shape[0], last.shape[1], 6, 2).transpose(1, 2)
        return output[..., 0], F.softplus(output[..., 1])


class MambaSpatial(nn.Module):
    """Official Mamba over independently padded road components."""
    def __init__(self, width: int, backend: str, state: int = 64):
        super().__init__()
        try:
            if backend == "mamba1":
                from mamba_ssm import Mamba
                self.ssm = Mamba(d_model=width, d_state=state, d_conv=2, expand=1)
            else:
                from mamba_ssm import Mamba2
                # The torch-convolution path also works without causal-conv1d.
                self.ssm = Mamba2(d_model=width, d_state=state, d_conv=2, expand=1,
                                  headdim=min(32, width), chunk_size=32, use_mem_eff_path=False)
        except ImportError as error:
            raise RuntimeError("Install requirements-mamba.txt in WSL; no implicit scan fallback is used") from error
        self.layouts = {}

    def forward(self, x, segments, reverse=False):
        batch, steps, nodes, width = x.shape
        key = segments, reverse, x.device
        if key not in self.layouts:
            buckets = {}
            for start, stop in segments:
                length = 1 << max((stop - start - 1).bit_length(), 0)
                indices = list(range(start, stop))
                if reverse:
                    indices.reverse()
                buckets.setdefault(length, []).append(indices)
            layouts = []
            for length, components in sorted(buckets.items()):
                gather = torch.zeros((len(components), length), dtype=torch.long, device=x.device)
                valid = torch.zeros_like(gather, dtype=torch.bool)
                for row, indices in enumerate(components):
                    gather[row, :len(indices)] = torch.tensor(indices, device=x.device)
                    valid[row, :len(indices)] = True
                layouts.append((gather, valid, gather[valid]))
            self.layouts[key] = layouts
        out = torch.zeros_like(x)
        for gather, valid, destination in self.layouts[key]:
            packed = x[:, :, gather] * valid[None, None, :, :, None]
            packed = packed.reshape(batch * steps * len(gather), gather.shape[1], width).contiguous()
            scanned = self.ssm(packed).reshape(batch, steps, *gather.shape, width)
            out = out.index_copy(2, destination, scanned[:, :, valid].to(out.dtype))
        return out


class GraphResidual(nn.Module):
    def __init__(self, width, mode="mean"):
        super().__init__()
        if mode not in ("mean", "reliability"):
            raise ValueError(f"Unknown graph mode: {mode}")
        self.mode = mode
        self.up = nn.Linear(width, width, bias=False)
        self.down = nn.Linear(width, width, bias=False)
        if mode == "reliability":
            self.edge_gate = nn.Linear(2 * width + 14, 1)
            self.mix_gate = nn.Linear(width + 7, 1)
            nn.init.zeros_(self.edge_gate.weight)
            nn.init.zeros_(self.edge_gate.bias)
            nn.init.zeros_(self.mix_gate.weight)
            nn.init.constant_(self.mix_gate.bias, -1)

    def forward(self, x, edges, observation=None):
        out = torch.zeros_like(x)
        if self.mode == "reliability":
            if observation is None or observation.shape[-1] < 17:
                raise ValueError("Reliability graph needs the public channel masks, quality, and ages")
            masks = observation[..., 10:13].clamp(0, 1)
            quality = observation[..., 13:14].clamp(0, 1)
            ages = observation[..., 14:17].clamp(0, 1)
            evidence = torch.cat((masks, quality, ages), -1).to(x.dtype)
            confidence = masks.mean(-1, keepdim=True) * quality
        if edges.numel():
            for source, target, projection in ((edges[0], edges[1], self.up), (edges[1], edges[0], self.down)):
                degree = torch.bincount(target, minlength=x.shape[2]).clamp_min(1)
                neighbor = x.index_select(2, source)
                gathered = projection(neighbor)
                if self.mode == "reliability":
                    edge_input = torch.cat((neighbor, x.index_select(2, target),
                                            evidence.index_select(2, source), evidence.index_select(2, target)), -1)
                    weight = self.edge_gate(edge_input).sigmoid() * confidence.index_select(2, source)
                    gathered = gathered * weight
                # Keep absolute confidence: normalizing by weight sum would
                # cancel quality when a node has only one neighbor.
                gathered = gathered / degree[target][None, None, :, None]
                out = out.index_add(2, target, gathered.to(out.dtype))
        if self.mode == "reliability":
            out = out * self.mix_gate(torch.cat((x, evidence), -1)).sigmoid()
        return out


class TrafficBlock(nn.Module):
    def __init__(self, width, heads, dropout, backend, graph, graph_mode="mean"):
        super().__init__()
        self.time_a = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.time_b = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_a = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.ffn_b = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(5))
        cls = (lambda: SpatialScan(width)) if backend == "scan" else (lambda: MambaSpatial(width, backend))
        self.forward_scan, self.reverse_scan = cls(), cls()
        self.graph = GraphResidual(width, graph_mode) if graph else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, segments, edges, bidirectional, observation=None):
        x = self.norms[0](x + self.dropout(FASTBlock.attend(x, self.time_a)))
        x = self.norms[1](x + self.dropout(self.ffn_a(x)))
        spatial = self.forward_scan(x, segments, False)
        if bidirectional:
            spatial = 0.5 * (spatial + self.reverse_scan(x, segments, True))
        if self.graph is not None:
            spatial = spatial + self.graph(x, edges, observation)
        x = self.norms[2](x + self.dropout(spatial))
        x = self.norms[3](x + self.dropout(FASTBlock.attend(x, self.time_b)))
        return self.norms[4](x + self.dropout(self.ffn_b(x)))


def haar_split(x):
    """One orthonormal Haar level on time, replicating an odd final sample."""
    if x.shape[1] % 2:
        x = torch.cat((x, x[:, -1:]), 1)
    even, odd = x[:, ::2], x[:, 1::2]
    return (even + odd) / 2 ** 0.5, (even - odd) / 2 ** 0.5


def haar_merge(low, high, steps):
    return torch.stack(((low + high) / 2 ** 0.5, (low - high) / 2 ** 0.5), 2).flatten(1, 2)[:, :steps]


class WaveletResidual(nn.Module):
    """Learn separate trend/detail filters, then reconstruct every time step."""
    def __init__(self, width, levels):
        super().__init__()
        self.levels = levels
        self.bands = nn.ModuleList(nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, groups=width), nn.GELU(), nn.Conv1d(width, width, 1)
        ) for _ in range(levels + 1))
        self.gain = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def filter(x, band):
        batch, steps, nodes, width = x.shape
        y = band(x.permute(0, 2, 3, 1).reshape(batch * nodes, width, steps))
        return y.reshape(batch, nodes, width, steps).permute(0, 3, 1, 2)

    def forward(self, x):
        details, lengths = [], []
        low = x
        for _ in range(self.levels):
            lengths.append(low.shape[1])
            low, high = haar_split(low)
            details.append(high)
        result = self.filter(low, self.bands[0])
        for level in reversed(range(self.levels)):
            result = haar_merge(result, self.filter(details[level], self.bands[level + 1]), lengths[level])
        return self.gain.sigmoid() * result


class AdaptiveGraphResidual(nn.Module):
    """Panel-local learned dependencies, separate from physical road edges."""
    def __init__(self, width, neighbors):
        super().__init__()
        self.neighbors = neighbors
        rank = min(width, 16)
        self.source = nn.Linear(width, rank, bias=False)
        self.target = nn.Linear(width, rank, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width + 7, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)

    def layout(self, identities):
        count = identities.shape[0]
        # ponytail: dense scores fit current panels; use blocked top-k for larger networks.
        scores = self.target(identities) @ self.source(identities).T / self.source.out_features ** 0.5
        scores = scores.masked_fill(torch.eye(count, dtype=torch.bool, device=scores.device), -torch.inf)
        values, neighbors = scores.topk(min(self.neighbors, count - 1), dim=-1)
        return neighbors, values.softmax(-1)

    def forward(self, x, layout, observation):
        neighbors, weights = layout
        if neighbors.shape[1] == 0:
            return torch.zeros_like(x)
        evidence = observation[..., 10:17].clamp(0, 1).to(x.dtype)
        # Temporal inference remains usable when the selected node is currently missing.
        confidence = 0.25 + 0.75 * evidence[..., :3].mean(-1) * evidence[..., 3]
        message = torch.zeros_like(x)
        for neighbor in range(neighbors.shape[1]):
            source = neighbors[:, neighbor]
            weight = weights[:, neighbor].to(x.dtype)[None, None] * confidence.index_select(2, source)
            message = message + x.index_select(2, source) * weight[..., None]
        return self.gate(torch.cat((x, evidence), -1)).sigmoid() * self.project(message)


class TrafficEncoder(nn.Module):
    def __init__(self, features, node_count, width=64, blocks=2, heads=4,
                 dropout=0.1, backend="mamba2", graph=False, missing_aware=False, graph_mode="mean",
                 adaptive_graph_k=0, wavelet_levels=0, wavelet_input="embedded"):
        super().__init__()
        if width % heads or backend.startswith("mamba") and width % min(width, 32):
            raise ValueError("width must be divisible by attention heads and the Mamba head dimension")
        if graph_mode not in ("mean", "reliability") or graph_mode != "mean" and not graph:
            raise ValueError("Reliability graph mode requires graph=True")
        if (missing_aware or graph_mode == "reliability") and features < 17:
            raise ValueError("Missing-aware features require channel masks, quality, and ages")
        if adaptive_graph_k < 0 or adaptive_graph_k == 1 or not 0 <= wavelet_levels <= 3:
            raise ValueError("Adaptive graph neighbors must be 0 or at least 2; wavelet levels must lie in [0, 3]")
        if (adaptive_graph_k or wavelet_levels) and features < 31:
            raise ValueError("Adaptive graph and wavelet require the v2 observation/prior features")
        if wavelet_input not in ("embedded", "signals"):
            raise ValueError("Unknown wavelet input")
        # Configs without this field retain the original checkpoint architecture.
        self.wavelet_input = wavelet_input
        self.input = nn.Linear(features, width)
        self.node = nn.Embedding(node_count + 1, width, padding_idx=0)
        self.slot = nn.Embedding(288, width)
        self.weekday = nn.Embedding(7, width)
        self.relative = nn.Embedding(72, width)
        self.blocks = nn.ModuleList(TrafficBlock(width, heads, dropout, backend, graph, graph_mode) for _ in range(blocks))
        self.skip = nn.Linear(width * blocks, width)
        self.dropout = nn.Dropout(dropout)
        self.missing_embedding = nn.Parameter(torch.empty(3, width)) if missing_aware else None
        if self.missing_embedding is not None:
            nn.init.normal_(self.missing_embedding, std=0.02)
        self.adaptive_graph = AdaptiveGraphResidual(width, adaptive_graph_k) if adaptive_graph_k else None
        self.wavelet = WaveletResidual(3 if wavelet_input == "signals" else width, wavelet_levels) if wavelet_levels else None
        self.wavelet_project = nn.Linear(3, width, bias=False) if wavelet_levels and wavelet_input == "signals" else None

    def project_input(self, x, fill_missing=False):
        if self.missing_embedding is None and self.wavelet is None:
            return self.input(x)
        masks = x[..., 10:13].clamp(0, 1)
        clean = x.clone()
        for channel, position in enumerate((0, 1, 9)):
            fallback = x[..., 25 + channel] if fill_missing else 0
            clean[..., position] = torch.where(masks[..., channel] > 0, x[..., position], fallback)
        # Separate trainable tokens for missing speed, flow, and occupancy.
        result = self.input(clean)
        return result if self.missing_embedding is None else result + (1 - masks) @ self.missing_embedding

    def forward(self, x, node_ids, segments, edges, bidirectional=True):
        steps = x.shape[1]
        if steps > 72:
            raise ValueError("The input window must contain at most 72 steps")
        slot = (x[..., 28] * 287).round().long().clamp(0, 287)
        weekday = (x[..., 29] * 6).round().long().clamp(0, 6)
        projected = self.project_input(x)
        identities = self.node(node_ids)
        embedding = (identities[None, None].to(projected.dtype) + self.slot(slot).to(projected.dtype)
                     + self.weekday(weekday).to(projected.dtype)
                     + self.relative(torch.arange(steps, device=x.device))[None, :, None].to(projected.dtype))
        hidden = self.dropout(projected + embedding)
        if self.wavelet is not None:
            if self.wavelet_input == "signals":
                # DWT sees only continuous measurements; masks and embeddings stay in the main path.
                signals = torch.where(x[..., 10:13] > 0, x[..., (0, 1, 9)], x[..., 25:28])
                hidden = hidden + self.wavelet_project(self.wavelet(signals))
            else:
                hidden = hidden + self.wavelet(self.project_input(x, fill_missing=True) + embedding)
        layout = self.adaptive_graph.layout(identities) if self.adaptive_graph is not None else None
        outputs = []
        for block in self.blocks:
            hidden = block(hidden, segments, edges, bidirectional, x)
            if self.adaptive_graph is not None:
                hidden = hidden + self.adaptive_graph(hidden, layout, x)
            outputs.append(hidden)
        return self.skip(torch.cat(outputs, -1))


class TrafficStateModel(nn.Module):
    def __init__(self, station_decoder="linear", **config):
        super().__init__()
        if station_decoder not in ("linear", "gated", "split"):
            raise ValueError("Unknown station decoder")
        self.station_decoder = station_decoder
        self.encoder = TrafficEncoder(**config)
        width = config.get("width", 64)
        if station_decoder == "split":
            self.head = nn.ModuleList(nn.Sequential(nn.Linear(width + len(STATE_CONTEXT_NAMES), width),
                                                   nn.GELU(), nn.Linear(width, 1)) for _ in range(2))
            self.gate = nn.ModuleList(nn.Linear(width + len(STATE_CONTEXT_NAMES), 1) for _ in range(2))
            for head, gate in zip(self.head, self.gate):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -1)
        elif station_decoder == "gated":
            self.head = nn.Sequential(nn.Linear(width + len(STATE_CONTEXT_NAMES), width), nn.GELU(), nn.Linear(width, 2))
            self.gate = nn.Linear(width + len(STATE_CONTEXT_NAMES), 2)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -1)
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        else:
            self.head = nn.Linear(width + 3, 2)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x, context, station_link, node_ids, segments, edges, bidirectional=True):
        hidden = self.encoder(x, node_ids, segments, edges, bidirectional)[:, :, station_link]
        expected = len(STATE_CONTEXT_NAMES) if self.station_decoder != "linear" else 3
        if context.shape[-1] != expected:
            raise ValueError(f"Station decoder requires {expected} context features")
        combined = torch.cat((hidden, context), -1)
        if self.station_decoder == "split":
            correction = torch.cat([head(combined) * gate(combined).sigmoid()
                                    for head, gate in zip(self.head, self.gate)], -1)
        else:
            correction = self.head(combined)
            if self.station_decoder == "gated":
                correction = correction * self.gate(combined).sigmoid()
        return F.relu(context[..., :2] + correction)


class TrafficQueueModel(nn.Module):
    def __init__(self, station_decoder="linear", queue_head="linear", queue_changes=False, **config):
        super().__init__()
        if queue_head not in ("linear", "pooled-mlp"):
            raise ValueError("Unknown Queue head")
        self.queue_head = queue_head
        self.encoder = TrafficEncoder(**config)
        width = config.get("width", 64)
        self.head = (nn.Linear(width, 12) if queue_head == "linear" else
                     nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                   nn.Dropout(config.get("dropout", 0.1)), nn.Linear(width, 12)))

    def forward(self, x, node_ids, segments, edges, bidirectional=True):
        history = self.encoder(x, node_ids, segments, edges, bidirectional)
        hidden = history[:, -1]
        if self.queue_head == "pooled-mlp":
            hidden = torch.cat((hidden, history.mean(1)), dim=-1)
        output = self.head(hidden).reshape(hidden.shape[0], hidden.shape[1], 6, 2).transpose(1, 2)
        return output[..., 0], F.softplus(output[..., 1])
