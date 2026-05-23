"""SE-LeWM: Sensor-Evolving Latent World Model adapter.

Design constraint: do NOT touch the backbone. We sit *outside* a frozen
backbone like A2TTA's ResidualCalibrator does, and emit a small residual
`delta_y` that is gated by a learnable scalar initialized so that
`sigmoid(gate)` is tiny (~0.047 at gate=-3.0). This guarantees the model
behaves close to the frozen backbone at init and ramps up only if warmup /
adaptation finds the residual helps.

Pieces
------
  * ContextEncoder       : x history (per-node) -> z_ctx  [B,N,D_z]
  * GraphLatentTransition: rollout H steps in latent space using A_norm
                           -> z_pred_seq  [B,H,N,D_z]
  * TargetEncoder        : y future -> z_tgt_seq  [B,H,N,D_z]
                           (training / warmup / delayed-label only — NEVER at
                           inference time. The adapter exposes this through
                           `encode_target` and only the trainer / warmup code
                           calls it; `forward(...)` does not consume `y`.)
  * ResidualDecoder      : z_pred_seq -> delta_y  [B,N,H] or [B*N,H]
                           last layer zero-init, multiplied by sigmoid(gate).
"""
from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _sym_norm_adj(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric Laplacian-style normalization: D^{-1/2}(A+I)D^{-1/2}.

    Robust to:
      * already-row-normalized adjacency (e.g. eac repo passes that in)
      * zero-degree rows
      * non-square / non-finite input (returns identity in that case).
    """
    if adj is None:
        return None
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        # Defensive: build an identity of expected size from the caller.
        N = adj.shape[-1]
        return torch.eye(N, device=adj.device, dtype=adj.dtype)
    A = adj.float()
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    N = A.shape[0]
    A = A + torch.eye(N, device=A.device, dtype=A.dtype)
    d = A.sum(dim=-1).clamp_min(eps)
    d_inv_sqrt = d.pow(-0.5)
    A_norm = A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
    return A_norm


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def to_BNT(x: torch.Tensor, N: int) -> torch.Tensor:
    """Take whatever the trainer hands us and return `[B, N, T]`.

    Accepts `[B*N, T]`, `[B, N, T]`, or `[B, T, N]` (we detect the temporal axis
    using `N` since `N` is known per year).
    """
    if x.dim() == 2:
        # [B*N, T]  -- the eac repo convention.
        BN, T = x.shape
        if BN % N == 0:
            B = BN // N
            return x.reshape(B, N, T)
        raise ValueError(f"2D input shape {tuple(x.shape)} not divisible by N={N}")
    if x.dim() == 3:
        B, A, C = x.shape
        if A == N:
            return x  # [B, N, T]
        if C == N:
            return x.transpose(1, 2).contiguous()  # [B, T, N] -> [B, N, T]
        if A * B == 0 or A == 0:
            raise ValueError(f"3D shape {tuple(x.shape)} doesn't contain N={N}")
        # Last-resort: assume [B,N,T].
        return x
    raise ValueError(f"Unsupported input ndim={x.dim()} shape={tuple(x.shape)}")


# ---------------------------------------------------------------------------
# Context encoder: per-node temporal encoder + 1-step graph mixing
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    def __init__(
        self,
        x_len: int,
        z_dim: int,
        hidden_dim: int,
        num_nodes_max: int,
        node_emb_dim: int = 16,
        time_feat_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.x_len = x_len
        self.z_dim = z_dim
        self.node_emb_dim = node_emb_dim
        self.time_feat_dim = time_feat_dim
        self.num_nodes_max = num_nodes_max

        self.node_emb = nn.Parameter(
            torch.zeros(num_nodes_max, node_emb_dim).normal_(0.0, 0.02)
        )

        # Temporal stat features (last, mean, std, slope) — cheap, dataset-agnostic.
        n_stat = 4
        in_dim = x_len + n_stat + node_emb_dim + time_feat_dim
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc_h = nn.Linear(hidden_dim, hidden_dim)
        # After graph mixing, project to z_dim.
        self.fc_z = nn.Linear(hidden_dim, z_dim)

    def expand_nodes(self, new_num_nodes: int):
        if new_num_nodes <= self.num_nodes_max:
            return
        extra = new_num_nodes - self.num_nodes_max
        new_rows = torch.zeros(
            extra, self.node_emb_dim,
            dtype=self.node_emb.dtype, device=self.node_emb.device,
        ).normal_(0.0, 0.02)
        self.node_emb = nn.Parameter(torch.cat([self.node_emb.data, new_rows], dim=0))
        self.num_nodes_max = new_num_nodes

    @staticmethod
    def _temporal_stats(x_BNT: torch.Tensor) -> torch.Tensor:
        T = x_BNT.shape[-1]
        last = x_BNT[..., -1:]
        mean = x_BNT.mean(dim=-1, keepdim=True)
        std = x_BNT.std(dim=-1, unbiased=False, keepdim=True)
        t = torch.arange(T, device=x_BNT.device, dtype=x_BNT.dtype)
        t = t - t.mean()
        denom = (t * t).sum().clamp_min(1e-6)
        slope = ((x_BNT - mean) * t).sum(dim=-1, keepdim=True) / denom
        return torch.cat([last, mean, std, slope], dim=-1)  # [B,N,4]

    def forward(
        self,
        x_BNT: torch.Tensor,                  # [B, N, T_in]
        A_norm: torch.Tensor,                 # [N, N]
        time_feat: Optional[torch.Tensor] = None,   # [B, time_feat_dim] or [B,N,time_feat_dim]
    ) -> torch.Tensor:
        B, N, T = x_BNT.shape
        stats = self._temporal_stats(x_BNT)                # [B,N,4]
        idx = torch.arange(N, device=x_BNT.device).clamp_max(self.num_nodes_max - 1)
        emb = self.node_emb[idx].unsqueeze(0).expand(B, N, self.node_emb_dim)
        feats = [x_BNT, stats, emb]
        if self.time_feat_dim > 0 and time_feat is not None:
            tf = time_feat
            if tf.dim() == 2:
                tf = tf.unsqueeze(1).expand(B, N, -1)
            feats.append(tf)
        elif self.time_feat_dim > 0:
            feats.append(torch.zeros(B, N, self.time_feat_dim, device=x_BNT.device, dtype=x_BNT.dtype))
        h = torch.cat(feats, dim=-1)                       # [B,N, in_dim]
        h = self.drop(self.act(self.fc_in(h)))
        # Graph message passing (one round). A_norm @ h on the node axis.
        msg = torch.einsum("ij,bjd->bid", A_norm, h)
        h = self.drop(self.act(self.fc_h(h + msg)))
        z = self.fc_z(h)                                   # [B, N, z_dim]
        return z


# ---------------------------------------------------------------------------
# Graph-aware latent transition (rollout H steps)
# ---------------------------------------------------------------------------

class GraphLatentTransition(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        # GRUCell consumes concat(z, msg, step_emb)
        self.step_emb = nn.Embedding(horizon, z_dim)
        nn.init.normal_(self.step_emb.weight, std=0.02)
        cell_in = z_dim + z_dim + z_dim   # state, msg(A@z), step embedding
        self.cell = nn.GRUCell(cell_in, z_dim)

    def forward(self, z_ctx: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """z_ctx: [B, N, D_z]  ->  z_pred_seq: [B, H, N, D_z]."""
        B, N, D = z_ctx.shape
        H = self.horizon
        z = z_ctx
        outs = []
        # Flatten (B*N) for GRUCell which expects 2-D.
        for h in range(H):
            msg = torch.einsum("ij,bjd->bid", A_norm, z)         # [B,N,D]
            se = self.step_emb.weight[h].view(1, 1, -1).expand(B, N, D)
            inp = torch.cat([z, msg, se], dim=-1).reshape(B * N, -1)
            state = z.reshape(B * N, D)
            new_state = self.cell(inp, state)
            z = new_state.reshape(B, N, D)
            outs.append(z)
        return torch.stack(outs, dim=1)   # [B, H, N, D]


# ---------------------------------------------------------------------------
# Target encoder: y future -> z_tgt_seq (training-time only)
# ---------------------------------------------------------------------------

class TargetEncoder(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, num_nodes_max: int, node_emb_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.node_emb_dim = node_emb_dim
        self.num_nodes_max = num_nodes_max
        self.node_emb = nn.Parameter(torch.zeros(num_nodes_max, node_emb_dim).normal_(0.0, 0.02))
        # Per-horizon-step encoding: take a single scalar y_{B,N,h} -> latent.
        self.fc1 = nn.Linear(1 + node_emb_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, z_dim)

    def expand_nodes(self, new_num_nodes: int):
        if new_num_nodes <= self.num_nodes_max:
            return
        extra = new_num_nodes - self.num_nodes_max
        new_rows = torch.zeros(
            extra, self.node_emb_dim,
            dtype=self.node_emb.dtype, device=self.node_emb.device,
        ).normal_(0.0, 0.02)
        self.node_emb = nn.Parameter(torch.cat([self.node_emb.data, new_rows], dim=0))
        self.num_nodes_max = new_num_nodes

    def forward(self, y_BNH: torch.Tensor) -> torch.Tensor:
        """y_BNH: [B, N, H] -> z_tgt_seq: [B, H, N, D_z]."""
        B, N, H = y_BNH.shape
        idx = torch.arange(N, device=y_BNH.device).clamp_max(self.num_nodes_max - 1)
        emb = self.node_emb[idx]                              # [N, E]
        emb = emb.unsqueeze(0).unsqueeze(2).expand(B, N, H, self.node_emb_dim)
        y = torch.nan_to_num(y_BNH, nan=0.0).unsqueeze(-1)    # [B,N,H,1]
        h = torch.cat([y, emb], dim=-1)
        h = self.act(self.fc1(h))
        z = self.fc2(h)                                       # [B,N,H,D]
        return z.permute(0, 2, 1, 3).contiguous()             # [B,H,N,D]


# ---------------------------------------------------------------------------
# Residual decoder
# ---------------------------------------------------------------------------

class ResidualDecoder(nn.Module):
    """[B,H,N,D] -> [B,N,H] residual; last layer zero-init."""
    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, z_pred_seq: torch.Tensor) -> torch.Tensor:
        # z_pred_seq: [B, H, N, D]
        h = self.act(self.fc1(z_pred_seq))
        delta = self.fc_out(h).squeeze(-1)                   # [B, H, N]
        return delta.permute(0, 2, 1).contiguous()           # [B, N, H]


# ---------------------------------------------------------------------------
# SE-LeWM adapter
# ---------------------------------------------------------------------------

class SELeWMAdapter(nn.Module):
    """Frozen-backbone-friendly latent residual adapter.

    Configurable ablation flags (set via constructor or per-call kwargs):
      * use_residual : add `sigmoid(gate) * delta_y` to `y_base`.
                       False -> y_hat == y_base (used for "latent_only" /
                       "sigreg_only" probes where we want to study the latent
                       losses without disturbing predictions).
      * target_detach: stop-grad on z_tgt_seq before computing L_latent.
                       Off by default; flip on as an ablation.

    forward(...) does NOT consume y. Use `encode_target(y)` separately when
    you have access to ground-truth (training / warmup / delayed-label).
    """

    def __init__(
        self,
        num_nodes_max: int,
        x_len: int = 12,
        y_len: int = 12,
        z_dim: int = 64,
        hidden_dim: int = 128,
        node_emb_dim: int = 16,
        time_feat_dim: int = 0,
        residual_gate_init: float = -3.0,
        target_detach: bool = False,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes_max = num_nodes_max
        self.x_len = x_len
        self.y_len = y_len
        self.z_dim = z_dim
        self.target_detach = target_detach
        self.use_residual = use_residual

        self.context_encoder = ContextEncoder(
            x_len=x_len, z_dim=z_dim, hidden_dim=hidden_dim,
            num_nodes_max=num_nodes_max, node_emb_dim=node_emb_dim,
            time_feat_dim=time_feat_dim, dropout=dropout,
        )
        self.transition = GraphLatentTransition(z_dim=z_dim, hidden_dim=hidden_dim, horizon=y_len)
        self.target_encoder = TargetEncoder(
            z_dim=z_dim, hidden_dim=hidden_dim,
            num_nodes_max=num_nodes_max, node_emb_dim=node_emb_dim,
        )
        self.decoder = ResidualDecoder(z_dim=z_dim, hidden_dim=hidden_dim)
        # Learnable scalar gate; sigmoid(-3.0) ≈ 0.047 -> nearly identity at init.
        self.gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def expand_nodes(self, new_num_nodes: int):
        self.context_encoder.expand_nodes(new_num_nodes)
        self.target_encoder.expand_nodes(new_num_nodes)
        self.num_nodes_max = max(self.num_nodes_max, new_num_nodes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode_context_and_rollout(
        self,
        x_BNT: torch.Tensor,
        A_norm: torch.Tensor,
        time_feat: Optional[torch.Tensor] = None,
    ):
        z_ctx = self.context_encoder(x_BNT, A_norm, time_feat=time_feat)   # [B,N,D]
        z_pred_seq = self.transition(z_ctx, A_norm)                        # [B,H,N,D]
        return z_ctx, z_pred_seq

    def encode_target(self, y_BNH: torch.Tensor) -> torch.Tensor:
        z = self.target_encoder(y_BNH)
        if self.target_detach:
            z = z.detach()
        return z

    def forward(
        self,
        x: torch.Tensor,                          # [B*N, T_in] or [B,N,T_in]
        y_base: torch.Tensor,                     # same flat shape as backbone output
        adj: torch.Tensor,                        # [N, N]
        y: Optional[torch.Tensor] = None,         # delayed/training labels (NEVER at inference)
        time_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,      # [B*N, H] / [B,N,H] / [B,N] etc.
        mode: str = "train",                      # "train" | "eval"
        N: Optional[int] = None,
    ) -> dict:
        if N is None:
            # Try to recover N from adj.
            N = adj.shape[0]

        x_BNT = to_BNT(x, N)
        H = self.y_len
        # y_base may arrive as [B*N, H] or [B,N,H]; canonicalize.
        if y_base.dim() == 2:
            BN, Ho = y_base.shape
            B = BN // N
            y_base_BNH = y_base.reshape(B, N, Ho)
        else:
            y_base_BNH = y_base
            B = y_base_BNH.shape[0]

        A_norm = _sym_norm_adj(adj)

        z_ctx, z_pred_seq = self.encode_context_and_rollout(x_BNT, A_norm, time_feat=time_feat)
        delta_BNH = self.decoder(z_pred_seq)                  # [B,N,H]

        gate_val = torch.sigmoid(self.gate)
        if self.use_residual:
            y_hat_BNH = y_base_BNH + gate_val * delta_BNH
        else:
            y_hat_BNH = y_base_BNH

        # Latent target loss (training/warmup/delayed-label only).
        z_tgt_seq = None
        latent_loss = y_base_BNH.new_zeros(())
        if y is not None and mode != "eval":
            if y.dim() == 2:
                y_BNH = y.reshape(B, N, H)
            else:
                y_BNH = y
            z_tgt_seq = self.encode_target(y_BNH)             # [B,H,N,D]
            # Build a finite-value mask on y for the latent loss, expand to [B,H,N].
            if mask is None:
                m_BNH = torch.isfinite(y_BNH).float()         # [B,N,H]
            else:
                m_BNH = mask if mask.dim() == 3 else mask.reshape(B, N, H)
                m_BNH = m_BNH.float()
            m_BHN = m_BNH.permute(0, 2, 1).unsqueeze(-1)      # [B,H,N,1]
            diff2 = (z_pred_seq - z_tgt_seq) ** 2 * m_BHN
            denom = m_BHN.sum().clamp_min(1.0) * z_pred_seq.shape[-1]
            latent_loss = diff2.sum() / denom

        # Flat output to match repo's [B*N, H] convention.
        y_hat_flat = y_hat_BNH.reshape(B * N, H)
        delta_flat = delta_BNH.reshape(B * N, H)
        return {
            "prediction": y_hat_flat,
            "prediction_BNH": y_hat_BNH,
            "base_prediction": y_base if y_base.dim() == 2 else y_base.reshape(B * N, H),
            "delta": delta_flat,
            "z_ctx": z_ctx,
            "z_pred_seq": z_pred_seq,
            "z_tgt_seq": z_tgt_seq,
            "gate": gate_val,
            "loss_components": {
                "latent": latent_loss,
                "gate": gate_val.detach(),
            },
        }
