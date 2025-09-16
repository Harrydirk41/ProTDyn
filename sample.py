import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import glob
import random
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy

import mdtraj as md
from tqdm import tqdm
from huggingface_hub import login

from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils import encoding
from esm.models.esm3 import ESM3
import functools
import einops


from custom_rotary import RotaryEmbedding

import argparse
import yaml
from huggingface_hub import hf_hub_download
three_to_one = {
            "UNK": 'X', #3
            'LEU': 'L',  # 4
            'ALA': 'A',  # 5
            'GLY': 'G',  # 6
            'VAL': 'V',  # 7
            'SER': 'S',  # 8
            'GLU': 'E',  # 9
            'ARG': 'R',  # 10
            'THR': 'T',  # 11
            'ILE': 'I',  # 12
            'ASP': 'D',  # 13
            'PRO': 'P',  # 14
            'LYS': 'K',  # 15
            'GLN': 'Q',  # 16
            'ASN': 'N',  # 17
            'PHE': 'F',  # 18
            'TYR': 'Y',  # 19
            'MET': 'M',  # 20
            'HIS': 'H',  # 21
            'TRP': 'W',  # 22
            'CYS': 'C',  # 23
        }

class ProTDynDataset(Dataset):
    def __init__(self, sequence_list,sequence_token_list,structure_token_first_list,structure_token_last_list):
        self.sequence_list = sequence_list
        self.sequence_token_list = sequence_token_list
        self.structure_token_first_list = structure_token_first_list
        self.structure_token_last_list = structure_token_last_list
    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        return {'sequence': self.sequence_list[idx], 'sequence_token': self.sequence_token_list[idx],
               'structure_token_first_list':self.structure_token_first_list[idx],'structure_token_last_list':self.structure_token_last_list[idx]}

def custom_collate_fn(batch):
    seq_list = [item['sequence_token'] for item in batch]
    structure_token_first_list = [item['structure_token_first_list'] for item in batch]
    structure_token_last_list = [item['structure_token_last_list'] for item in batch]
    sequence = batch[0]["sequence"]
    padded_seq = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=1)
    if structure_token_first_list[0] is None:
        padded_structure_tokens_start = None
    else:
        padded_structure_tokens_start = torch.nn.utils.rnn.pad_sequence(structure_token_first_list, batch_first=True, padding_value=1)
    if structure_token_last_list[0] is None:
        padded_structure_tokens_end = None
    else:
        padded_structure_tokens_end = torch.nn.utils.rnn.pad_sequence(structure_token_last_list, batch_first=True, padding_value=1)
    return {'sequence_token': padded_seq,"sequence":sequence,'structure_tokens_start':padded_structure_tokens_start,'structure_tokens_end':padded_structure_tokens_end}
def RegressionHead(
    d_model: int, output_dim: int, hidden_dim: int | None = None
) -> nn.Module:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )

class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)
def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )
def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )
class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,ffn_type="gelu",
        expansion_ratio: float = 4.0,scaling_factor = 1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,n_heads,bias,qk_layernorm)

        self.scaling_factor = scaling_factor
        if ffn_type == "swiglu":
            self.ffn_seq = swiglu_ln_ffn(d_model, expansion_ratio, bias)
            self.ffn_struc = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn_seq = gelu_ln_ffn(d_model, expansion_ratio, bias)
            self.ffn_struc = gelu_ln_ffn(d_model, expansion_ratio, bias)
    def forward(self,x_seq,x_struc,mask,traj_sample = False,T = 1):
        r1_seq,r1_struc = self.attn(x_seq, x_struc,mask,traj_sample,T)
        x_seq = x_seq + r1_seq / self.scaling_factor
        x_struc = x_struc + r1_struc / self.scaling_factor
        
        r3_seq = self.ffn_seq(x_seq)
        x_seq = x_seq + r3_seq / self.scaling_factor
        r3_struc = self.ffn_struc(x_struc) 
        x_struc = x_struc + r3_struc / self.scaling_factor
        return x_seq,x_struc
class Transformerstack(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,n_layers=24):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,bias,qk_layernorm) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, bias=False)
    def forward(self,x_seq,x_struc,mask,traj_sample = False,T = 1):
        for block in self.blocks:
            x_seq,x_struc = block(x_seq,x_struc,mask,traj_sample,T)
        return self.norm(x_struc), x_struc

class ESMDynamics(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,n_layers=24):
        super().__init__()
        self.transformer = Transformerstack(d_model,n_heads,bias,qk_layernorm,n_layers)
        self.structure_head = RegressionHead(d_model, 4096)
        self.seq_embed = nn.Embedding(64, d_model)
        self.struc_embed = nn.Embedding(4096 + 5, d_model)

    def forward(self,seq_tokens,struc_tokens,mask,traj_sample = False,T = 1):
        N_seq = seq_tokens.shape[1]
        N_struc = int(struc_tokens.shape[1] / seq_tokens.shape[1])
        x_seq = self.seq_embed(seq_tokens)
        x_struc = self.struc_embed(struc_tokens)
        x_struc_out,x_struc_out_embed = self.transformer(x_seq,x_struc,mask,traj_sample,T)
        
        structure_logits = self.structure_head(x_struc_out)
        return structure_logits
class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv_seq = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.layernorm_qkv_struc = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj_seq = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_struc = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln_seq = nn.LayerNorm(d_model, bias=bias)
            self.k_ln_seq = nn.LayerNorm(d_model, bias=bias)
            self.q_ln_struc = nn.LayerNorm(d_model, bias=bias)
            self.k_ln_struc = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln_seq = nn.Identity()
            self.k_ln_seq = nn.Identity()
            self.q_ln_struc = nn.Identity()
            self.k_ln_struc = nn.Identity()

        self.rotary_seq = RotaryEmbedding(d_model // n_heads)
        self.rotary_struc = RotaryEmbedding(d_model // n_heads)
        self.rotary_time = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor,use_struc = True,N_seq = None,N_seg = None, T = None):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        if use_struc:
            position_res = torch.arange(N_seq).to(q.device)
            position_res = position_res.repeat(N_seg)
            q, k = self.rotary_struc(q, k,positions=position_res)
            position_T = torch.arange(N_seg).to(q.device) * T
            position_T = position_T.repeat_interleave(N_seq)
            q, k = self.rotary_time(q, k,positions=position_T)
        else:
            q, k = self.rotary_seq(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k
    def make_autoregressive_attn_mask(self,batch_mask: torch.Tensor, N_seg: int, *, as_float: bool = True):
        """
        Build [B, 1, L, L] mask where L = (N_seg+1)*N_res.
        - Autoregressive (causal)
        - Keys at masked positions are never attended
        - Queries at masked positions cannot attend to anything (whole row blocked)
        
        Args
        ----
        batch_mask: [B, N_res], 1=valid, 0=masked
        N_seg:      total length = (N_seg+1) * N_res
        as_float:   True -> additive mask (0 / -inf), False -> boolean (True=blocked)
        """
        B, N_res = batch_mask.shape
        L = (N_seg + 1) * N_res
        device = batch_mask.device
    
        # causal: [1,1,L,L]
        causal = torch.ones((L, L), dtype=torch.bool, device=device).tril().view(1,1,L,L)
    
        # tile per-segment validity
        tiled = batch_mask.repeat(1, N_seg + 1).bool()         # [B, L]
        key_valid = tiled.view(B, 1, 1, L)                     # [B,1,1,L]
        q_valid   = tiled.view(B, 1, L, 1)                     # [B,1,L,1]
    
        # allowed if: causal AND key is valid AND query is valid
        allowed = causal & key_valid & q_valid                 # [B,1,L,L]
        return allowed
    

    def forward(self, x_seq, x_struc,mask,traj_sample = False,T = 1):
        N_seq = x_seq.shape[1]
        N_seg = int(x_struc.shape[1] / x_seq.shape[1])
        if N_seg == 1:
            traj_sample = False
        qkv_BLD3_seq = self.layernorm_qkv_seq(x_seq)
        query_BLD_seq, key_BLD_seq, value_BLD_seq = torch.chunk(qkv_BLD3_seq, 3, dim=-1)
        query_BLD_seq, key_BLD_seq = (
            self.q_ln_seq(query_BLD_seq).to(query_BLD_seq.dtype),
            self.k_ln_seq(key_BLD_seq).to(query_BLD_seq.dtype),
        )
        query_BLD_seq, key_BLD_seq = self._apply_rotary(query_BLD_seq, key_BLD_seq,use_struc = False)

        qkv_BLD3_struc = self.layernorm_qkv_struc(x_struc)
        query_BLD_struc, key_BLD_struc, value_BLD_struc = torch.chunk(qkv_BLD3_struc, 3, dim=-1)
        query_BLD_struc, key_BLD_struc = (
            self.q_ln_struc(query_BLD_struc).to(query_BLD_struc.dtype),
            self.k_ln_struc(key_BLD_struc).to(query_BLD_struc.dtype),
        )
        query_BLD_struc, key_BLD_struc = self._apply_rotary(query_BLD_struc, key_BLD_struc,use_struc = True,N_seq = N_seq,N_seg = N_seg, T = T)
        if traj_sample:
            query_BLD_struc = torch.cat([query_BLD_struc[:,:N_seq],query_BLD_struc[:,-N_seq:],query_BLD_struc[:,N_seq:-N_seq]],dim=1)
            key_BLD_struc = torch.cat([key_BLD_struc[:,:N_seq],key_BLD_struc[:,-N_seq:],key_BLD_struc[:,N_seq:-N_seq]],dim=1)
            value_BLD_struc = torch.cat([value_BLD_struc[:,:N_seq],value_BLD_struc[:,-N_seq:],value_BLD_struc[:,N_seq:-N_seq]],dim=1)
        N_struc = int(query_BLD_struc.shape[1] / x_seq.shape[1])
        query_BLD_input_struc = torch.cat([query_BLD_seq,query_BLD_struc],dim = 1)
        key_BLD_input_struc = torch.cat([key_BLD_seq,key_BLD_struc],dim = 1)
        value_BLD_input_struc = torch.cat([value_BLD_seq,value_BLD_struc],dim = 1)
        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)
    
        query_BHLD_input_struc, key_BHLD_input_struc, value_BHLD_input_struc = map(reshaper, (query_BLD_input_struc,key_BLD_input_struc, value_BLD_input_struc))

        
        mask_BHLL_struc = self.make_autoregressive_attn_mask(mask, N_struc, as_float=True).bool()
        context_BHLD_struc = F.scaled_dot_product_attention(
            query_BHLD_input_struc, key_BHLD_input_struc, value_BHLD_input_struc, mask_BHLL_struc
        )
        context_BHLD_struc = einops.rearrange(context_BHLD_struc, "b h s d -> b s (h d)")
        out_struc = self.out_proj_struc(context_BHLD_struc[:,N_seq:])
        out_seq = self.out_proj_seq(context_BHLD_struc[:,:N_seq])

        if traj_sample:
            out_struc = torch.cat([out_struc[:,:N_seq],out_struc[:,2*N_seq:],out_struc[:,N_seq:2*N_seq]],dim=1)
        return out_seq,out_struc

class ESM3LightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=4099) 
    def forward(self, seq_tokens, struc_tokens,mask,traj_sample = False,T = 1):
        structure_logits = self.model(seq_tokens=seq_tokens, struc_tokens=struc_tokens,mask = mask,traj_sample = traj_sample,T = T)
        return structure_logits
    
    def sample_i_with_temperature(self,logits, i, temperature=1):
        logits_i = logits[:, i, :]
        scaled_logits = logits_i / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled_idx
    def predict_step(self, batch, batch_idx):
        folder = self.folder
        merge_folder = self.merge_folder
        traj_sample = self.traj_sample
        T = self.T
        sequence_tokens = batch['sequence_token'].to(self.device)
        structure_tokens_start = batch['structure_tokens_start']
        if structure_tokens_start is not None:
            structure_tokens_start = structure_tokens_start.to(self.device)
        structure_tokens_end = batch['structure_tokens_end']
        if structure_tokens_end is not None:
            structure_tokens_end = structure_tokens_end.to(self.device)

        N_batch = sequence_tokens.shape[0]
        N_seq = sequence_tokens.shape[1]
        N_seg = self.num_seg
        
        structure_tokens = torch.zeros_like(sequence_tokens)
        structure_tokens[:,0] = 4098
        structure_tokens[:,-1] = 4097
        structure_tokens = structure_tokens.repeat(1,N_seg)
            
        sequence = batch['sequence']
        name = self.name
        start = 1
        end = N_seq - 1
            
        mask = (sequence_tokens != 1).to(structure_tokens.device)
        if structure_tokens_start is not None:
            start_seg = 1
        else:
            start_seg = 0
        if structure_tokens_end is not None:
            end_seg = N_seg - 1
        else:
            end_seg = N_seg
        for i in range(start_seg,end_seg):
            for j in range(start,end):
                if i < 10:
                    if not traj_sample:
                        start_index = 0
                        end_index = (i + 1) * N_seq
                        sample_index = i * N_seq + j
                    else:
                        start_index = 0
                        end_index = N_seg * N_seq
                        sample_index = i * N_seq + j
                else:
                    start_index = (i - 9) * N_seq
                    end_index = (i + 1) * N_seq
                    sample_index = 9 * N_seq + j
                token_index = i * N_seq + j
                logits = self.forward(seq_tokens=sequence_tokens, 
                                  struc_tokens=structure_tokens[:,start_index:end_index],
                                 mask = mask,
                                 traj_sample = traj_sample,
                                 T = T)
                sample_token = self.sample_i_with_temperature(logits,sample_index - 1)
                structure_tokens[:,token_index] = sample_token
            structure_tokens[:,i * N_seq:(i + 1) * N_seq] = (
            structure_tokens[:,i * N_seq:(i + 1) * N_seq].where(sequence_tokens != 0, 4098)  # BOS
            .where(sequence_tokens != 2, 4097)  # EOS
            .where(sequence_tokens != 31, 4100)  # Chainbreak
            )
        for i in range(N_seg):
            bb_coords = (
                self.decoder.decode(
                    structure_tokens[:,i * N_seq:(i + 1) * N_seq],
                    torch.ones_like(sequence_tokens),
                    torch.zeros_like(sequence_tokens),
                )["bb_pred"]
                .detach()
                .cpu()
            )
            for j in range(N_batch):
                chain = ProteinChain.from_backbone_atom_coordinates(
                bb_coords[j:j+1], sequence="X" + sequence + "X"
                )
                chain.infer_oxygen().to_pdb(f"{folder}/{name}_{batch_idx}_{j}_{i}.pdb")
def get_coord(traj):
    atom_names = ['N', 'CA', 'C']
    top = traj.topology
    
    # Get atom indices in per-residue blocks
    residues = list(top.residues)
    N_res = len(residues)
    N_frame = traj.n_frames
    
    # Get atom indices for each (residue, atom_name)
    indices = np.array([
        [next(a.index for a in res.atoms if a.name == name) for name in atom_names]
        for res in residues
    ])  # shape (N_res, 5)
    print(indices[:2])
    # Now extract coordinates using advanced indexing
    # traj.xyz: shape (N_frame, N_atoms, 3)
    # We want: (N_frame, N_res, 5, 3)
    
    coords = traj.xyz[:, indices, :]  # shape (N_frame, N_res, 5, 3)
    return torch.from_numpy(coords)
def encode_xtc(pdb_dir,xtc_dir = None,encoder = None,device="cuda"):
    if xtc_dir is not None:
        traj = md.load(xtc_dir,top = pdb_dir)
    else:
        traj = md.load(pdb_dir)
    npy = get_coord(traj)[:3]
    coords = torch.zeros(npy.shape[0],npy.shape[1],37,3)
    coords[:,:,:3,:] = npy[:,:,:3,:]
    coords = coords.to(device)
    diff = coords[:, 1:,1, :] - coords[:, :-1,1, :]  # shape: (N_batch, N_res - 1, 3)
    dist = torch.norm(diff, dim=-1)   # shape: (N_batch, N_res - 1)
    mean_consecutive_dist = dist.mean()  # scalar
    if mean_consecutive_dist < 1:
        coords = coords * 10
    coords[:, :, 3:, :] = float('nan')
    batch_num = 50
    initial = 0
    structure_tokens_list = []
    while initial < coords.shape[0]:
        _, structure_tokens = encoder.encode(coords[initial:initial+batch_num])
        structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
        structure_tokens[:, 0] = 4098
        structure_tokens[:, -1] = 4097
        structure_tokens_list.append(structure_tokens)
        initial += batch_num
    structure_tokens = torch.cat(structure_tokens_list,dim = 0)
    return structure_tokens.detach().cpu()
def sample(tag = "thermo",num_sample = 1000,batch_size = 1,sequence = None, pdb_dir = None,xtc_dir = None,use_initial_struc = False,timestep = 10,num_step = 100,sample_dir = None,ckpt_dir = None):
    ckpt_path = hf_hub_download(
    repo_id="harrydirk41/ProTDyn-model",
    filename="model.ckpt",
    cache_dir=ckpt_dir)
    # Initialize your model
    
    model = ESM3_sm_open_v0("cpu").train()
    backbone_model = ESMDynamics(1536,24)
    backbone_model.seq_embed = model.encoder.sequence_embed
    backbone_model.struc_embed = model.encoder.structure_tokens_embed
    lightning_model = ESM3LightningModule.load_from_checkpoint(ckpt_path,model = backbone_model)
    lightning_model.decoder = ESM3_structure_decoder_v0("cuda")
    encoder = ESM3_structure_encoder_v0("cuda")

    if pdb_dir is not None:
        pdb = md.load(pdb_dir)
        sequence = ''.join(
                three_to_one.get(res.name, 'X') for res in pdb.topology.residues
            )
    sequence_tokens = encoding.tokenize_sequence(
                sequence, model.tokenizers.sequence, add_special_tokens=True)
    
    if tag == "thermo":
        traj_sample = False
        num_step = 1
        structure_tokens = None
    elif tag == "dynamics":
        traj_sample = False
        if use_initial_struc:
            if pdb_dir is None:
                raise ValueError("pdb_dir must be provided when use_initial_struc=True")
            structure_tokens = encode_xtc(pdb_dir,xtc_dir,encoder = encoder)
        else:
            structure_tokens = None
    elif tag == "dynamics_inpaint":
        traj_sample = True
        if pdb_dir is None or xtc_dir is None:
            raise ValueError("pdb_dir and xtc_dir must be provided in dynamics_inpaint module")
        structure_tokens = encode_xtc(pdb_dir,xtc_dir,encoder = encoder)
        num_sample = (structure_tokens.shape[0] - 1) * num_sample

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lightning_model.name = time_str
    lightning_model.traj_sample = traj_sample
    lightning_model.T = timestep
    lightning_model.num_seg = num_step
    folder = os.path.join(sample_dir,time_str,"separate")
    merge_folder = os.path.join(sample_dir,time_str,  "merge",tag)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(merge_folder, exist_ok=True)
    lightning_model.merge_folder= merge_folder
    lightning_model.folder = folder
    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,  # Adjust for your GPU setup
        num_nodes = 1
    )
    # Training
    if tag == "thermo":
        structure_token_first_list = [None] * num_sample
        structure_token_last_list = [None] * num_sample
    elif tag == "dynamics":
        if structure_tokens is None:
            structure_token_first_list = [None] * num_sample
            structure_token_last_list = [None] * num_sample
        else:
            structure_token_first_list = structure_tokens.repeat_interleave(num_sample, dim=0)
            num_struc = structure_tokens.shape[0]
            num_sample = num_struc * num_sample
            structure_token_last_list = [None] * num_sample
    elif tag == "dynamics_inpaint":
        num_struc = structure_tokens.shape[0]
        structure_token_first_list = structure_tokens[:-1].repeat_interleave(num_sample, dim=0)
        structure_token_last_list = structure_tokens[1:].repeat_interleave(num_sample, dim=0)
        num_sample = (num_struc - 1) * num_sample
        
    dataset = ProTDynDataset(sequence_list = [sequence] * num_sample,sequence_token_list = [sequence_tokens] *num_sample,structure_token_first_list=structure_token_first_list,structure_token_last_list=structure_token_last_list)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,num_workers=9)
    trainer.predict(lightning_model, train_loader)

    if tag == "thermo":
        pdb_files = sorted(glob.glob(os.path.join(folder, "*.pdb")))

        if not pdb_files:
            raise FileNotFoundError(f"No .pdb files found in {folder}")
        
        # Load and merge trajectories
        traj_list = [md.load(pdb) for pdb in pdb_files]
        merged_traj = md.join(traj_list)
        
        # Save merged trajectory as XTC
        xtc_path = os.path.join(merge_folder, "thermo_merged.xtc")
        merged_traj.save_xtc(xtc_path)
        
        # Save first frame as PDB
        pdb_path = os.path.join(merge_folder, "thermo_merged.pdb")
        merged_traj[0].save_pdb(pdb_path)
        
        # Delete original pdb files in folder
        for pdb_file in pdb_files:
            os.remove(pdb_file)
        
        print(f"Merged {len(pdb_files)} PDB files into {xtc_path}")
        print(f"Saved first frame as {pdb_path}")
        print(f"Deleted original {len(pdb_files)} PDB files from {folder}")
    else:
        count_n = 0
        for k in range(int(num_sample / batch_size)):
            for j in range(batch_size):
                trajs = []
                trajs_list = []
                for i in range(num_step):
                    pdb_path = os.path.join(folder, f"{lightning_model.name}_{k}_{j}_{i}.pdb")
                    if os.path.exists(pdb_path):
                        trajs.append(md.load(pdb_path))
                        trajs_list.append(pdb_path)
                if not trajs:
                    raise FileNotFoundError("No matching PDB files found.")
                merged = md.join(trajs)
                xtc_path = os.path.join(
    merge_folder, f"dynamics_merged_{count_n}_{timestep}_{num_step}.xtc"
)
                merged.save_xtc(xtc_path)
                pdb_path = os.path.join(
    merge_folder, f"dynamics_merged_{count_n}_{timestep}_{num_step}.pdb"
)
                merged[0].save_pdb(pdb_path)
                count_n+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="thermo_config.yml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sample(**cfg)

    
