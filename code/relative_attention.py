""" implementation of transformer encoder with relative attention
    references:
        - https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a
        - https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        - https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
        - https://github.com/jiezouguihuafu/ClassicalModelreproduced/blob/main/Transformer/transfor_rpe.py
"""

import copy
from os.path import basename, dirname, join, isfile
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Dropout, LayerNorm
import time
import networkx as nx

from . import structure
from . import models


class RelativePosition3D(nn.Module):
    """ Contact map-based relative position embeddings """

    #  need to compute a bucket_mtx for each structure
    #  need to know which bucket_mtx to use when grabbing the embeddings in forward()
    #   - on init, get a list of all PDB files we will be using
    #   - use a dictionary to store PDB files --> bucket_mtxs
    #   - forward() gets a new arg: the pdb file, which indexes into the dictionary to grab the right bucket_mtx
    def __init__(self,
                 embedding_len: int,
                 contact_threshold: int,
                 clipping_threshold: int,
                 pdb_fns: Optional[Union[str, list, tuple]] = None,
                 default_pdb_dir: str = "data/pdb_files"):

        # preferably, pdb_fns contains full paths to the PDBs, but if just the PDB filename is given
        # then it defaults to the path data/pdb_files/<pdb_fn>
        super().__init__()
        self.embedding_len = embedding_len
        self.clipping_threshold = clipping_threshold
        self.contact_threshold = contact_threshold
        self.default_pdb_dir = default_pdb_dir

        # dummy buffer for getting correct device for on-the-fly bucket matrix generation
        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)

        # for 3D-based positions, the number of embeddings is generally the number of buckets
        # for contact map-based distances, that is clipping_threshold + 1
        num_embeddings = clipping_threshold + 1

        # this is the embedding lookup table E_r
        self.embeddings_table = nn.Embedding(num_embeddings, embedding_len)

        # set up pdb_fns that were passed in on init (can also be set up during runtime in forward())
        # todo: i'm using a workaround to move the bucket_mtxs to the correct device
        #   i tried to make it more efficient by registering bucket matrices as buffers, but i was
        #   having problems with DDP syncing the buffers across processes, potentially caused by
        #   pdbs being out of order in the datamodule.
        self.bucket_mtxs = {}
        self.bucket_mtxs_device = self.dummy_buffer.device
        self._init_pdbs(pdb_fns)

    def forward(self, pdb_fn):
        # compute matrix R by grabbing the embeddings from the embeddings lookup table
        embeddings = self.embeddings_table(self._get_bucket_mtx(pdb_fn))
        return embeddings

    def _move_bucket_mtxs(self, device):
        for k, v in self.bucket_mtxs.items():
            self.bucket_mtxs[k] = v.to(device)
        self.bucket_mtxs_device = device

    def _get_bucket_mtx(self, pdb_fn):
        """ retrieve a bucket matrix given the pdb_fn.
            if the pdb_fn was provided at init or has already been computed, then the bucket matrix will be
            retrieved from the bucket_mtxs dictionary. else, it will be computed now on-the-fly """

        # ensure that all the bucket matrices are on the same device as the nn.Embedding
        if self.bucket_mtxs_device != self.dummy_buffer.device:
            self._move_bucket_mtxs(self.dummy_buffer.device)

        pdb_attr = self._pdb_key(pdb_fn)
        if pdb_attr in self.bucket_mtxs:
            return self.bucket_mtxs[pdb_attr]
        else:
            # encountering a new PDB at runtime... process it
            # if there's a new PDB at runtime, it will be initialized separately in each instance
            # of RelativePosition3D, for each layer. It would be more efficient to have a global
            # bucket_mtx registry... perhaps in the RelativeTransformerEncoder class, that can be passed through
            self._init_pdb(pdb_fn)
            return self.bucket_mtxs[pdb_attr]

    def _set_bucket_mtx(self, pdb_fn, bucket_mtx):
        """ store a bucket matrix in the bucket dict """

        # move the bucket_mtx to the same device that the other bucket matrices are on
        bucket_mtx = bucket_mtx.to(self.bucket_mtxs_device)

        self.bucket_mtxs[self._pdb_key(pdb_fn)] = bucket_mtx

    @staticmethod
    def _pdb_key(pdb_fn):
        """ return a unique key for the given pdb_fn, used to map unique PDBs """
        # note this key does NOT currently support PDBs with the same basename but different paths
        # assumes every PDB is in the format <pdb_name>.pdb
        # should be a compatible with being a class attribute, as it is used as a pytorch buffer name
        return f"pdb_{basename(pdb_fn).split('.')[0]}"

    def _init_pdbs(self, pdb_fns):
        start = time.time()

        if pdb_fns is None:
            # nothing to initialize if pdb_fns is None
            return

        # make sure pdb_fns is a list
        if not isinstance(pdb_fns, list) and not isinstance(pdb_fns, tuple):
            pdb_fns = [pdb_fns]

        # init each pdb fn in the list
        for pdb_fn in pdb_fns:
            self._init_pdb(pdb_fn)

        # print("Initialized PDB bucket matrices in: {:.3f}".format(time.time() - start))

    def _init_pdb(self, pdb_fn):
        """ process a pdb file for use with structure-based relative attention """
        # if pdb_fn is not a full path, default to the path data/pdb_files/<pdb_fn>
        if dirname(pdb_fn) == "":
            # handle the case where the pdb file is in the current working directory
            # if there is a PDB file in the cwd.... then just use it as is. otherwise, append the default.
            if not isfile(pdb_fn):
                pdb_fn = join(self.default_pdb_dir, pdb_fn)

        # create a structure graph from the pdb_fn and contact threshold
        cbeta_mtx = structure.cbeta_distance_matrix(pdb_fn)
        structure_graph = structure.dist_thresh_graph(cbeta_mtx, self.contact_threshold)

        # bucket_mtx indexes into the embedding lookup table to create the final distance matrix
        bucket_mtx = self._compute_bucket_mtx(structure_graph)

        self._set_bucket_mtx(pdb_fn, bucket_mtx)

    def _compute_bucketed_neighbors(self, structure_graph, source_node):
        """ gets the bucketed neighbors from the given source node and structure graph"""
        if self.clipping_threshold < 0:
            raise ValueError("Clipping threshold must be >= 0")

        sspl = _inv_dict(nx.single_source_shortest_path_length(structure_graph, source_node))

        if self.clipping_threshold is not None:
            num_buckets = 1 + self.clipping_threshold
            sspl = _combine_d(sspl, self.clipping_threshold, num_buckets - 1)

        return sspl

    def _compute_bucket_mtx(self, structure_graph):
        """ get the bucket_mtx for the given structure_graph
            calls _get_bucketed_neighbors for every node in the structure_graph """
        num_residues = len(list(structure_graph))

        # index into the embedding lookup table to create the final distance matrix
        bucket_mtx = torch.zeros(num_residues, num_residues, dtype=torch.long)

        for node_num in sorted(list(structure_graph)):
            bucketed_neighbors = self._compute_bucketed_neighbors(structure_graph, node_num)

            for bucket_num, neighbors in bucketed_neighbors.items():
                bucket_mtx[node_num, neighbors] = bucket_num

        return bucket_mtx


class RelativePosition(nn.Module):
    """ creates the embedding lookup table E_r and computes R """

    def __init__(self, embedding_len: int, clipping_threshold: int):
        """
        embedding_len: the length of the embedding, may be d_model, or d_model // num_heads for multihead
        clipping_threshold: the maximum relative position, referred to as k by Shaw et al.
        """
        super().__init__()
        self.embedding_len = embedding_len
        self.clipping_threshold = clipping_threshold
        # for sequence-based distances, the number of embeddings is 2*k+1, where k is the clipping threshold
        num_embeddings = 2 * clipping_threshold + 1

        # this is the embedding lookup table E_r
        self.embeddings_table = nn.Embedding(num_embeddings, embedding_len)

        # for getting the correct device for range vectors in forward
        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)

    def forward(self, length_q, length_k):
        # supports different length sequences, but in self-attention length_q and length_k are the same
        range_vec_q = torch.arange(length_q, device=self.dummy_buffer.device)
        range_vec_k = torch.arange(length_k, device=self.dummy_buffer.device)

        # this sets up the standard sequence-based distance matrix for relative positions
        # the current position is 0, positions to the right are +1, +2, etc, and to the left -1, -2, etc
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_threshold, self.clipping_threshold)

        # convert to indices, indexing into the embedding table
        final_mat = (distance_mat_clipped + self.clipping_threshold).long()

        # compute matrix R by grabbing the embeddings from the embedding lookup table
        embeddings = self.embeddings_table(final_mat)

        return embeddings


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, pos_encoding, clipping_threshold, contact_threshold, pdb_fns):
        """
        Multi-head attention with relative position embeddings.  Input data should be in batch_first format.
        :param embed_dim: aka d_model, aka hid_dim
        :param num_heads: number of heads
        :param dropout: how much dropout for scaled dot product attention

        :param pos_encoding: what type of positional encoding to use, relative or relative3D
        :param clipping_threshold: clipping threshold for relative position embedding
        :param contact_threshold: for relative_3D, the threshold in angstroms for the contact map
        :param pdb_fns: pdb file(s) to set up the relative position object

        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # model dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # pos encoding stuff
        self.pos_encoding = pos_encoding
        self.clipping_threshold = clipping_threshold
        self.contact_threshold = contact_threshold
        if pdb_fns is not None and not isinstance(pdb_fns, list):
            pdb_fns = [pdb_fns]
        self.pdb_fns = pdb_fns

        # relative position embeddings for use with keys and values
        # Shaw et al. uses relative position information for both keys and values
        # Huang et al. only uses it for the keys, which is probably enough
        if pos_encoding == "relative":
            self.relative_position_k = RelativePosition(self.head_dim, self.clipping_threshold)
            self.relative_position_v = RelativePosition(self.head_dim, self.clipping_threshold)
        elif pos_encoding == "relative_3D":
            self.relative_position_k = RelativePosition3D(self.head_dim, self.contact_threshold,
                                                          self.clipping_threshold, self.pdb_fns)
            self.relative_position_v = RelativePosition3D(self.head_dim, self.contact_threshold,
                                                          self.clipping_threshold, self.pdb_fns)
        else:
            raise ValueError("unrecognized pos_encoding: {}".format(pos_encoding))

        # WQ, WK, and WV from attention is all you need
        # note these default to bias=True, same as PyTorch implementation
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # WO from attention is all you need
        # used for the final projection when computing multi-head attention
        # PyTorch uses NonDynamicallyQuantizableLinear instead of Linear to avoid triggering an obscure
        # error quantizing the model https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L122
        # if quantizing the model, explore if the above is a concern for us
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # dropout for scaled dot product attention
        self.dropout = nn.Dropout(dropout)

        # scaling factor for scaled dot product attention
        scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        # persistent=False if you don't want to save it inside state_dict
        self.register_buffer('scale', scale)

        # toggles meant to be set directly by user
        self.need_weights = False
        self.average_attn_weights = True

    def _compute_attn_weights(self, query, key, len_q, len_k, batch_size, mask, pdb_fn):
        """ computes the attention weights (a "compatability function" of queries with corresponding keys) """

        # calculate the first term in the numerator attn1, which is Q*K
        # r_q1 = [batch_size, num_heads, len_q, head_dim]
        r_q1 = query.view(batch_size, len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # we could directly permute r_k1 to [batch_size, num_heads, head_dim, len_k]
        # to make it compatible for matrix multiplication with r_q1, instead of 2-step approach
        # r_k1 = [batch_size, num_heads, len_k, head_dim]
        r_k1 = key.view(batch_size, len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # attn1 = [batch_size, num_heads, len_q, len_k]
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        # calculate the second term in the numerator attn2, which is Q*R
        # r_q2 = [query_len, batch_size * num_heads, head_dim]
        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.num_heads, self.head_dim)

        if self.pos_encoding == "relative":
            # rel_pos_k = [len_q, len_k, head_dim]
            rel_pos_k = self.relative_position_k(len_q, len_k)
        elif self.pos_encoding == "relative_3D":
            # rel_pos_k = [sequence length (from PDB structure), head_dim]
            rel_pos_k = self.relative_position_k(pdb_fn)
        else:
            raise ValueError("unrecognized pos_encoding: {}".format(self.pos_encoding))

        # the matmul basically computes the dot product between each input position’s query vector and
        # its corresponding relative position embeddings across all input sequences in the heads and batch
        # attn2 = [batch_size * num_heads, len_q, len_k]
        attn2 = torch.matmul(r_q2, rel_pos_k.transpose(1, 2)).transpose(0, 1)
        # attn2 = [batch_size, num_heads, len_q, len_k]
        attn2 = attn2.contiguous().view(batch_size, self.num_heads, len_q, len_k)

        # calculate attention weights
        attn_weights = (attn1 + attn2) / self.scale

        # apply mask if given
        if mask is not None:
            # pytorch uses float("-inf") instead of -1e10
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)

        # softmax gives us attn_weights weights
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # attn_weights = [batch_size, num_heads, len_q, len_k]
        attn_weights = self.dropout(attn_weights)

        return attn_weights

    def _compute_avg_val(self, value, len_q, len_k, len_v, attn_weights, batch_size, pdb_fn):
        # calculate the first term, the attn*values
        # r_v1 = [batch_size, num_heads, len_v, head_dim]
        r_v1 = value.view(batch_size, len_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # avg1 = [batch_size, num_heads, len_q, head_dim]
        avg1 = torch.matmul(attn_weights, r_v1)

        # calculate the second term, the attn*R
        # similar to how relative embeddings are factored in the attention weights calculation
        if self.pos_encoding == "relative":
            # rel_pos_v = [query_len, value_len, head_dim]
            rel_pos_v = self.relative_position_v(len_q, len_v)
        elif self.pos_encoding == "relative_3D":
            # rel_pos_v = [sequence length (from PDB structure), head_dim]
            rel_pos_v = self.relative_position_v(pdb_fn)
        else:
            raise ValueError("unrecognized pos_encoding: {}".format(self.pos_encoding))

        # r_attn_weights = [len_q, batch_size * num_heads, len_v]
        r_attn_weights = attn_weights.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.num_heads, len_k)
        avg2 = torch.matmul(r_attn_weights, rel_pos_v)
        # avg2 = [batch_size, num_heads, len_q, head_dim]
        avg2 = avg2.transpose(0, 1).contiguous().view(batch_size, self.num_heads, len_q, self.head_dim)

        # calculate avg value
        x = avg1 + avg2  # [batch_size, num_heads, len_q, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, len_q, num_heads, head_dim]
        # x = [batch_size, len_q, embed_dim]
        x = x.view(batch_size, len_q, self.embed_dim)

        return x

    def forward(self, query, key, value, pdb_fn=None, mask=None):
        # query = [batch_size, q_len, embed_dim]
        # key = [batch_size, k_len, embed_dim]
        # value = [batch_size, v_en, embed_dim]
        batch_size = query.shape[0]
        len_k, len_q, len_v = (key.shape[1], query.shape[1], value.shape[1])

        # in projection (multiply inputs by WQ, WK, WV)
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # first compute the attention weights, then multiply with values
        # attn = [batch size, num_heads, len_q, len_k]
        attn_weights = self._compute_attn_weights(query, key, len_q, len_k, batch_size, mask, pdb_fn)

        # take weighted average of values (weighted by attention weights)
        attn_output = self._compute_avg_val(value, len_q, len_k, len_v, attn_weights, batch_size, pdb_fn)

        # output projection
        # attn_output = [batch_size, len_q, embed_dim]
        attn_output = self.out_proj(attn_output)

        if self.need_weights:
            # return attention weights in addition to attention
            # average the weights over the heads (to get overall attention)
            # attn_weights = [batch_size, len_q, len_k]
            if self.average_attn_weights:
                attn_weights = attn_weights.sum(dim=1) / self.num_heads
            return {"attn_output": attn_output, "attn_weights": attn_weights}
        else:
            return attn_output


class RelativeTransformerEncoderLayer(nn.Module):
    """
    d_model: the number of expected features in the input (required).
    nhead: the number of heads in the MultiHeadAttention models (required).
    clipping_threshold: the clipping threshold for relative position embeddings
    dim_feedforward: the dimension of the feedforward network model (default=2048).
    dropout: the dropout value (default=0.1).
    activation: the activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    norm_first: if ``True``, layer norm is done prior to attention and feedforward
        operations, respectively. Otherwise, it's done after. Default: ``False`` (after).
    """

    # this is some kind of torch jit compiling helper... will also ensure these values don't change
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model,
                 nhead,
                 pos_encoding="relative",
                 clipping_threshold=3,
                 contact_threshold=7,
                 pdb_fns=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False) -> None:

        self.batch_first = True

        super(RelativeTransformerEncoderLayer, self).__init__()

        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout,
                                                    pos_encoding, clipping_threshold, contact_threshold, pdb_fns)

        # feed forward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = models.get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, src: Tensor, pdb_fn=None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), pdb_fn=pdb_fn)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, pdb_fn=None) -> Tensor:
        x = self.self_attn(x, x, x, pdb_fn=pdb_fn)
        if isinstance(x, dict):
            # handle the case where we are returning attention weights
            x = x["attn_output"]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class RelativeTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, reset_params=True):
        super(RelativeTransformerEncoder, self).__init__()
        # using get_clones means all layers have the same initialization
        # this is also a problem in PyTorch's TransformerEncoder implementation, which this is based on
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        # important because get_clones means all layers have same initialization
        # should recursively reset parameters for all submodules
        if reset_params:
            self.apply(models.reset_parameters_helper)

    def forward(self, src: Tensor, pdb_fn=None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, pdb_fn=pdb_fn)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, num_clones):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])


def _inv_dict(d):
    """ helper function for contact map-based position embeddings """
    inv = dict()
    for k, v in d.items():
        # collect dict keys into lists based on value
        inv.setdefault(v, list()).append(k)
    for k, v in inv.items():
        # put in sorted order
        inv[k] = sorted(v)
    return inv


def _combine_d(d, threshold, combined_key):
    """ helper function for contact map-based position embeddings
        d is a dictionary with ints as keys and lists as values.
        for all keys >= threshold, this function combines the values of those keys into a single list """
    out_d = {}
    for k, v in d.items():
        if k < threshold:
            out_d[k] = v
        elif k >= threshold:
            if combined_key not in out_d:
                out_d[combined_key] = v
            else:
                out_d[combined_key] += v
    if combined_key in out_d:
        out_d[combined_key] = sorted(out_d[combined_key])
    return out_d
