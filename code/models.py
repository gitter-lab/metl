""" model implementations """

import collections
import inspect
import math
from argparse import ArgumentParser
import enum
from os.path import isfile
from typing import List, Tuple, Callable, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from . import relative_attention as ra
    from . import tasks
except ImportError:
    import relative_attention as ra
    import tasks

def reset_parameters_helper(m: nn.Module):
    """ helper function for resetting model parameters, meant to be used with model.apply() """

    # the PyTorch MultiHeadAttention has a private function _reset_parameters()
    # other layers have a public reset_parameters()... go figure
    reset_parameters = getattr(m, "reset_parameters", None)
    reset_parameters_private = getattr(m, "_reset_parameters", None)

    if callable(reset_parameters) and callable(reset_parameters_private):
        raise RuntimeError("Module has both public and private methods for resetting parameters. "
                           "This is unexpected... probably should just call the public one.")

    if callable(reset_parameters):
        m.reset_parameters()

    if callable(reset_parameters_private):
        m._reset_parameters()


class SequentialWithArgs(nn.Sequential):
    """ a sequential model wrapper  that allows for passing in additional arguments
        to the forward function. this is useful for passing in the pdb_fn to the
        relative transformer encoder as well as other auxiliary inputs """

    def __init__(self, *args):
        super().__init__(*args)
        # cache which modules accept kwargs
        self._kwargs_support_cache = {
            module: self._accepts_kwargs(module.forward)
            for module in self
        }

    def forward(self, x, **kwargs):
        for module in self:
            if self._kwargs_support_cache.get(module, False):
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x

    @staticmethod
    def _accepts_kwargs(func):
        # new way of checking if the module accepts kwargs
        # by inspecting the signatures instead of maintaining a list manually
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return True
        return False


class PositionalEncoding(nn.Module):
    # originally from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # they have since updated their implementation, but it is functionally equivalent
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # note the implementation on Pytorch's website expects [seq_len, batch_size, embedding_dim]
        # however our data is in [batch_size, seq_len, embedding_dim] (i.e. batch_first)
        # fixed by changing pe = pe.unsqueeze(0).transpose(0, 1) to pe = pe.unsqueeze(0)
        # also down below, changing our indexing into the position encoding to reflect new dimensions
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        # note the implementation on Pytorch's website expects [seq_len, batch_size, embedding_dim]
        # however our data is in [batch_size, seq_len, embedding_dim] (i.e. batch_first)
        # fixed by changing x = x + self.pe[:x.size(0)] to x = x + self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledEmbedding(nn.Module):
    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    # a helper function for embedding that scales by sqrt(d_model) in the forward()
    # makes it, so we don't have to do the scaling in the main AttnModel forward()

    # be aware of embedding scaling factor
    # regarding the scaling factor, it's unclear exactly what the purpose is and whether it is needed
    # there are several theories on why it is used, and it shows up in all the transformer reference implementations
    # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod
    #   1. Has something to do with weight sharing between the embedding and the decoder output
    #   2. Scales up the embeddings so the signal doesn't get overwhelmed when adding the absolute positional encoding
    #   3. It cancels out with the scaling factor in scaled dot product attention, and helps make the model robust
    #      to the choice of embedding_len
    #   4. It's not actually needed

    # Regarding #1, not really sure about this. In section 3.4 of attention is all you need,
    # that's where they state they multiply the embedding weights by sqrt(d_model), and the context is that they
    # are sharing the same weight matrix between the two embedding layers and the pre-softmax linear transformation.
    # there may be a reason that we want those weights scaled differently for the embedding layers vs. the linear
    # transformation. It might have something to do with the scale at which embedding weights are initialized
    # is more appropriate for the decoder linear transform vs how they are used in the attention function. Might have
    # something to do with computing the correct next-token probabilities. Overall, I'm really not sure about this,
    # but we aren't using a decoder anyway. So if this is the reason, then we don't need to perform the multiply.

    # Regarding #2, it seems like in one implementation of transformers (fairseq), the sinusoidal positional encoding
    # has a range of (-1.0, 1.0), but the word embedding are initialized with mean 0 and s.d embedding_dim ** -0.5,
    # which for embedding_dim=512, is a range closer to (-0.10, 0.10). Thus, the positional embedding would overwhelm
    # the word embeddings when they are added together. The scaling factor increases the signal of the word embeddings.
    # for embedding_dim=512, it scales word embeddings by 22, increasing range of the word embeddings to (-2.2, 2.2).
    # link to fairseq implementation, search for nn.init to see them do the initialization
    # https://fairseq.readthedocs.io/en/v0.7.1/_modules/fairseq/models/transformer.html
    #
    # For PyTorch, PyTorch initializes nn.Embedding with a standard normal distribution mean 0, variance 1: N(0,1).
    # this puts the range for the word embeddings around (-3, 3). the pytorch implementation for positional encoding
    # also has a range of (-1.0, 1.0). So already, these are much closer in scale, and it doesn't seem like we need
    # to increase the scale of the word embeddings. However, PyTorch example still multiply by the scaling factor
    # unclear whether this is just a carryover that is not actually needed, or if there is a different reason
    #
    # EDIT! I just realized that even though nn.Embedding defaults to a range of around (-3, 3), the PyTorch
    # transformer example actually re-initializes them using a uniform distribution in the range of (-0.1, 0.1)
    # that makes it very similar to the fairseq implementation, so the scaling factor that PyTorch uses actually would
    # bring the word embedding and positional encodings much closer in scale. So this could be the reason why pytorch
    # does it

    # Regarding #3, I don't think so. Firstly, does it actually cancel there? Secondly, the purpose of the scaling
    # factor in scaled dot product attention, according to attention is all you need, is to counteract dot products
    # that are very high in magnitude due to choice of large mbedding length (aka d_k). The problem with high magnitude
    # dot products is that potentially, the softmax is pushed into regions where it has extremely small gradients,
    # making learning difficult. If the scaling factor in the embedding was meant to counteract the scaling factor in
    # scaled dot product attention, then what would be the point of doing all that?

    # Regarding #4, I don't think the scaling will have any effects in practice, it's probably not needed

    # Overall, I think #2 is the most likely reason why this scaling is performed. In theory, I think
    # even if the scaling wasn't performed, the network might learn to up-scale the word embedding weights to increase
    # word embedding signal vs. the position signal on its own. Another question I have is why not just initialize
    # the embedding weights to have higher initial values? Why put it in the range (-0.1, 0.1)?

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: bool):
        super(ScaledEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.emb_size = embedding_dim
        self.embed_scale = math.sqrt(self.emb_size)

        self.scale = scale

        self.init_weights()

    def init_weights(self):
        # not sure why PyTorch example initializes weights like this
        # might have something to do with word embedding scaling factor (see above)
        # could also just try the default weight initialization for nn.Embedding()
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, tokens: Tensor, **kwargs):
        if self.scale:
            return self.embedding(tokens.long()) * self.embed_scale
        else:
            return self.embedding(tokens.long())


class FCBlock(nn.Module):
    """ a fully connected block with options for batchnorm and dropout
        can extend in the future with option for different activation, etc """

    def __init__(self,
                 in_features: int,
                 num_hidden_nodes: int = 64,
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 norm_before_activation: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        super().__init__()

        if use_batchnorm and use_layernorm:
            raise ValueError("Only one of use_batchnorm or use_layernorm can be set to True")

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_layernorm = use_layernorm
        self.norm_before_activation = norm_before_activation

        self.fc = nn.Linear(in_features=in_features, out_features=num_hidden_nodes)

        self.activation = get_activation_fn(activation, functional=False)

        if use_batchnorm:
            self.norm = nn.BatchNorm1d(num_hidden_nodes)

        if use_layernorm:
            self.norm = nn.LayerNorm(num_hidden_nodes)

        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, **kwargs):
        x = self.fc(x)

        # norm can be before or after activation, using flag
        if (self.use_batchnorm or self.use_layernorm) and self.norm_before_activation:
            x = self.norm(x)

        x = self.activation(x)

        # batchnorm being applied after activation, there is some discussion on this online
        if (self.use_batchnorm or self.use_layernorm) and not self.norm_before_activation:
            x = self.norm(x)

        # dropout being applied last
        if self.use_dropout:
            x = self.dropout(x)

        return x


class TaskSpecificPredictionLayers(nn.Module):
    """ Constructs num_tasks [dense(num_hidden_nodes)+relu+dense(1)] layers, each independently transforming input
        into a single output node. All num_tasks outputs are then concatenated into a single tensor. """

    # the independent layers are run in sequence rather than in parallel, causing a slowdown that
    # scales with the number of tasks. might be able to run in parallel by hacking convolution operation
    # https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch
    # https://github.com/pytorch/pytorch/issues/54147
    # https://github.com/pytorch/pytorch/issues/36459

    def __init__(self,
                 num_tasks: int,
                 in_features: int,
                 num_hidden_nodes: int = 64,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        super().__init__()

        # each task-specific layer outputs a single node,
        # which can be combined with torch.cat into prediction vector
        self.task_specific_pred_layers = nn.ModuleList()
        for i in range(num_tasks):
            layers = [FCBlock(in_features=in_features,
                              num_hidden_nodes=num_hidden_nodes,
                              use_batchnorm=use_batchnorm,
                              use_dropout=use_dropout,
                              dropout_rate=dropout_rate,
                              activation=activation),
                      nn.Linear(in_features=num_hidden_nodes, out_features=1)]
            self.task_specific_pred_layers.append(nn.Sequential(*layers))

    def forward(self, x, **kwargs):
        # run each task-specific layer and concatenate outputs into a single output vector
        task_specific_outputs = []
        for layer in self.task_specific_pred_layers:
            task_specific_outputs.append(layer(x))

        output = torch.cat(task_specific_outputs, dim=1)
        return output


class GlobalAveragePooling(nn.Module):
    """ helper class for global average pooling """

    def __init__(self, dim=1):
        super().__init__()
        # our data is in [batch_size, sequence_length, embedding_length]
        # with global pooling, we want to pool over the sequence dimension (dim=1)
        self.dim = dim

    def forward(self, x, **kwargs):
        return torch.mean(x, dim=self.dim)


class CLSPooling(nn.Module):
    """ helper class for CLS token extraction """

    def __init__(self, cls_position=0):
        super().__init__()

        # the position of the CLS token in the sequence dimension
        # currently, the CLS token is in the first position, but may move it to the last position
        self.cls_position = cls_position

    def forward(self, x, **kwargs):
        # assumes input is in [batch_size, sequence_len, embedding_len]
        # thus sequence dimension is dimension 1
        return x[:, self.cls_position, :]


class TransformerEncoderWrapper(nn.TransformerEncoder):
    """ wrapper around PyTorch's TransformerEncoder that re-initializes layer parameters,
        so each transformer encoder layer has a different initialization """

    # PyTorch is changing its transformer API... check up on and see if there is a better way
    def __init__(self, encoder_layer, num_layers, norm=None, reset_params=True):
        super().__init__(encoder_layer, num_layers, norm)
        if reset_params:
            self.apply(reset_parameters_helper)


class AttnModel(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--pos_encoding', type=str, default="absolute",
                            choices=["none", "absolute", "relative", "relative_3D"],
                            help="what type of positional encoding to use")
        parser.add_argument('--pos_encoding_dropout', type=float, default=0.1,
                            help="out much dropout to use in positional encoding, for pos_encoding==absolute")
        parser.add_argument('--clipping_threshold', type=int, default=3,
                            help="clipping threshold for relative position embedding, for relative and relative_3D")
        parser.add_argument('--contact_threshold', type=int, default=7,
                            help="threshold, in angstroms, for contact map, for relative_3D")
        parser.add_argument('--embedding_len', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--num_hidden', type=int, default=64)
        parser.add_argument('--num_enc_layers', type=int, default=2)
        parser.add_argument('--enc_layer_dropout', type=float, default=0.1)
        parser.add_argument('--use_final_encoder_norm', action="store_true", default=False)

        parser.add_argument('--global_average_pooling', action="store_true", default=False)
        parser.add_argument('--cls_pooling', action="store_true", default=False)

        parser.add_argument('--use_task_specific_layers', action="store_true", default=False,
                            help="exclusive with use_final_hidden_layer; takes priority over use_final_hidden_layer"
                                 " if both flags are set")
        parser.add_argument('--task_specific_hidden_nodes', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer', action="store_true", default=False)
        parser.add_argument('--final_hidden_size', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer_norm', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_final_hidden_layer_dropout', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_dropout_rate', type=float, default=0.2)

        parser.add_argument('--activation', type=str, default="relu",
                            help="activation function used for all activations in the network")
        return parser

    def __init__(self,
                 # data args
                 num_tasks: int,
                 aa_seq_len: int,
                 num_tokens: int,
                 # transformer encoder model args
                 pos_encoding: str = "absolute",
                 pos_encoding_dropout: float = 0.1,
                 clipping_threshold: int = 3,
                 contact_threshold: int = 7,
                 pdb_fns: List[str] = None,
                 embedding_len: int = 64,
                 num_heads: int = 2,
                 num_hidden: int = 64,
                 num_enc_layers: int = 2,
                 enc_layer_dropout: float = 0.1,
                 use_final_encoder_norm: bool = False,
                 # pooling to fixed-length representation
                 global_average_pooling: bool = True,
                 cls_pooling: bool = False,
                 # prediction layers
                 use_task_specific_layers: bool = False,
                 task_specific_hidden_nodes: int = 64,
                 use_final_hidden_layer: bool = False,
                 final_hidden_size: int = 64,
                 use_final_hidden_layer_norm: bool = False,
                 final_hidden_layer_norm_before_activation: bool = False,
                 use_final_hidden_layer_dropout: bool = False,
                 final_hidden_layer_dropout_rate: float = 0.2,
                 # activation function
                 activation: str = "relu",
                 *args, **kwargs):

        super().__init__()

        # store embedding length for use in the forward function
        self.embedding_len = embedding_len
        self.aa_seq_len = aa_seq_len

        # build up layers
        layers = collections.OrderedDict()

        # amino acid embedding
        layers["embedder"] = ScaledEmbedding(num_embeddings=num_tokens, embedding_dim=embedding_len, scale=True)

        # absolute positional encoding
        if pos_encoding == "absolute":
            layers["pos_encoder"] = PositionalEncoding(embedding_len, dropout=pos_encoding_dropout, max_len=512)

        # transformer encoder layer for none or absolute positional encoding
        if pos_encoding in ["none", "absolute"]:
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_len,
                                                             nhead=num_heads,
                                                             dim_feedforward=num_hidden,
                                                             dropout=enc_layer_dropout,
                                                             activation=get_activation_fn(activation),
                                                             norm_first=True,
                                                             batch_first=True)

            # layer norm that is used after the transformer encoder layers
            # if the norm_first is False, this is *redundant* and not needed
            # but if norm_first is True, this can be used to normalize outputs from
            # the transformer encoder before inputting to the final fully connected layer
            encoder_norm = None
            if use_final_encoder_norm:
                encoder_norm = nn.LayerNorm(embedding_len)

            layers["tr_encoder"] = TransformerEncoderWrapper(encoder_layer=encoder_layer,
                                                             num_layers=num_enc_layers,
                                                             norm=encoder_norm)

        # transformer encoder layer for relative position encoding
        elif pos_encoding in ["relative", "relative_3D"]:
            relative_encoder_layer = ra.RelativeTransformerEncoderLayer(d_model=embedding_len,
                                                                        nhead=num_heads,
                                                                        pos_encoding=pos_encoding,
                                                                        clipping_threshold=clipping_threshold,
                                                                        contact_threshold=contact_threshold,
                                                                        pdb_fns=pdb_fns,
                                                                        dim_feedforward=num_hidden,
                                                                        dropout=enc_layer_dropout,
                                                                        activation=get_activation_fn(activation),
                                                                        norm_first=True)

            encoder_norm = None
            if use_final_encoder_norm:
                encoder_norm = nn.LayerNorm(embedding_len)

            layers["tr_encoder"] = ra.RelativeTransformerEncoder(encoder_layer=relative_encoder_layer,
                                                                 num_layers=num_enc_layers,
                                                                 norm=encoder_norm)

        # global average pooling or CLS token
        # set up the layers and output shapes (i.e. input shapes for the pred layer)
        if global_average_pooling:
            # pool over the sequence dimension
            layers["avg_pooling"] = GlobalAveragePooling(dim=1)
            pred_layer_input_features = embedding_len
        elif cls_pooling:
            layers["cls_pooling"] = CLSPooling(cls_position=0)
            pred_layer_input_features = embedding_len
        else:
            # no global average pooling or CLS token
            # sequence dimension is still there, just flattened
            layers["flatten"] = nn.Flatten()
            pred_layer_input_features = embedding_len * aa_seq_len

        # prediction layers
        if use_task_specific_layers:
            # task specific prediction layers (nonlinear transform for each task)
            layers["prediction"] = TaskSpecificPredictionLayers(num_tasks=num_tasks,
                                                                in_features=pred_layer_input_features,
                                                                num_hidden_nodes=task_specific_hidden_nodes,
                                                                activation=activation)
        elif use_final_hidden_layer:
            # combined prediction linear (linear transform for each task)
            layers["fc1"] = FCBlock(in_features=pred_layer_input_features,
                                    num_hidden_nodes=final_hidden_size,
                                    use_batchnorm=False,
                                    use_layernorm=use_final_hidden_layer_norm,
                                    norm_before_activation=final_hidden_layer_norm_before_activation,
                                    use_dropout=use_final_hidden_layer_dropout,
                                    dropout_rate=final_hidden_layer_dropout_rate,
                                    activation=activation)

            layers["prediction"] = nn.Linear(in_features=final_hidden_size, out_features=num_tasks)
        else:
            layers["prediction"] = nn.Linear(in_features=pred_layer_input_features, out_features=num_tasks)

        # final model
        self.model = SequentialWithArgs(layers)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


class Transpose(nn.Module):
    """ helper layer to swap data from (batch, seq, channels) to (batch, channels, seq)
        used as a helper in the convolutional network which pytorch defaults to channels-first """

    def __init__(self, dims: Tuple[int, ...] = (1, 2)):
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        x = x.transpose(*self.dims).contiguous()
        return x


def conv1d_out_shape(seq_len, kernel_size, stride=1, pad=0, dilation=1):
    return (seq_len + (2 * pad) - (dilation * (kernel_size - 1)) - 1 // stride) + 1


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 padding: str = "same",
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 norm_before_activation: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        super().__init__()

        if use_batchnorm and use_layernorm:
            raise ValueError("Only one of use_batchnorm or use_layernorm can be set to True")

        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.norm_before_activation = norm_before_activation
        self.use_dropout = use_dropout

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation)

        self.activation = get_activation_fn(activation, functional=False)

        if use_batchnorm:
            self.norm = nn.BatchNorm1d(out_channels)

        if use_layernorm:
            self.norm = nn.LayerNorm(out_channels)

        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, **kwargs):
        x = self.conv(x)

        # norm can be before or after activation, using flag
        if self.use_batchnorm and self.norm_before_activation:
            x = self.norm(x)
        elif self.use_layernorm and self.norm_before_activation:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        x = self.activation(x)

        # batchnorm being applied after activation, there is some discussion on this online
        if self.use_batchnorm and not self.norm_before_activation:
            x = self.norm(x)
        elif self.use_layernorm and not self.norm_before_activation:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # dropout being applied after batchnorm, there is some discussion on this online
        if self.use_dropout:
            x = self.dropout(x)

        return x


class ConvModel2(nn.Module):
    """ convolutional source model that supports padded inputs, pooling, etc """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--use_embedding', action="store_true", default=False)
        parser.add_argument('--embedding_len', type=int, default=128)

        parser.add_argument('--num_conv_layers', type=int, default=1)
        parser.add_argument('--kernel_sizes', type=int, nargs="+", default=[7])
        parser.add_argument('--out_channels', type=int, nargs="+", default=[128])
        parser.add_argument('--dilations', type=int, nargs="+", default=[1])
        parser.add_argument('--padding', type=str, default="valid", choices=["valid", "same"])
        parser.add_argument('--use_conv_layer_norm', action="store_true", default=False)
        parser.add_argument('--conv_layer_norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_conv_layer_dropout', action="store_true", default=False)
        parser.add_argument('--conv_layer_dropout_rate', type=float, default=0.2)

        parser.add_argument('--global_average_pooling', action="store_true", default=False)

        parser.add_argument('--use_task_specific_layers', action="store_true", default=False)
        parser.add_argument('--task_specific_hidden_nodes', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer', action="store_true", default=False)
        parser.add_argument('--final_hidden_size', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer_norm', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_final_hidden_layer_dropout', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_dropout_rate', type=float, default=0.2)

        parser.add_argument('--activation', type=str, default="relu",
                            help="activation function used for all activations in the network")

        return parser

    def __init__(self,
                 # data
                 num_tasks: int,
                 aa_seq_len: int,
                 aa_encoding_len: int,
                 num_tokens: int,
                 # convolutional model args
                 use_embedding: bool = False,
                 embedding_len: int = 64,
                 num_conv_layers: int = 1,
                 kernel_sizes: List[int] = (7,),
                 out_channels: List[int] = (128,),
                 dilations: List[int] = (1,),
                 padding: str = "valid",
                 use_conv_layer_norm: bool = False,
                 conv_layer_norm_before_activation: bool = False,
                 use_conv_layer_dropout: bool = False,
                 conv_layer_dropout_rate: float = 0.2,
                 # pooling
                 global_average_pooling: bool = True,
                 # prediction layers
                 use_task_specific_layers: bool = False,
                 task_specific_hidden_nodes: int = 64,
                 use_final_hidden_layer: bool = False,
                 final_hidden_size: int = 64,
                 use_final_hidden_layer_norm: bool = False,
                 final_hidden_layer_norm_before_activation: bool = False,
                 use_final_hidden_layer_dropout: bool = False,
                 final_hidden_layer_dropout_rate: float = 0.2,
                 # activation function
                 activation: str = "relu",
                 *args, **kwargs):

        super(ConvModel2, self).__init__()

        # build up the layers
        layers = collections.OrderedDict()

        # amino acid embedding
        if use_embedding:
            layers["embedder"] = ScaledEmbedding(num_embeddings=num_tokens, embedding_dim=embedding_len, scale=False)

        # transpose the input to match PyTorch's expected format
        layers["transpose"] = Transpose(dims=(1, 2))

        # build up the convolutional layers
        for layer_num in range(num_conv_layers):
            # determine the number of input channels for the first convolutional layer
            if layer_num == 0 and use_embedding:
                # for the first convolutional layer, the in_channels is the embedding_len
                in_channels = embedding_len
            elif layer_num == 0 and not use_embedding:
                # for the first convolutional layer, the in_channels is the aa_encoding_len
                in_channels = aa_encoding_len
            else:
                in_channels = out_channels[layer_num - 1]

            layers[f"conv{layer_num}"] = ConvBlock(in_channels=in_channels,
                                                   out_channels=out_channels[layer_num],
                                                   kernel_size=kernel_sizes[layer_num],
                                                   dilation=dilations[layer_num],
                                                   padding=padding,
                                                   use_batchnorm=False,
                                                   use_layernorm=use_conv_layer_norm,
                                                   norm_before_activation=conv_layer_norm_before_activation,
                                                   use_dropout=use_conv_layer_dropout,
                                                   dropout_rate=conv_layer_dropout_rate,
                                                   activation=activation)

        # handle transition from convolutional layers to fully connected layer
        # either use global average pooling or flatten
        # take into consideration whether we are using valid or same padding
        if global_average_pooling:
            # global average pooling (mean across the seq len dimension)
            # the seq len dimensions is the last dimension (batch_size, num_filters, seq_len)
            layers["avg_pooling"] = GlobalAveragePooling(dim=-1)
            # the prediction layers will take num_filters input features
            pred_layer_input_features = out_channels[-1]

        else:
            # no global average pooling. flatten instead.
            layers["flatten"] = nn.Flatten()
            # calculate the final output len of the convolutional layers
            # and the number of input features for the prediction layers
            if padding == "valid":
                # valid padding (aka no padding) results in shrinking length in progressive layers
                conv_out_len = conv1d_out_shape(aa_seq_len, kernel_size=kernel_sizes[0], dilation=dilations[0])
                for layer_num in range(1, num_conv_layers):
                    conv_out_len = conv1d_out_shape(conv_out_len,
                                                    kernel_size=kernel_sizes[layer_num],
                                                    dilation=dilations[layer_num])
                pred_layer_input_features = conv_out_len * out_channels[-1]
            else:
                # padding == "same"
                pred_layer_input_features = aa_seq_len * out_channels[-1]

        # prediction layer
        if use_task_specific_layers:
            layers["prediction"] = TaskSpecificPredictionLayers(num_tasks=num_tasks,
                                                                in_features=pred_layer_input_features,
                                                                num_hidden_nodes=task_specific_hidden_nodes,
                                                                activation=activation)

        # final hidden layer (with potential additional dropout)
        elif use_final_hidden_layer:
            layers["fc1"] = FCBlock(in_features=pred_layer_input_features,
                                    num_hidden_nodes=final_hidden_size,
                                    use_batchnorm=False,
                                    use_layernorm=use_final_hidden_layer_norm,
                                    norm_before_activation=final_hidden_layer_norm_before_activation,
                                    use_dropout=use_final_hidden_layer_dropout,
                                    dropout_rate=final_hidden_layer_dropout_rate,
                                    activation=activation)
            layers["prediction"] = nn.Linear(in_features=final_hidden_size, out_features=num_tasks)

        else:
            layers["prediction"] = nn.Linear(in_features=pred_layer_input_features, out_features=num_tasks)

        self.model = nn.Sequential(layers)

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class ConvModel(nn.Module):
    """ a convolutional network with convolutional layers followed by a fully connected layer """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_conv_layers', type=int, default=1)
        parser.add_argument('--kernel_sizes', type=int, nargs="+", default=[7])
        parser.add_argument('--out_channels', type=int, nargs="+", default=[128])
        parser.add_argument('--padding', type=str, default="valid", choices=["valid", "same"])
        parser.add_argument('--use_final_hidden_layer', action="store_true",
                            help="whether to use a final hidden layer")
        parser.add_argument('--final_hidden_size', type=int, default=128,
                            help="number of nodes in the final hidden layer")
        parser.add_argument('--use_dropout', action="store_true",
                            help="whether to use dropout in the final hidden layer")
        parser.add_argument('--dropout_rate', type=float, default=0.2,
                            help="dropout rate in the final hidden layer")
        parser.add_argument('--use_task_specific_layers', action="store_true", default=False)
        parser.add_argument('--task_specific_hidden_nodes', type=int, default=64)
        return parser

    def __init__(self,
                 num_tasks: int,
                 aa_seq_len: int,
                 aa_encoding_len: int,
                 num_conv_layers: int = 1,
                 kernel_sizes: List[int] = (7,),
                 out_channels: List[int] = (128,),
                 padding: str = "valid",
                 use_final_hidden_layer: bool = True,
                 final_hidden_size: int = 128,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 use_task_specific_layers: bool = False,
                 task_specific_hidden_nodes: int = 64,
                 *args, **kwargs):

        super(ConvModel, self).__init__()

        # set up the model as a Sequential block (less to do in forward())
        layers = collections.OrderedDict()

        layers["transpose"] = Transpose(dims=(1, 2))

        for layer_num in range(num_conv_layers):
            # for the first convolutional layer, the in_channels is the feature_len
            in_channels = aa_encoding_len if layer_num == 0 else out_channels[layer_num - 1]

            layers["conv{}".format(layer_num)] = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels[layer_num],
                          kernel_size=kernel_sizes[layer_num],
                          padding=padding),
                nn.ReLU()
            )

        layers["flatten"] = nn.Flatten()

        # calculate the final output len of the convolutional layers
        # and the number of input features for the prediction layers
        if padding == "valid":
            # valid padding (aka no padding) results in shrinking length in progressive layers
            conv_out_len = conv1d_out_shape(aa_seq_len, kernel_size=kernel_sizes[0])
            for layer_num in range(1, num_conv_layers):
                conv_out_len = conv1d_out_shape(conv_out_len, kernel_size=kernel_sizes[layer_num])
            next_dim = conv_out_len * out_channels[-1]
        elif padding == "same":
            next_dim = aa_seq_len * out_channels[-1]
        else:
            raise ValueError("unexpected value for padding: {}".format(padding))

        # final hidden layer (with potential additional dropout)
        if use_final_hidden_layer:
            layers["fc1"] = FCBlock(in_features=next_dim,
                                    num_hidden_nodes=final_hidden_size,
                                    use_batchnorm=False,
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate)
            next_dim = final_hidden_size

        # final prediction layer
        # either task specific nonlinear layers or a single linear layer
        if use_task_specific_layers:
            layers["prediction"] = TaskSpecificPredictionLayers(num_tasks=num_tasks,
                                                                in_features=next_dim,
                                                                num_hidden_nodes=task_specific_hidden_nodes)
        else:
            layers["prediction"] = nn.Linear(in_features=next_dim, out_features=num_tasks)

        self.model = nn.Sequential(layers)

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class FCModel(nn.Module):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_hidden', nargs="+", type=int, default=[128])
        parser.add_argument('--use_batchnorm', action="store_true", default=False)
        parser.add_argument('--use_layernorm', action="store_true", default=False)
        parser.add_argument('--norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_dropout', action="store_true", default=False)
        parser.add_argument('--dropout_rate', type=float, default=0.2)
        parser.add_argument('--activation', type=str, default="relu")
        return parser

    def __init__(self,
                 num_tasks: int,
                 seq_encoding_len: int,
                 num_layers: int = 1,
                 num_hidden: List[int] = (128,),
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 norm_before_activation: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu",
                 *args, **kwargs):
        super().__init__()

        # set up the model as a Sequential block (less to do in forward())
        layers = collections.OrderedDict()

        # flatten inputs as this is all fully connected
        layers["flatten"] = nn.Flatten()

        # build up the variable number of hidden layers (fully connected + ReLU + dropout (if set))
        for layer_num in range(num_layers):
            # for the first layer (layer_num == 0), in_features is determined by given input
            # for subsequent layers, the in_features is the previous layer's num_hidden
            in_features = seq_encoding_len if layer_num == 0 else num_hidden[layer_num - 1]

            layers["fc{}".format(layer_num)] = FCBlock(in_features=in_features,
                                                       num_hidden_nodes=num_hidden[layer_num],
                                                       use_batchnorm=use_batchnorm,
                                                       use_layernorm=use_layernorm,
                                                       norm_before_activation=norm_before_activation,
                                                       use_dropout=use_dropout,
                                                       dropout_rate=dropout_rate,
                                                       activation=activation)

        # finally, the linear output layer
        in_features = num_hidden[-1] if num_layers > 0 else seq_encoding_len
        layers["output"] = nn.Linear(in_features=in_features, out_features=num_tasks)

        self.model = nn.Sequential(layers)

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class LRModel(nn.Module):
    """ a simple linear model """

    def __init__(self, num_tasks, seq_encoding_len, *args, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_encoding_len, out_features=num_tasks))

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


def modify_dropout_rate(module, rate):
    if isinstance(module, torch.nn.Dropout):
        module.p = rate


class TransferModel(nn.Module):
    """ transfer learning model """

    @staticmethod
    def add_model_specific_args(parent_parser):

        def none_or_int(value: str):
            return None if value.lower() == "none" else int(value)

        p = ArgumentParser(parents=[parent_parser], add_help=False)

        # for model set up
        p.add_argument('--pretrained_ckpt_path', type=str, default=None)

        # where to cut off the backbone
        p.add_argument("--backbone_cutoff", type=none_or_int, default=-1,
                       help="where to cut off the backbone. can be a negative int, indexing back from "
                            "pretrained_model.model.model. a value of -1 would chop off the backbone prediction head. "
                            "a value of -2 chops the prediction head and FC layer. a value of -3 chops"
                            "the above, as well as the global average pooling layer. all depends on architecture.")

        p.add_argument("--pred_layer_input_features", type=int, default=None,
                       help="if None, number of features will be determined based on backbone_cutoff and standard "
                            "architecture. otherwise, specify the number of input features for the prediction layer")

        p.add_argument("--new_backbone_dropout_rate", type=float, default=None)

        # top net args
        p.add_argument("--dropout_after_backbone", action="store_true")
        p.add_argument("--dropout_after_backbone_rate", type=float, default=0.1)
        p.add_argument("--top_net_type", type=str, default="linear", choices=["linear", "nonlinear", "sklearn"])
        p.add_argument("--top_net_hidden_nodes", type=int, default=256)
        p.add_argument("--top_net_use_batchnorm", action="store_true")
        p.add_argument("--top_net_use_layernorm", action="store_true")
        p.add_argument("--top_net_norm_before_activation", action="store_true")
        p.add_argument("--top_net_use_dropout", action="store_true")
        p.add_argument("--top_net_dropout_rate", type=float, default=0.1)

        return p

    def __init__(self,
                 # pretrained model
                 pretrained_ckpt_path: Optional[str] = None,
                 pretrained_hparams: Optional[dict] = None,
                 backbone_cutoff: Optional[int] = -1,
                 new_backbone_dropout_rate: Optional[float] = None,
                 # top net
                 dropout_after_backbone: bool = False,
                 dropout_after_backbone_rate: float = 0.1,
                 pred_layer_input_features: Optional[int] = None,
                 top_net_type: str = "linear",
                 top_net_hidden_nodes: int = 256,
                 top_net_use_batchnorm: bool = False,
                 top_net_use_layernorm: bool = False,
                 top_net_norm_before_activation: bool = False,
                 top_net_use_dropout: bool = False,
                 top_net_dropout_rate: float = 0.1,
                 *args, **kwargs):

        super().__init__()

        # error checking: if pretrained_ckpt_path is None, then pretrained_hparams must be specified
        if pretrained_ckpt_path is None and pretrained_hparams is None:
            raise ValueError("Either pretrained_ckpt_path or pretrained_hparams must be specified")

        # note: pdb_fns is loaded from transfer model arguments rather than original source model hparams
        # if pdb_fns is specified as a kwarg, pass it on for structure-based RPE
        # otherwise, can just set pdb_fns to None, and structure-based RPE will handle new PDBs on the fly
        pdb_fns = kwargs["pdb_fns"] if "pdb_fns" in kwargs else None

        # generate a fresh backbone using pretrained_hparams if specified
        # otherwise load the backbone from the pretrained checkpoint
        # we prioritize pretrained_hparams over pretrained_ckpt_path because
        # pretrained_hparams will only really be specified if we are loading from a DMSTask checkpoint
        # meaning the TransferModel has already been fine-tuned on DMS data, and we are likely loading
        # weights from that finetuning (including weights for the backbone)
        # whereas if pretrained_hparams is not specified but pretrained_ckpt_path is, then we are
        # likely finetuning the TransferModel for the first time, and we need the pretrained weights for the
        # backbone from the RosettaTask checkpoint
        if pretrained_hparams is not None:
            # pretrained_hparams will only be specified if we are loading from a DMSTask checkpoint
            pretrained_hparams["pdb_fns"] = pdb_fns
            pretrained_model = Model[pretrained_hparams["model_name"]].cls(**pretrained_hparams)
            self.pretrained_hparams = pretrained_hparams
        else:
            # check if we have a PyTorch Lightning checkpoint or a pure pytorch checkpoint
            if pretrained_ckpt_path.endswith(".ckpt"):
                # in this scenario, the pre-trained weights are loaded from the specified checkpoint file
                # the checkpoint file is a RosettaTask checkpoint, containing source model weights
                pretrained_checkpoint = tasks.RosettaTask.load_from_checkpoint(pretrained_ckpt_path, pdb_fns=pdb_fns)
                # update the pre-trained model hparams with the new pdb_fns
                pretrained_checkpoint.hparams["pdb_fns"] = pdb_fns
                self.pretrained_hparams = pretrained_checkpoint.hparams
                pretrained_model = pretrained_checkpoint.model

            else:
                # load from PyTorch checkpoint
                ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
                state_dict = ckpt["state_dict"]
                pretrained_hparams = ckpt["hyper_parameters"]

                # update the pre-trained model hparams with the new pdb_fns
                pretrained_hparams["pdb_fns"] = pdb_fns
                self.pretrained_hparams = pretrained_hparams

                # create the model and load pretrained weights from the state dict
                pretrained_model = Model[pretrained_hparams["model_name"]].cls(**pretrained_hparams)
                pretrained_model.load_state_dict(state_dict)

        layers = collections.OrderedDict()

        # set the backbone to all layers except the last layer (the pre-trained prediction layer)
        if backbone_cutoff is None:
            layers["backbone"] = SequentialWithArgs(*list(pretrained_model.model.children()))
        else:
            layers["backbone"] = SequentialWithArgs(*list(pretrained_model.model.children())[0:backbone_cutoff])

        # change the dropout rate of the backbone if specified
        if new_backbone_dropout_rate is not None:
            layers["backbone"].apply(lambda m: modify_dropout_rate(m, new_backbone_dropout_rate))

        # add dropout after backbone if specified
        if dropout_after_backbone:
            layers["dropout"] = nn.Dropout(dropout_after_backbone_rate)

        if top_net_type == "sklearn":
            # sklearn top not doesn't require any more layers, just return model for the repr layer
            self.model = SequentialWithArgs(layers)
            return

        # figure out dimensions of input into the prediction layer
        if pred_layer_input_features is None:
            # todo: can make this more robust by checking if the pretrained_mode.hparams for use_final_hidden_layer,
            #   global_average_pooling, etc. then can determine what the layer will be based on backbone_cutoff.
            # currently, assumes that pretrained_model uses global average pooling and a final_hidden_layer
            if backbone_cutoff is None:
                # no backbone cutoff... use the full network (including tasks) as the backbone
                pred_layer_input_features = self.pretrained_hparams["num_tasks"]
            elif backbone_cutoff == -1:
                pred_layer_input_features = self.pretrained_hparams["final_hidden_size"]
            elif backbone_cutoff == -2:
                pred_layer_input_features = self.pretrained_hparams["embedding_len"]
            elif backbone_cutoff == -3:
                pred_layer_input_features = self.pretrained_hparams["embedding_len"] * kwargs["aa_seq_len"]
            else:
                raise ValueError("can't automatically determine pred_layer_input_features for given backbone_cutoff")

        layers["flatten"] = nn.Flatten(start_dim=1)

        # create a new prediction layer on top of the backbone
        if top_net_type == "linear":
            # linear layer for prediction
            layers["prediction"] = nn.Linear(in_features=pred_layer_input_features, out_features=1)
        elif top_net_type == "nonlinear":
            # fully connected with hidden layer
            fc_block = FCBlock(in_features=pred_layer_input_features,
                               num_hidden_nodes=top_net_hidden_nodes,
                               use_batchnorm=top_net_use_batchnorm,
                               use_layernorm=top_net_use_layernorm,
                               norm_before_activation=top_net_norm_before_activation,
                               use_dropout=top_net_use_dropout,
                               dropout_rate=top_net_dropout_rate)

            pred_layer = nn.Linear(in_features=top_net_hidden_nodes, out_features=1)

            layers["prediction"] = SequentialWithArgs(fc_block, pred_layer)
        else:
            raise ValueError("Unexpected type of top net layer: {}".format(top_net_type))

        self.model = SequentialWithArgs(layers)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


def get_activation_fn(activation, functional=True):
    if activation == "relu":
        return F.relu if functional else nn.ReLU()
    elif activation == "gelu":
        return F.gelu if functional else nn.GELU()
    elif activation == "silo" or activation == "swish":
        return F.silu if functional else nn.SiLU()
    elif activation == "leaky_relu" or activation == "lrelu":
        return F.leaky_relu if functional else nn.LeakyReLU()
    else:
        raise RuntimeError("unknown activation: {}".format(activation))


class Model(enum.Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, cls, transfer_model):
        self.cls = cls
        self.transfer_model = transfer_model

    linear = LRModel, False
    fully_connected = FCModel, False
    cnn = ConvModel, False
    cnn2 = ConvModel2, False
    transformer_encoder = AttnModel, False
    transfer_model = TransferModel, True


def main():
    pass


if __name__ == "__main__":
    main()
