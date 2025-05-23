""" PyTorch Lightning tasks """

from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

try:
    from . import training_utils
    from .training_utils import CosineWarmupScheduler, ConstantWarmupScheduler
    from .datamodules import DMSDataModule
    from . import models
except ImportError:
    import training_utils
    from training_utils import CosineWarmupScheduler, ConstantWarmupScheduler
    from datamodules import DMSDataModule
    import models


class RosettaTask(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument('--learning_rate', type=float, default=0.0001)
        p.add_argument('--lr_scheduler', type=str, default="constant", choices=["constant", "warmup_cosine_decay",
                                                                                "warmup_constant"])
        p.add_argument('--warmup_steps', type=float, default=.02,
                       help="number or fraction of warmup steps for warmup_cosine_decay")
        return p

    @staticmethod
    def custom_mse_loss(inputs, targets):
        """ MSE loss function that works for both single-task and multitask
            For multitask, computes MSE for each task and takes sum """
        return torch.sum(torch.mean((inputs - targets) ** 2, dim=0))

    def __init__(self,
                 # the model to use (see models.py)
                 model_name: str,
                 # input data params, important for model construction
                 num_tasks: int,
                 num_tokens: int,
                 aa_seq_len: int,
                 aa_encoding_len: int,
                 # seq_encoding_len defaults to 0 solely for backwards compatability with checkpoints from
                 # before this was added as an argument
                 seq_encoding_len: int = 0,
                 # pdb files used for setting up relative position 3D. optional.
                 pdb_fns: List[str] = None,
                 # optimizer params
                 learning_rate: float = .0001,
                 lr_scheduler: str = "constant",
                 warmup_steps: float = .02,
                 # example input array from the datamodule
                 example_input_array: torch.Tensor = None,
                 # this is a static hyperparameter used to identify the task type
                 _task_type: str = "rosetta",
                 # all other trainer and model params
                 *args, **kwargs):

        super().__init__()

        self.save_hyperparameters(ignore=["example_input_array"])

        self.pdb_fns = pdb_fns
        self.num_tasks = num_tasks
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps

        self.example_input_array = example_input_array

        # create the model based on the model name and arguments
        # args from the datamodule (aa_seq_len, num_tokens, etc.) must be forwarded to model
        self.model = models.Model[model_name].cls(num_tasks=num_tasks,
                                                  aa_seq_len=aa_seq_len,
                                                  aa_encoding_len=aa_encoding_len,
                                                  seq_encoding_len=seq_encoding_len,
                                                  num_tokens=num_tokens,
                                                  pdb_fns=pdb_fns,
                                                  **kwargs)

        # collection of metrics for computing pearson's on test set
        self.test_pearson_collection = nn.ModuleList([torchmetrics.PearsonCorrCoef() for _ in range(num_tasks)])

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _shared_step(self, data_batch, batch_idx, compute_loss=True):
        inputs = data_batch["inputs"]
        labels = data_batch["targets"] if "targets" in data_batch else None
        # the pdb file if one is provided by the dataloader (for relative position 3D)
        # we only support one pdb file per batch, so just choose the first one (index 0)
        # in the future, if we support multiple per batch, we can pass in all the pdb files
        pdb_fn = data_batch["pdb_fns"][0] if "pdb_fns" in data_batch else None

        outputs = self(inputs, pdb_fn=pdb_fn)
        if compute_loss:
            loss = self.custom_mse_loss(outputs, labels)
            return outputs, loss
        else:
            return outputs

    def training_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        # sync_dist=False because we always run the test set on 1 GPU
        self.log("test_loss", loss, sync_dist=False)

        labels = data_batch["targets"]

        for task_num in range(self.num_tasks):
            outputs_task = outputs[:, task_num]
            labels_task = labels[:, task_num]
            self.test_pearson_collection[task_num](outputs_task, labels_task)

        for task_num in range(self.num_tasks):
            task_name = self.trainer.datamodule.target_names[task_num]
            self.log("test/pearson_{}".format(task_name), self.test_pearson_collection[task_num],
                     on_step=False, on_epoch=True)

    def predict_step(self, data_batch, batch_idx, dataloader_idx=0):
        # if we use dataloader_idx in the future, need to handle that in _shared_step
        # or do the output computation in this function
        outputs = self._shared_step(data_batch, batch_idx, compute_loss=False)
        return outputs

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.lr_scheduler == "warmup_cosine_decay":

            # the number of training steps
            training_steps = self.trainer.estimated_stepping_batches
            print("Number of training steps is {}".format(training_steps))

            # calculate the number of warmup steps (could be absolute number or fraction of total steps)
            ws = self.warmup_steps
            if 0 < self.warmup_steps < 1:
                ws = self.warmup_steps * training_steps
            print("Number of warmup steps is {}".format(ws))

            lr_scheduler_config = {
                "scheduler": CosineWarmupScheduler(optimizer, warmup=ws, max_iters=training_steps),
                "interval": "step",
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        elif self.lr_scheduler == "warmup_constant":

            # the number of warmup steps
            ws = self.warmup_steps
            if 0 < self.warmup_steps < 1:
                # only calculate estimated stepping batches if warmup steps is a fraction
                training_steps = self.trainer.estimated_stepping_batches
                print("Number of training steps is {}".format(training_steps))
                ws = self.warmup_steps * training_steps
            print("Number of warmup steps is {}".format(ws))

            # set up the learning rate scheduler configuration
            lr_scheduler_config = {
                "scheduler": ConstantWarmupScheduler(optimizer, warmup=ws),
                # "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, constant_warmup(ws)),
                "interval": "step",
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        elif self.lr_scheduler == "constant":
            return optimizer
        else:
            raise ValueError("unknown learning rate scheduler: {}".format(self.lr_scheduler))


class DMSTask(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        # add arguments related to the optimizer
        parser_with_optimizer_args = training_utils.OptimizerConfig.add_model_specific_args(parent_parser)
        # add arguments related to the loss function
        p = ArgumentParser(parents=[parser_with_optimizer_args], add_help=False)
        p.add_argument('--loss_func', type=str, default="mse", choices=["mse", "corn"] ,
                       help= """Specifies the loss function to use. "mse" refers to mean squared error. 
                                For ordinal data, consider using the "corn" loss function, which requires also specifying 
                                --top_net_output_dim (this should be equal to N-1, where N is the number of bins in the data). 
                                For more details, see the paper:
                                    Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities — 
                                    https://doi.org/10.48550/arXiv.2111.08851""")
        p.add_argument('--corn_pred_feature', type=str, default="raw_predictions",
                       help="""Specifies which CORN feature to use for calculating test metrics and generating scatter plots (for training log purposes only). 
                            "raw_predictions" refers to the default behavior: selecting the highest-ranking bin with a probability greater than 0.5. 
                            An input of "1" will use the probability of the second-lowest rank bin (the lowest bin always has P(dead) = 1). 
                            An input of "N" (where N is the number of bins) will use the probability of the highest-ranking bin. 
                            This setting does not affect the saved predictions—raw_predictions and all bin probabilities will be saved regardless.""")
        p.add_argument("--importance_weights",
                            help= """Weights to assign to ordinal classes. For MSE loss, the weight vector should have dimension equal to the number of classes. 
                                    For ranking tasks such as CORN or CORAL loss, the vector dimension should be (num_classes - 1). 
                                    If None is passed (i.e., not specified), the importance weights will be automatically calculated based on 
                                    class prevalence in the training set. 
                                    Note: To use importance weights, class labels must range from 0 to N-1, where N is the number of classes.""",
                            type=list, default=None, nargs='+')
        p.add_argument("--use_importance_weights",
                            help="""If you wish to use importance weights, you must specify this flag.
                                    If --importance_weights is not provided, they will be automatically calculated based on 
                                    class prevalence in the training set.
                                    Note: To use importance weights, class labels must range from 0 to N-1, 
                                    where N-1 is the highest-ranking class.
                                    Must specify --num_classes if using importance weights with MSE loss.""",
                                #todo: provide an error message telling the user that you must pass in these
                                # labels for ordinal work.
                             action="store_true")
        return p
    def __init__(self,
                 # the model to use (see models.py)
                 model_name: str,
                 # input data params, important for model construction
                 num_tasks: int,
                 num_tokens: int,
                 aa_seq_len: int,
                 aa_encoding_len: int,
                 # seq_encoding_len defaults to 0 solely for backwards compatability with checkpoints from
                 # before this was added as an argument
                 seq_encoding_len: int = 0,
                 # pdb files used for setting up relative position 3D. optional.
                 pdb_fns: List[str] = None,
                 # optimizer params
                 optimizer: str = "adamw",
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.0001,
                 lr_scheduler: str = "constant",
                 warmup_steps: float = .02,
                 phase2_lr_ratio: float = 1.0,
                 # example input array from the datamodule
                 example_input_array: torch.Tensor = None,
                 # this is a static hyperparameter used to identify the task type
                 _task_type: str = "dms",
                 # all other trainer and model params
                 save_hyperparams=True,
                 loss_func: str = "mse",
                 #specific corn loss feature, which feature to use for metrics
                 corn_pred_feature: str = 'raw_predictions',
                 use_importance_weights: bool = False,
                 importance_weights: list = None,
                 # pass in the datamodule to be used for importance weights
                 dm: DMSDataModule =None,
                 *args, **kwargs):

        super().__init__()

        if save_hyperparams:
            self.save_hyperparameters(ignore=["example_input_array"])

        if num_tasks > 1:
            raise NotImplementedError("DMSTask currently only supports 1 task "
                                      "(eval metrics not set up for multiple tasks)")

        self.pdb_fns = pdb_fns
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.phase2_lr_ratio = phase2_lr_ratio

        self.example_input_array = example_input_array

        # create the model based on the model name and arguments
        # args from the datamodule (aa_seq_len, num_tokens, etc.) must be forwarded to model
        self.model = models.Model[model_name].cls(num_tasks=num_tasks,
                                                  aa_seq_len=aa_seq_len,
                                                  aa_encoding_len=aa_encoding_len,
                                                  seq_encoding_len=seq_encoding_len,
                                                  num_tokens=num_tokens,
                                                  pdb_fns=pdb_fns,
                                                  **kwargs)

        self.test_pearson = torchmetrics.PearsonCorrCoef()
        self.test_spearman = torchmetrics.SpearmanCorrCoef()


        ## corn specific loss terms
        self.loss_func = loss_func
        self.corn_pred_feature = corn_pred_feature
        # if the model type is a TransferModel, we also want to save the
        # hyperparameters of the pre-trained model that we are transferring from
        # this will help reconstruct the model when loading from a checkpoint
        # this needs to be done after the model is created
        if save_hyperparams:
            if isinstance(self.model, models.TransferModel):
                # put the pre-trained model hyperparameters in a separate namespace
                # so that they don't overwrite the current model's hyperparameters when saving
                # the reason we convert to a dict is currently they are
                # a pytorch_lightning.utilities.parsing.AttributeDict, and we want to be fully pure-pytorch compatible
                pretrained_hparams = dict(self.model.pretrained_hparams)
                self.save_hyperparameters({"pretrained_hparams": pretrained_hparams})


        # class imbalance training with an ordinal function
        if importance_weights is not None and use_importance_weights == False:
            raise ValueError(
                """--importance_weights (used for class imbalance training on ordinal data) 
            was specified, however, the flag --use_importance_weights was not specified!
            Please pass in the --use_importance_weights flag to use --importance_weights, 
            or do not specify --importance_weights at all."""
            )
        if use_importance_weights:
            if importance_weights is None:

                # going to make an assumption the data must be labeled from 0 to N-1. (Where N is number of classes)

                unique, counts = np.unique(dm._get_raw_targets("train").flatten(), return_counts=True)
                count_dict = dict(sorted(zip(unique, counts)))
                count_dict = {int(k): v for k, v in count_dict.items()}

                if loss_func=='corn':
                    nb_classes = kwargs['top_net_output_dim'] + 1
                    raw_importance_weights = torch.tensor([(count_dict[i]+count_dict[i+1])/2 for i in np.arange(0,nb_classes-1,1)])
                    importance_weights_inverse = (1 / raw_importance_weights)
                    self.importance_weights = importance_weights_inverse / importance_weights_inverse.sum()
                elif loss_func=='mse':
                    nb_classes = kwargs['num_classes']
                    raw_importance_weights = torch.tensor([count_dict[i] for i in np.arange(0, nb_classes, 1)])
                    importance_weights_inverse=(1 / raw_importance_weights)
                    self.importance_weights =importance_weights_inverse/ importance_weights_inverse.sum()
            else:
                # the importance weights were specified by the user
                importance_weights = [float(im[0]) for im in importance_weights]
                # tensor([0.0981, 0.1324, 0.1738, 0.5957])
                # tensor([0.4048, 0.3000, 0.2286, 0.0667])
                if loss_func == 'corn':
                    nb_classes = kwargs['top_net_output_dim'] + 1
                    assert len(importance_weights) == nb_classes - 1, \
                        'Importance length must be N-1 for ordinal based loss, where N is the number of output classes'
                elif loss_func=='mse':
                    nb_classes = kwargs['num_classes']
                    assert len(importance_weights) == nb_classes , \
                        'Importance length must be N for MSE, where N is the number of output classes'
                else:
                    raise ValueError('must choose a valid loss function from specified options')

                self.importance_weights = torch.tensor(importance_weights)

        else:
            # we don't want to use importance weights, the default case
            self.importance_weights = None

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _shared_step(self, data_batch, batch_idx, compute_loss=True):
        inputs = data_batch["inputs"]

        # labels might not be provided if we are doing inference
        labels = data_batch["targets"] if "targets" in data_batch else None

        # the pdb file if one is provided by the dataloader (for relative position 3D)
        # we only support one pdb file per batch, so just choose the first one (index 0)
        # in the future, if we support multiple per batch, we can pass in all the pdb files
        pdb_fn = data_batch["pdb_fns"][0] if "pdb_fns" in data_batch else None

        # auxiliary inputs for the model
        # if they are not provided, we just pass in an empty dictionary
        aux = data_batch.get("aux", {})

        outputs = self(inputs, pdb_fn=pdb_fn, **aux)
        if compute_loss:
            if self.loss_func == "mse":
                if self.importance_weights is None:
                    loss = F.mse_loss(outputs, labels)
                else:
                    loss= self.weighted_mse_loss(outputs,labels,self.importance_weights)

                return outputs, loss
            elif self.loss_func == "corn":
                print(self.importance_weights)
                num_classes = outputs.shape[1] + 1
                loss = self.loss_corn(outputs, labels, num_classes,self.importance_weights)
                outputs = self._shared_step_corn_inference(outputs)
                return outputs, loss

        else:
            if self.loss_func == "mse":
                pass
            elif self.loss_func == "corn":
                outputs=self._shared_step_corn_inference(outputs)
            return outputs

    def weighted_mse_loss(self,outputs, labels, class_weights):
        # Ensure outputs are flattened
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        # Get weights per sample based on class label
        sample_weights = class_weights[labels.long()]

        # Compute element-wise squared error
        mse_per_sample = (outputs - labels.float()) ** 2

        # Apply weights
        weighted_mse = sample_weights * mse_per_sample

        # Return mean of weighted loss
        return weighted_mse.mean()

    def training_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        # track the validation loss as the hparam metric in tensorboard
        self.log("hp_metric", loss)
        return loss

    def test_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)


        if self.loss_func=='corn':
            # if we have a corn loss we need the user specified prediction feature
            outputs = self._get_corn_outputs(outputs)

        self.log("test_loss", loss)

        labels = data_batch["targets"]
        self.test_pearson(torch.squeeze(outputs, dim=-1), torch.squeeze(labels, dim=-1))
        self.test_spearman(torch.squeeze(outputs, dim=-1), torch.squeeze(labels, dim=-1))

        self.log("test_pearson", self.test_pearson, on_step=False, on_epoch=True)
        self.log("test_spearman", self.test_spearman, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, data_batch, batch_idx, dataloader_idx=0):

        outputs = self._shared_step(data_batch, batch_idx, compute_loss=False)

        if self.loss_func=='corn':
            # if we have a corn loss we need the user specified prediction feature
            log_prediction = self._get_corn_outputs(outputs)
            # return two values to save all probabilities (included in outputs)
            # while also having a prediction for scatterplots, test metrics, etc.
            # predictions will be handled by training_utils.parse_raw_preds_and_save()
            return (log_prediction,outputs)

        return outputs

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        # these params are used for dual phase warmup learning rate scheduler
        unfreeze_backbone_at_epoch = None
        if hasattr(self, "hparams") and self.hparams is not None:
            unfreeze_backbone_at_epoch = self.hparams.get("unfreeze_backbone_at_epoch", None)

        optimizer_config = training_utils.OptimizerConfig(self.optimizer,
                                                          self.weight_decay,
                                                          self.learning_rate,
                                                          self.lr_scheduler,
                                                          self.warmup_steps,
                                                          self.phase2_lr_ratio,
                                                          unfreeze_backbone_at_epoch=unfreeze_backbone_at_epoch,
                                                          max_epochs=self.trainer.max_epochs)

        return optimizer_config.get_optimizer_config(trainable_parameters, self.trainer.estimated_stepping_batches)

    def loss_corn(self,logits, y_train, num_classes, importance_weights=None):
        sets = []

        for i in range(num_classes - 1):
            label_mask = y_train > i - 1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.

        if importance_weights is None:
            importance_weights = torch.ones(len(sets))

        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples.flatten(), task_index]

            loss = -torch.sum(F.logsigmoid(pred) * train_labels
                              + (F.logsigmoid(pred) - pred) * (1 - train_labels))

            # losses += loss
            losses += importance_weights[task_index] * loss

        return losses / num_examples

    def proba_from_logits(self, logits):
        # new specific functions only for CORN loss
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        return probas

    def label_from_logits(self, logits):
        """ Converts logits to class labels.
        This is function is specific to CORN.
        """
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        return predicted_labels

    def _shared_step_corn_inference(self, outputs):
        labels = self.label_from_logits(outputs)  # shape: (batch_size, 1) or (batch_size,)
        probas = self.proba_from_logits(outputs)  # shape: (batch_size, num_classes)
        # If labels is 1D, unsqueeze it to make it 2D for stacking
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)  # shape: (batch_size, 1)
        # Now horizontally stack them
        outputs = torch.cat([labels, probas], dim=1)  # shape: (batch_size, 1 + num_classes)
        return outputs

    def _get_corn_outputs(self,outputs):
        if self.corn_pred_feature == 'raw_predictions':
            return outputs[:, 0]
        else:
            # if we want a probability input it's the bin that was input
            # with 1 indexing.
            return outputs[:, int(self.corn_pred_feature)-1]

class SKTopNetTask(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch["inputs"]
        labels = batch["targets"]
        pdb_fn = batch["pdb_fns"][0] if "pdb_fns" in batch else None

        outputs = self(inputs, pdb_fn=pdb_fn)
        return outputs

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
