""" pytorch lightning finetuning callback for transfer learning """
import logging
from functools import reduce
from typing import Callable, Optional, Dict, Any, Union, Iterable

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim import Optimizer

log = logging.getLogger(__name__)


def multiplicative(epoch):
    return 2


def get_module_by_name(module, access_string):
    """ https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8 """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


class AnyFinetuning(BaseFinetuning):
    """ It's the Pytorch Lightning backbone finetuning callback, but modified to support different name for
        the backbone (due to how I have my model set up and preference)
        # todo: if updating to a new version of PyTorch Lightning, check if this is still correct
        https://github.com/PyTorchLightning/pytorch-lightning/blob/45c45dc7b018f9a2db60f5df1a3f7dbbb45ccb36/pytorch_lightning/callbacks/finetuning.py
        https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/finetuning.html#BackboneFinetuning
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.BaseFinetuning.html """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        always_align_lr: bool = False,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
        backbone_access_string: str = "model.model.backbone"
    ) -> None:
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.always_align_lr: bool = always_align_lr
        self.lambda_func: Callable = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align

        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_backbone_lr: Optional[float] = None
        self.backbone_access_string = backbone_access_string

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict["previous_backbone_lr"]
        super().load_state_dict(state_dict)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # make sure the backbone exists (via backbone_access_string)
        try:
            backbone = get_module_by_name(pl_module, self.backbone_access_string)
        except AttributeError:
            raise MisconfigurationException("The LightningModule should have a nn.Module `{}` attribute".format(
                self.backbone_access_string))

        if not isinstance(backbone, Module):
            raise MisconfigurationException("The LightningModule should have a nn.Module `{}` attribute".format(
                self.backbone_access_string))

        return super().on_fit_start(trainer, pl_module)

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(get_module_by_name(pl_module, self.backbone_access_string))

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]

            if self.always_align_lr:
                initial_backbone_lr = current_lr
            else:
                initial_backbone_lr = (
                    self.backbone_initial_lr
                    if self.backbone_initial_lr is not None
                    else current_lr * self.backbone_initial_ratio_lr
                )

            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                get_module_by_name(pl_module, self.backbone_access_string),
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]

            # handle a special w/ 0 initial value for backbone....
            if self.always_align_lr:
                next_current_backbone_lr = current_lr
            else:
                next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
                next_current_backbone_lr = (
                    current_lr
                    if (self.should_align and next_current_backbone_lr > current_lr)
                    else next_current_backbone_lr
                )

            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.rounding)}"
                )

    @staticmethod
    def unfreeze_and_add_param_group(
            modules: Union[Module, Iterable[Union[Module, Iterable]]],
            optimizer: Optimizer,
            lr: Optional[float] = None,
            initial_denom_lr: float = 10.0,
            train_bn: bool = True,
    ) -> None:

        BaseFinetuning.make_trainable(modules)
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)

        # note: it's necessary to override this method to avoid a bug with finetuning ESM
        # ESM includes the same parameter (the AA embedding weights) multiple times
        # So we modify this function to add an extra check and remove those params from ESM
        # todo: follow up with the PyTorch Lightning and/or ESM teams
        #   https://github.com/Lightning-AI/lightning/issues/16465
        filter_unique_params = True
        if filter_unique_params:
            unique_params = set()
            unique_params_list = []
            for param in params:
                if param not in unique_params:
                    unique_params.add(param)
                    unique_params_list.append(param)
                else:
                    print("Filtered out a duplicate parameter when unfreezing the backbone")
                    # for name, model_param in modules.named_parameters():
                    #     if torch.equal(param, model_param):
                    #         print(f'Removing duplicate parameter from: {name}')

            if unique_params_list:
                optimizer.add_param_group({"params": unique_params_list, "lr": params_lr / denom_lr})
        elif params:
            optimizer.add_param_group({"params": params, "lr": params_lr / denom_lr})


class BackboneFreezer(Callback):
    def __init__(self, backbone_access_string: str = "model.model.backbone") -> None:
        super().__init__()
        self.backbone_access_string = backbone_access_string

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        try:
            backbone = get_module_by_name(pl_module, self.backbone_access_string)
        except AttributeError:
            raise MisconfigurationException("The LightningModule should have a nn.Module `{}` attribute".format(
                self.backbone_access_string))

        if not isinstance(backbone, Module):
            raise MisconfigurationException("The LightningModule should have a nn.Module `{}` attribute".format(
                self.backbone_access_string))

        # use BaseFinetuning freeze function for convenience
        BaseFinetuning.freeze(get_module_by_name(pl_module, self.backbone_access_string))


class BackboneUnfreezer(Callback):
    """ note this does NOT add params back to the optimizer... so optimizer needs to be reconfigured...
        or... new task needs to be loaded... """
    def __init__(self, backbone_access_string: str = "model.model.backbone") -> None:
        super().__init__()
        self.backbone_access_string = backbone_access_string

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        try:
            backbone = get_module_by_name(pl_module, self.backbone_access_string)
        except AttributeError:
            raise MisconfigurationException("The LightningModule should have a nn.Module `{}` attribute".format(
                self.backbone_access_string))

        if not isinstance(backbone, Module):
            raise MisconfigurationException("The LightningModule should have a nn.Module `{}` attribute".format(
                self.backbone_access_string))

        # use BaseFinetuning freeze function for convenience
        BaseFinetuning.make_trainable(get_module_by_name(pl_module, self.backbone_access_string))
