from functools import partial
from typing import List
from typing import Optional

import attr


@attr.s(auto_attribs=True)
class ModelParams:
    # encoder model selection
    encoder_arch: str = "resnet18" # resnet18, resnet34, resnet50
    # note that embedding_dim is 512 * expansion parameter in ws_resnet
    embedding_dim: int = 512 * 1  # must match embedding dim of encoder
    # projection size
    dim: int = 64 


    # optimization parameters
    optimizer_name: str = "lars"
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 320
    final_lr_schedule_value: float = 0.0
    lars_warmup_epochs: int = 1
    lars_eta: float = 1e-3
    exclude_matching_parameters_from_lars: List[str] = []  # set to [".bias", ".bn"] to match paper


    # loss parameters
    loss_constant_factor: float = 1
    invariance_loss_weight: float = 25.0
    variance_loss_weight: float = 25.0
    covariance_loss_weight: float = 1.0
    variance_loss_epsilon: float = 1e-04
    kmeans_weight: float = 1e-03

    # MLP parameters
    projection_mlp_layers: int = 2
    prediction_mlp_layers: int = 0 # by default prediction mlp is identity
    mlp_hidden_dim: int = 512
    mlp_normalization: Optional[str] = None
    prediction_mlp_normalization: Optional[str] = "same"  # if same will use mlp_normalization
    use_mlp_weight_standardization: bool = False



# Differences between these parameters and those used in the paper (on image net):
# max_epochs=1000,
# lr=1.6,
# batch_size=2048,
# weight_decay=1e-6,
# mlp_hidden_dim=8192,
# dim=8192,
VICRegParams = partial(
    ModelParams,
    exclude_matching_parameters_from_lars=[".bias", ".bn"],
    projection_mlp_layers=3,
    final_lr_schedule_value=0.002,
    mlp_normalization="bn",
    lars_warmup_epochs=10,
    kmeans_weight=1e-03,
)
