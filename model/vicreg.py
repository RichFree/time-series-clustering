import copy
import math
import warnings
from functools import partial
from typing import Optional
from typing import Union

import attr
import torch
import torch.nn.functional as F
# from pytorch_lightning.utilities import AttributeDict
from lightning.fabric.utilities.data import AttributeDict
from torch.utils.data import DataLoader
import lightning as L

import model.utils as utils
from model.batchrenorm import BatchRenorm1d
from model.lars import LARS
from model.model_params import ModelParams
# from sklearn.linear_model import LogisticRegression
# from sklearn.cluster import KMeans
# from sklearn.metrics import rand_score, normalized_mutual_info_score

import pandas as pd



def get_mlp_normalization(hparams: ModelParams, prediction=False):
    normalization_str = hparams.mlp_normalization
    if prediction and hparams.prediction_mlp_normalization != "same":
        normalization_str = hparams.prediction_mlp_normalization

    if normalization_str is None:
        return None
    elif normalization_str == "bn":
        return partial(torch.nn.BatchNorm1d, num_features=hparams.mlp_hidden_dim)
    elif normalization_str == "br":
        return partial(BatchRenorm1d, num_features=hparams.mlp_hidden_dim)
    elif normalization_str == "ln":
        return partial(torch.nn.LayerNorm, normalized_shape=[hparams.mlp_hidden_dim])
    elif normalization_str == "gn":
        return partial(torch.nn.GroupNorm, num_channels=hparams.mlp_hidden_dim, num_groups=32)
    else:
        raise NotImplementedError(f"mlp normalization {normalization_str} not implemented")

# class KMeansLoss:
# 
#     def __init__(self, num_clusters, embedding_dim, device):
#         self.num_clusters = num_clusters
#         self.centroids = torch.randn(num_clusters, embedding_dim, device=device)
#         self.device=device
#     
#     def update_centroids(self, embeddings, assignments):
#         for i in range(self.num_clusters):
#             assigned_embeddings = embeddings[assignments == i]
#             if len(assigned_embeddings) > 1: # good if more than singleton
#                 # implement ewma update for centroids
#                 weight1 = torch.tensor(0.3, device='cpu')
#                 weight2 = torch.tensor(0.7, device='cpu') # give more weight to new embeddings
#                 self.centroids[i] = self.centroids[i] * weight1 + assigned_embeddings.mean(dim=0).cpu() * weight2
# 
#     def set_centroids(self, embeddings, assignments):
#         for i in range(self.num_clusters):
#             assigned_embeddings = embeddings[assignments == i]
#             if len(assigned_embeddings) > 1: # good if more than singleton
#                 # implement ewma update for centroids
#                 self.centroids[i] = assigned_embeddings.mean(dim=0).cpu()
# 
#     
#     def compute_loss(self, embeddings):
#         # move centroids to same device as embeddings
#         centroids = self.centroids.to(embeddings.device)
#         distances = torch.cdist(embeddings, centroids, p=self.num_clusters)
#         min_distances, assignments = distances.min(dim=1)
#         loss = min_distances.pow(2).sum()
#         return loss, assignments
#     
#     def forward(self, embeddings, step_count):
#         loss, assignments = self.compute_loss(embeddings)
#         detached_embeddings = embeddings.detach()
#         detached_assignments = assignments.detach()
# 
#         if (step_count < 5):
#             self.set_centroids(detached_embeddings, detached_assignments)
#         if (step_count % 2 == 0):
#             self.update_centroids(detached_embeddings, detached_assignments)
#         return loss


class SelfSupervisedMethod(L.LightningModule):
    model: torch.nn.Module
    hparams: AttributeDict
    embedding_dim: Optional[int]

    def __init__(
        self,
        hparams: Union[ModelParams, dict, None] = None,
        **kwargs,
    ):
        super().__init__()

        # disable automatic optimization for lightning2
        self.automatic_optimization = False
        self.optimizer = None
        self.lr_scheduler = None

        # load from arguments
        if hparams is None:
            hparams = self.params(**kwargs)
        # if it is already an attributedict, then use it directly
        if hparams is not None:
            self.save_hyperparameters(attr.asdict(hparams))


        # Create encoder model
        self.model = utils.get_encoder(hparams.encoder_arch)

        # projection_mlp_layers = 3
        self.projection_model = utils.MLP(
            hparams.embedding_dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.projection_mlp_layers,
            normalization=get_mlp_normalization(hparams),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        # by default it is identity
        # prediction_mlp_layers = 0
        self.prediction_model = utils.MLP(
            hparams.dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.prediction_mlp_layers,
            normalization=get_mlp_normalization(hparams, prediction=True),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        # kmeans loss
        # self.kmeans_loss = KMeansLoss(num_clusters=hparams.num_clusters, embedding_dim=hparams.dim, device=self.device)


    def _get_embeddings(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            embeddings, targets
        """
        bsz, nd, nc, nh, nw = x.shape
        assert nd == 2, "second dimension should be the split image -- dims should be N2CHW"
        im_q = x[:, 0].contiguous()
        im_k = x[:, 1].contiguous()

        # compute query features
        emb_q = self.model(im_q)
        q_projection = self.projection_model(emb_q)
        # by default vicreg gives an identity for prediction model
        q = self.prediction_model(q_projection)  # queries: NxC
        emb_k = self.model(im_k)
        k_projection = self.projection_model(emb_k)
        k = self.prediction_model(k_projection)  # queries: NxC
        # q and k are the projection embeddings
        
        return emb_q, q, k


    def _get_vicreg_loss(self, z_a, z_b, batch_idx):
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a)) # differentiable max
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.hparams.invariance_loss_weight
        weighted_var = loss_var * self.hparams.variance_loss_weight
        weighted_cov = loss_cov * self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov


        return loss


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, class_labels = batch  # batch is a tuple, we just want the image

        emb_q, q, k = self._get_embeddings(x)

        vicreg_loss = self._get_vicreg_loss(q, k, batch_idx)

        total_loss = vicreg_loss.mean() * self.hparams.loss_constant_factor

        # here lies the manual optimizing code
        # Backward pass
        self.optimizer.zero_grad()
        self.manual_backward(total_loss)
        self.optimizer.step()
        self.lr_scheduler.step()
        # # Optimizer step
        # opt.step()
        # opt.zero_grad()
        # # Learning rate scheduler step (if using one that steps every batch)
        # self.lr_scheduler_step(scheduler)

        log_data = {
            "step_train_loss": total_loss,
        }

        self.log_dict(log_data, sync_dist=True, prog_bar=True)
        return {"loss": total_loss}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     _, q, k = self._get_embeddings(x)
    #     loss = self._get_vicreg_loss(q,k,batch_idx) * self.hparams.loss_constant_factor
    #     self.log('val_loss', loss)  # Log the validation loss
    #     return loss

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad is False:
                continue

            # for vicreg
            # exclude_matching_parameters_from_lars=[".bias", ".bn"],
            if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {
                "params": regular_parameters, 
                "names": regular_parameter_names, 
                "use_lars": True
            },
            {
                "params": excluded_parameters,
                "names": excluded_parameter_names,
                "use_lars": False,
                "weight_decay": 0,
            },
        ]
        if self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer_name == "lars":
            optimizer = partial(LARS, warmup_epochs=self.hparams.lars_warmup_epochs, eta=self.hparams.lars_eta)
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        self.optimizer = optimizer(
            param_groups,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            self.hparams.max_epochs,
            eta_min=self.hparams.final_lr_schedule_value,
        )
        return None # [encoding_optimizer], [self.lr_scheduler]

    # def configure_optimizers(self):
    #     # exclude bias and batch norm from LARS and weight decay
    #     regular_parameters = []
    #     regular_parameter_names = []
    #     excluded_parameters = []
    #     excluded_parameter_names = []
    #     for name, parameter in self.named_parameters():
    #         if parameter.requires_grad is False:
    #             continue
    #         if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
    #             excluded_parameters.append(parameter)
    #             excluded_parameter_names.append(name)
    #         else:
    #             regular_parameters.append(parameter)
    #             regular_parameter_names.append(name)

    #     param_groups = [
    #         # use LARS with weight decay
    #         {
    #             "params": regular_parameters, 
    #             "names": regular_parameter_names, 
    #             "use_lars": True
    #         },
    #         # # not use LARS with no weight decay
    #         # {
    #         #     "params": excluded_parameters,
    #         #     "names": excluded_parameter_names,
    #         #     "use_lars": False,
    #         #     "weight_decay": 0,
    #         # },
    #     ]
    #     optimizer = partial(LARS, warmup_epochs=self.hparams.lars_warmup_epochs, eta=self.hparams.lars_eta)

    #     encoding_optimizer = optimizer(
    #         param_groups,
    #         lr=self.hparams.lr,
    #         momentum=self.hparams.momentum,
    #         weight_decay=self.hparams.weight_decay,
    #     )
    #     self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         encoding_optimizer,
    #         self.hparams.max_epochs,
    #         eta_min=self.hparams.final_lr_schedule_value,
    #     )
    #     self.optimizer = encoding_optimizer
    #     # return [encoding_optimizer], [self.lr_scheduler]



    @classmethod
    def params(cls, **kwargs) -> ModelParams:
        return ModelParams(**kwargs)
