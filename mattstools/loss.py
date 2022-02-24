"""
Custom loss functions and methods to calculate them
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MyBCEWithLogit(nn.Module):
    """A wrapper for the calculating BCE using logits in pytorch that makes the syntax
    consistant with pytorch's CrossEntropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss_fn(outputs.squeeze(), targets.float())


class GANLoss(nn.Module):
    """Aversarial loss for use in GANs or AAEs
    - Requires both the inputs and the model
    - This is so it can regenerate samples for nonsaturating loss
    - This is also to allow for gradient penalties
    """

    def forward(
        self, inputs: T.Tensor, outputs: T.Tensor, labels: T.Tensor, network: nn.Module
    ):
        """
        args:
            inputs: The inputs to the discriminator
            outputs: The outputs of the discriminator
            labels: The labels of the dataset (1=True, 0=False)
            network: The discriminator network
        """

        ## Calculate the the BCE discriminator losss
        disc_loss = F.binary_cross_entropy_with_logits(outputs, labels.unsqueeze(-1))

        ## Freeze gradient tracking for the discriminator
        for param in network.parameters():
            param.requires_grad = False

        ## Calculate the non saturating generator loss using only generated samples
        gen_vals, _, _ = network.forward(inputs[~labels.bool()], None, get_loss=False)
        gen_loss = F.logsigmoid(gen_vals).mean()

        ## Unfreeze the parameters of the discriminator
        for param in network.parameters():
            param.requires_grad = True

        ## TODO Add gradient penalties here
        return disc_loss + gen_loss


class ChampferLoss(nn.Module):
    """The champfer loss function for use on batched and weighted pointclouds"""

    def forward(self, o_weights, outputs, t_weights, targets):
        """Constructor method for ChampferLoss
        args:
            o_weights:  The output point cloud weights
            outputs:    The output point cloud features
            t_weights:  The target point cloud weights
            targets:    The target point cloud features
        returns:
            loss per sample (no batch reduction)
        """

        ## Calculate the distance matrix (squared) between the outputs and targets
        # dist = myGR.masked_dist_matrix(
        #     tensor_a=outputs,
        #     mask_a=o_weights > 0,
        #     tensor_b=targets,
        #     mask_b=t_weights > 0,
        #     pad_val=1e6,
        # )
        dist = 0  ## TODO Fix this

        ## Get the sum of the minimum along each axis, square, and scale by the weights
        min1 = T.min(dist, dim=-1)[0] ** 2 * (o_weights)
        min2 = T.min(dist, dim=-2)[0] ** 2 * (t_weights)

        ## Add the two metrics together (no batch reduction)
        return T.sum(min1, dim=-1) + T.sum(min2, dim=-1)


class KLD2NormLoss(nn.Module):
    """The Kullback-Leibler divergence to unit normal loss useful for VAEs"""

    def forward(self, means: T.Tensor, log_stds: T.Tensor) -> T.Tensor:
        """
        args:
            means:  The set of mean values
            log_stds:  The natural logarithm (base e) of the standard deviations
        returns:
            loss per sample (no batch reduction)
        """
        return kld_to_norm(means, log_stds)


def kld_to_norm(means: T.Tensor, log_stds: T.Tensor) -> T.Tensor:
    """Calculate the KL-divergence to a unit normal distribution"""
    return 0.5 * T.mean(means * means + (2 * log_stds).exp() - 2 * log_stds - 1)


def masked_dist_loss(
    loss_fn: nn.Module,
    pc_a: T.Tensor,
    pc_a_mask: T.BoolTensor,
    pc_b: T.Tensor,
    pc_b_mask: T.BoolTensor,
) -> T.Tensor:
    """Calculates the distribution loss between two masked pointclouds
    - This is done by using the masks as weights and thus is compatible with the geomloss package

    args:
        loss_fn: The loss function to apply, must have a forward method taking same arguments
        pc_a: The first point cloud
        pc_a_mask: The mask of the first point cloud
        pc_b: The second point cloud
        pc_b_mask: The mask of the second point cloud
    """

    ## The pytorch geomloss package always renables gradient tracking
    ## So we check current status before using geomloss
    grad_status = T.is_grad_enabled()

    ## Calculate the weights by normalising the mask for each sample
    a_weights = pc_a_mask.float() / pc_a_mask.sum(dim=-1, keepdim=True)
    b_weights = pc_b_mask.float() / pc_b_mask.sum(dim=-1, keepdim=True)

    ## Calculate the loss using these weights
    loss = loss_fn(a_weights, pc_a, b_weights, pc_b).mean()

    ## Reset the gradient tracking to the previous status
    T.set_grad_enabled(grad_status)

    return loss
