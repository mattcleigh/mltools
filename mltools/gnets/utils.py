"""Mix of utility functions masking and passing information in graphs."""

import torch as T
import torch.nn as nn

from ..torch_utils import smart_cat

# An onnx save argument which is for the pass with mask function (makes it slower)
ONNX_SAFE = False


def ctxt_from_mask(context: list | T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Concatenates and returns conditional information expanded and then sampled using
    a mask.

    The returned tensor is compressed but repeated the appropriate number
    of times for each sample. Method uses pytorch's expand function so is light on
    memory usage.

    Primarily used for repeating high level information for deep sets, graph nets, or
    transformers.

    For example, given a deep set with feature tensor [batch, nodes, features],
    a mask tensor [batch, nodes], and context information for each sample in the batch
    [batch, c_features], then this will repeat the context information the appropriate
    number of times such that new context tensor will align perfectly with tensor[mask].

    Context and mask must have the same batch dimension

    Parameters
    ----------
    context : T.Tensor
        A tensor or a list of tensors containing the sample context info
    mask : T.BoolTensor
        A mask which determines the size and sampling of the context
    """

    # Get the expanded view sizes (Use shape[0] not len as it is ONNX safe!)
    b_size = mask.shape[0]
    new_dims = (len(mask.shape) - 1) * [1]
    view_size = (b_size, *new_dims, -1)  # Must be: (b, 1, ..., 1, features)
    ex_size = (*mask.shape, -1)

    # If there is only one context tensor
    if not isinstance(context, list):
        return context.view(view_size).expand(ex_size)[mask]

    # For multiple context tensors
    all_context = []
    for ctxt in context:
        if ctxt is not None:
            all_context.append(ctxt.view(view_size).expand(ex_size)[mask])
    return smart_cat(all_context)


def pass_with_mask(
    inputs: T.Tensor,
    module: nn.Module,
    mask: T.BoolTensor | None = None,
    high_level: T.Tensor | list | None = None,
    padval: float = 0.0,
    output_dim: int | None = None,
) -> T.Tensor:
    """Pass a collection of padded tensors through a module without wasting computation
    on the padded elements.

    Ensures that: output[mask] = module(input[mask], high_level)
    Reliably for models.dense.Dense and nn.Linear

    Uses different methods depending if the global ONNE_SAFE variable is set to true
    or false, note that this does not change the output! Only the method. The onnx safe
    method is slightly slower, so it is advised to only use it during training.

    Parameters
    ----------
    inputs : T.Tensor
        The padded tensor to pass through the module
    module : nn.Module
        The pytorch model to act over the final dimension of the tensors
    mask : Optional[T.BoolTensor], optional
        Mask showing the real vs padded elements of the inputs, by default None
    high_level_context : Optional[Union[T.Tensor, List[T.Tensor]]], optional
        Added high level context information to be passed through the model
        By high level we mean that this has less dimension than the inputs.
        Example: graph global properties.
    padval : float, optional
        The value for all padded outputs, by default 0.0
    output_dim : int, optional
        The shape of the output tensor, if None will attempt to infer from module

    Returns
    -------
    T.Tensor
        Padded tensor as if you had simply called module(inputs)

    Raises
    ------
    ValueError
        Needs to know what the desired output shape of the model should be
    """

    # For generalisability, if the mask is none then we just return the normal pass
    if mask is None:
        # Without context this is a simple pass
        if high_level is None:
            return module(inputs)

        # Reshape the high level so it can be concatenated with the inputs
        dim_diff = inputs.dim() - high_level.dim()
        for d in range(dim_diff):
            high_level = high_level.unsqueeze(-2)
        high_level = high_level.expand(*inputs.shape[:-1], -1)

        # Pass through with the conditioning information
        return module(inputs, high_level)

    # Try to infer the output shape if it has not been provided
    if output_dim is None:
        if hasattr(module, "outp_dim"):
            outp_dim = module.outp_dim
        elif hasattr(module, "out_features"):
            outp_dim = module.out_features
        elif hasattr(module, "output_size"):
            outp_dim = module.output_size
        else:
            raise ValueError("Dont know how to infer the output size from the model")

    # Determine the output type depending on if mixed precision is being used
    if T.is_autocast_enabled():
        out_type = T.float16
    elif T.is_autocast_cpu_enabled():
        out_type = T.bfloat16
    else:
        out_type = inputs.dtype

    # Create an output of the correct shape on the right device using the padval
    exp_size = (*inputs.shape[:-1], outp_dim)
    outputs = T.full(exp_size, padval, device=inputs.device, dtype=out_type)

    # Onnx safe operation, but slow, use only when exporting
    if ONNX_SAFE:
        o_mask = mask.unsqueeze(-1).expand(exp_size)
        if high_level is None:
            outputs.masked_scatter_(o_mask, module(inputs[mask]))
        else:
            outputs.masked_scatter_(
                o_mask, module(inputs[mask], ctxt=ctxt_from_mask(high_level, mask))
            )

    # Inplace allocation, not onnx safe but quick
    else:
        if high_level is None:
            outputs[mask] = module(inputs[mask])
        else:
            outputs[mask] = module(inputs[mask], ctxt=ctxt_from_mask(high_level, mask))

    return outputs
