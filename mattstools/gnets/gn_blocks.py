"""
The specific modules that make up the GNblock
"""

import wandb
from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from mattstools.modules import DenseNetwork
from mattstools.torch_utils import (
    smart_cat,
    empty_0dim_like,
    apply_residual,
    ctxt_from_mask,
    aggr_via_sparse,
    masked_pool,
    pass_with_mask,
    decompress,
    sparse_from_mask,
)

from mattstools.gnets.graphs import GraphBatch


class MssgPoolBlock(nn.Module):
    """The message pooling step of the graph block

    Pools together information from sender and receiver nodes and stacks them with the
    current edge features
    """

    def __init__(self, inpt_dim: list, msg_type: str = "d"):
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c] of the gn block
            msg_type: String on how nodes combine node features to create messages
                -> "s(end)r(ecv)d(iff)"
        """
        super().__init__()

        ## Configuration checking
        if len(msg_type) > 3 or len(msg_type) == 0:
            raise ValueError(
                f"Unrecognised message passing type:{msg_type}\n"
                + "Must be some subset of chars: s, r, d"
            )

        ## Save the input and output dimensions and the message passing type
        self.inpt_dim = inpt_dim
        self.outp_dim = self.inpt_dim[1] * len(msg_type)
        self.msg_type = msg_type

    def forward(self, graph: GraphBatch) -> GraphBatch:
        """Returns all messages passed along edges in compressed form"""

        ## Expand the node contributions (No mem is allocated in expand!)
        ex_size = (*graph.adjmat.shape, -1)
        if "s" in self.msg_type or "d" in self.msg_type:
            send_info = graph.nodes.unsqueeze(-2).expand(ex_size)[graph.adjmat]
        if "r" in self.msg_type or "d" in self.msg_type:
            recv_info = graph.nodes.unsqueeze(-3).expand(ex_size)[graph.adjmat]
        if "d" in self.msg_type:
            diff_info = send_info - recv_info

        ## Start the messages as empty and add each tensor as needed
        pooled_mssgs = []
        if "s" in self.msg_type:
            pooled_mssgs.append(send_info)
        if "r" in self.msg_type:
            pooled_mssgs.append(recv_info)
        if "d" in self.msg_type:
            pooled_mssgs.append(diff_info)

        ## Return all messages
        return smart_cat(pooled_mssgs, -1)


class EdgeBlock(nn.Module):
    """The edge updating step of a graph network block

    Combines the edge information with the pooled messages from send-recv pairs

    Can also contain a dense network to update the edges which take as inputs the:
    - Pooled messages
    - Current edge features (if present)
    - Global features (if present)
    - Conditional features (if present)

    Block can update the edges through residual connection based on rsdl_type
    - Can only apply residual update if previous graph edges existed
    - Residual connection can either be concatenation or additive
        - If addititve and dimensions change it will apply a linear resizing layer
          to the old features
    """

    def __init__(
        self,
        inpt_dim: list,
        pooled_mssg_dim: int,
        use_net: bool = False,
        rsdl_type: str = "none",
        net_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c] of the gn block
            pooled_mssg_dim: The size of the pooled node contributions per edge
            use_net: If a dense network will be used to update the edges
            rsdl_type: The type of residual connection: none, add, or cat
            net_kwargs: The dictionary of kwargs for the feature dense network
        """
        super().__init__()

        ## Dict default kwargs
        net_kwargs = net_kwargs or {}

        ## Save the input dimension of the graph object
        self.inpt_dim = inpt_dim
        self.pooled_mssg_dim = pooled_mssg_dim

        ## Save the options on how the edge features are combined/updated/pooled
        self.use_net = use_net
        self.rsdl_type = rsdl_type

        ## Residual updates require both the option and existing input edges
        self.do_rsdl = rsdl_type != "none" and inpt_dim[0]

        ## Calculate the sizes of the messages being passed
        self.e_inpt_dim, self.e_outp_dim, self.ctxt_dim = self._get_dims()

        ## If using a dense network to update messsages
        if self.use_net:
            self.dense_net = DenseNetwork(
                inpt_dim=self.e_inpt_dim, ctxt_dim=self.ctxt_dim, **net_kwargs
            )
            self.e_outp_dim += self.dense_net.outp_dim

        ## Include a resizing layer if rsdl adding and change of dim
        self.do_rsdl_lin = (
            self.do_rsdl and rsdl_type == "add" and (inpt_dim[0] != self.e_outp_dim)
        )
        if self.do_rsdl_lin:
            self.rsdl_lin = nn.Linear(inpt_dim[0], self.e_outp_dim)

    def _get_dims(self) -> Tuple[int, int]:
        """Calculates the edge input and output and context sizes based on config"""

        ## Without a network the messages are made up of pooled message info
        e_inpt_dim = self.pooled_mssg_dim
        e_outp_dim = e_inpt_dim

        ## With a network, input can include more info and output is set to 0 for now
        if self.use_net:
            e_inpt_dim += self.inpt_dim[0]
            e_outp_dim = 0  ## Increased by net output and residual connection

        ## If using shortcut residual connections, add the previous edge dimensions
        if self.do_rsdl and self.rsdl_type == "cat":
            e_outp_dim += self.inpt_dim[0]

        ## Context dimensions come from global and conditional information
        ctxt_dim = self.inpt_dim[2] + self.inpt_dim[3]

        return e_inpt_dim, e_outp_dim, ctxt_dim

    def forward(self, graph: GraphBatch, pooled_mssgs: T.Tensor) -> T.Tensor:
        """
        args:
            graph: The batched graph object to pass through the block
            pooled_mssgs: The combined info from sender and receiver nodes
        returns:
            pooled_mssgs: The new edge features for the graph
        """

        ## Get inputs and pass through the network (edges are already compressed)
        if self.use_net:
            pooled_mssgs = self.dense_net(
                inputs=smart_cat([pooled_mssgs, graph.edges]),
                ctxt=ctxt_from_mask([graph.globs, graph.cndts], graph.adjmat),
            )

        ## Add the old edges if residual updates (through a resizing layer if needed)
        if self.do_rsdl:
            rsdl = self.rsdl_lin(graph.edges) if self.do_rsdl_lin else graph.edges
            pooled_mssgs = apply_residual(self.rsdl_type, rsdl, pooled_mssgs)

        return pooled_mssgs


class EdgePoolBlock(nn.Module):
    """Applies pooling to the compressed edges in a graph across the receiver dimension.

    Can apply attention pooling, but then additional information is required in the
    forward pass to calculate the edge weights.
    """

    def __init__(
        self,
        inpt_dim: int,
        new_edge_dim: int,
        pooled_mssg_dim: int = 0,
        pool_type: str = "sum",
        attn_type: str = "sum",
        net_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c] of the gn block
            new_edge_dim: The size of the new edge features, this tensor is pooled
            pooled_mssg_dim: The size of the pooled node contributions per edge
            pool_type: The pooling method to apply (mean, sum, attn)
            attn_type: The attention pooling method to apply (mean, sum, raw)
            net_kwargs: The dictionary of kwargs for the attention dense network
        """
        super().__init__()

        ## Dict default kwargs
        net_kwargs = net_kwargs or {}

        ## Store the attributes for basic pooling
        self.inpt_dim = inpt_dim
        self.new_edge_dim = new_edge_dim
        self.pool_type = pool_type
        self.outp_dim = new_edge_dim

        ## Specifically for attention pooling
        if self.pool_type == "attn":


            ## Attributes needed for attention pooling
            self.pooled_mssg_dim = pooled_mssg_dim
            self.attn_type = attn_type

            ## For the attention network the default output must be set to 1
            if "outp_dim" not in net_kwargs:
                net_kwargs["outp_dim"] = 1

            attn_dim, ctxt_dim = self._get_dims()
            self.attn_net = DenseNetwork(attn_dim, ctxt_dim=ctxt_dim, **net_kwargs)

            ## Check that the dimension of each head makes internal sense
            self.head_dim = self.outp_dim // self.attn_net.outp_dim
            if self.head_dim * self.attn_net.outp_dim != self.outp_dim:
                raise ValueError("Output dimension must be divisible by # of heads!")


    def _get_dims(self) -> int:
        """Calculates the attention input and context sizes based on config"""

        ## Attention inputs are the pooled messages and the old edges
        attn_dim = self.pooled_mssg_dim + self.inpt_dim[0]

        ## Context dimensions come from global and conditional information
        ctxt_dim = self.inpt_dim[2] + self.inpt_dim[3]

        return attn_dim, ctxt_dim

    def forward(
        self, new_edges: T.Tensor, graph: GraphBatch, pooled_mssgs: T.Tensor = None
    ):
        """
        args:
            edges: The tensor to pool over
            graph: The graph with old edges and adjmat, used for pooling and attn
        kwargs:
            pooled_mssgs: Send and receive info to add to the attn network
        """

        ## For attention pooling
        if self.pool_type == "attn":

            ## Build the input tensors for the attention network
            attn_outs = self.attn_net(
                inputs=smart_cat([pooled_mssgs, graph.edges]),
                ctxt=ctxt_from_mask([graph.globs, graph.cndts], graph.adjmat),
            )

            ## Apply the softplus or softmax for weighted sum or mean
            if self.attn_type == "sum":
                attn_outs = F.softplus(attn_outs)
            if self.attn_type == "mean":
                attn_outs = aggr_via_sparse(
                    attn_outs, graph.adjmat, reduction="softmax", dim=1
                )

            ## Broadcast the attention weights with the features and sum
            attn_outs = attn_outs.unsqueeze(-1).expand(-1, -1, self.head_dim).flatten(1)
            new_edges = new_edges * attn_outs
            new_edges = aggr_via_sparse(new_edges, graph.adjmat, reduction="sum", dim=1)

        ## For normal pooling methods
        else:
            new_edges = aggr_via_sparse(
                new_edges, graph.adjmat, reduction=self.pool_type, dim=1
            )

        ## Since the pooled edges are per receiver node we must undo the compression
        return decompress(new_edges, graph.adjmat.any(1))


class NodeBlock(nn.Module):
    """The node undating step of a graph network block

    Combines the node information with the pooled edge information

    Can also contain a dense network to update the nodes which take as inputs the:
    - Pooled incomming edge information
    - Current node features
    - Global features (if present)
    - Conditional features (if present)

    Block can update the nodes through residual connection based on rsdl_type
    - Residual connection can either be concatenation or additive
        - If addititve and dimensions change it will apply a linear resizing layer
          to the old features

    This block also allows LOCKING certain nodes in such that they do not get updated
    even if they took part of the sending of information
    """

    def __init__(
        self,
        inpt_dim: list,
        pooled_edge_dim: int,
        use_net: bool = False,
        rsdl_type: str = "none",
        net_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c] of the gn block
            pooled_edge_dim: The size of the pooled edge tensors per receiver node
            use_net: If a dense network will be used to update the nodes
            rsdl_type: The type of residual connection: none, add, or cat
            net_kwargs: The dictionary of kwargs for the dense mlp
        """
        super().__init__()

        ## Dict default kwargs
        net_kwargs = net_kwargs or {}

        ## Save the input dimension of the graph object and the pooled edges
        self.inpt_dim = inpt_dim
        self.pooled_edge_dim = pooled_edge_dim

        ## Save the configuration for how the nodes are to be updated
        self.use_net = use_net
        self.rsdl_type = rsdl_type
        self.do_rsdl = rsdl_type != "none"

        ## Calculate the size node input/output information
        self.n_inpt_dim, self.n_outp_dim, self.ctxt_dim = self._get_dims()

        ## If using a dense netwok to update the nodes
        if self.use_net:
            self.dense_net = DenseNetwork(
                inpt_dim=self.n_inpt_dim, ctxt_dim=self.ctxt_dim, **net_kwargs
            )
            self.n_outp_dim += self.dense_net.outp_dim

        ## Include a resizing layer if rsdl adding and change of dim
        self.do_rsdl_lin = rsdl_type == "add" and (inpt_dim[1] != self.n_outp_dim)
        if self.do_rsdl_lin:
            self.rsdl_lin = nn.Linear(inpt_dim[1], self.n_outp_dim)

    def _get_dims(self) -> Tuple[int, int]:
        """Calculates the node input and output and context sizes sizes based on config"""

        ## Without a network the node inputs are made up of just the pooled info
        n_inpt_dim = self.pooled_edge_dim
        n_outp_dim = n_inpt_dim

        ## With a network, input can include more info and output is set to 0 for now
        if self.use_net:
            n_inpt_dim += self.inpt_dim[1]
            n_outp_dim = 0  ## Increased by net output and residual connection

        ## If using shortcut residual connections, add the previous node dimensions
        if self.do_rsdl and self.rsdl_type == "cat":
            n_outp_dim += self.inpt_dim[1]

        ## Context dimensions come from global and conditional information
        ctxt_dim = self.inpt_dim[2] + self.inpt_dim[3]

        return n_inpt_dim, n_outp_dim, ctxt_dim

    def forward(
        self,
        graph: GraphBatch,
        pooled_edges: T.Tensor,
        locked_nodes: T.Tensor = None,
    ) -> Tuple[T.Tensor, T.Tensor]:
        """
        args:
            graph: The batched graph object to be passed through the block
            pooled_edges: The pooled information for each receiver node
        kwargs:
            locked_nodes: A batched mask for the nodes nodes which may NOT be
                          updated by this block
        returns:
            The new node features for the graph
        """

        ## Check if the block allows for locked nodes and create the masks
        if locked_nodes is not None:
            if locked_nodes.any() and (self.inpt_dim[1] != self.n_outp_dim):
                raise ValueError(
                    "Can not lock any nodes as the dimension changes: "
                    + f"({self.inpt_dim[1]}->{self.n_outp_dim})"
                )

            ## Create a mask of which real nodes may be updated by this block
            lock_mask = graph.mask * locked_nodes
            free_mask = graph.mask * ~locked_nodes

        else:
            lock_mask = T.zeros_like(graph.mask).bool()
            free_mask = graph.mask

        ## Get inputs and pass through the network
        if self.use_net:
            pooled_edges = pass_with_mask(
                inputs=T.cat([pooled_edges, graph.nodes], dim=-1),  ## No smart_cat
                module=self.dense_net,
                mask=free_mask,
                context=[graph.globs, graph.cndts],
            )

        ## Add the old nodes if residual updates (through a resizing layer if needed)
        if self.do_rsdl:
            if self.do_rsdl_lin:
                rsdl = pass_with_mask(graph.nodes, self.rsdl_lin, free_mask)
            else:
                rsdl = graph.nodes
            pooled_edges = apply_residual(self.rsdl_type, rsdl, pooled_edges)

        ## Add the original-locked nodes back in (pass_with_mask made them all 0 padded)
        if lock_mask.any():
            pooled_edges[lock_mask] = graph.nodes[lock_mask]

        return pooled_edges


class NodePoolBlock(nn.Module):
    """Applies pooling to the padded nodes in a graph.

    Can apply attention pooling, but then additional information is required in the
    forward pass to calculate the node weights.
    """

    def __init__(
        self,
        inpt_dim: int,
        new_node_dim: int,
        pooled_edge_dim: int = 0,
        pool_type: str = "sum",
        attn_type: str = "sum",
        net_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c] of the gn block
            new_node_dim: The size of the new edge features, this tensor is pooled
            pooled_edge_dim: The size of the pooled edge features, for the attn net
            pool_type: The pooling method to apply (mean, sum, attn)
            attn_type: The attention pooling method to apply (mean, sum, raw)
            net_kwargs: The dictionary of kwargs for the attention dense network
        """
        super().__init__()

        ## Dict default kwargs
        net_kwargs = net_kwargs or {}

        ## Store the attributes for basic pooling
        self.inpt_dim = inpt_dim
        self.new_edge_dim = new_node_dim
        self.pool_type = pool_type
        self.outp_dim = new_node_dim

        ## Specifically for attention pooling
        if self.pool_type == "attn":

            ## Attributes needed for attention pooling
            self.pooled_edge_dim = pooled_edge_dim
            self.attn_type = attn_type

            ## For the attention network the default output must be set to 1
            if "outp_dim" not in net_kwargs:
                net_kwargs["outp_dim"] = 1

            attn_dim, ctxt_dim = self._get_dims()
            self.attn_net = DenseNetwork(attn_dim, ctxt_dim=ctxt_dim, **net_kwargs)

            ## Check that the dimension of each head makes internal sense
            self.head_dim = self.outp_dim // self.attn_net.outp_dim
            if self.head_dim * self.attn_net.outp_dim != self.outp_dim:
                raise ValueError("Output dimension must be divisible by # of heads!")

    def _get_dims(self) -> int:
        """Calculates the attention input and context sizes based on config"""

        ## Attention inputs are the pooled edges and the old nodes
        attn_dim = self.pooled_edge_dim + self.inpt_dim[1]

        ## Context dimensions come from global and conditional information
        ctxt_dim = self.inpt_dim[2] + self.inpt_dim[3]

        return attn_dim, ctxt_dim

    def forward(
        self, new_nodes: T.Tensor, graph: GraphBatch, pooled_edges: T.Tensor = None
    ):
        """
        args:
            new_nodes: The tensor to pool over
            graph: The graph with old edges and adjmat, used for pooling and attn
        kwargs:
            pooled_edges: Send and receive info to add to the attn network
        """

        ## For attention pooling
        if self.pool_type == "attn":

            ## Build the input tensors for the attention network
            attn_outs = pass_with_mask(
                inputs=smart_cat([pooled_edges, graph.nodes]),
                module=self.attn_net,
                mask=graph.mask,
                context=[graph.globs, graph.cndts],
                padval=0 if self.attn_type == "raw" else -T.inf,
            )

            ## Apply either a softmax for weighted mean or softplus for weighted sum
            if self.attn_type == "mean":
                attn_outs = F.softmax(attn_outs, dim=-2)
            elif self.attn_type == "sum":
                attn_outs = F.softplus(attn_outs)

            ## Broadcast the attention to get the multiple poolings and sum
            attn_outs = attn_outs.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).flatten(2)
            new_nodes = (new_nodes * attn_outs).sum(dim=-2)

        ## For normal pooling methods
        else:
            new_nodes = masked_pool(self.pool_type, new_nodes, graph.mask)

        return new_nodes


class GlobBlock(nn.Module):
    """The global vector updating step in a graph network block

    Combines the global information with the pooled node information

    Can also contain a dense network to update the globals which take as inputs the:
    - Pooled node information
    - Current global features (if present)
    - Current conditional features (if present)

    Block can update the globals through residual connection based on rsdl_type
    - Residual connection can either be concatenation or additive
        - If addititve and dimensions change it will apply a linear resizing layer
          to the old features
    """

    def __init__(
        self,
        inpt_dim: list,
        pooled_node_dim: list,
        use_net: bool = False,
        rsdl_type: str = "none",
        net_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c] of the gn block
            pooled_node_dim: The size of the pooled node tensors across the graph
            use_net: If a dense network will be used to update the globals
            rsdl_type: The type of residual connection: none, add, or cat
            net_kwargs: The dictionary of kwargs for the dense mlp
        """
        super().__init__()

        ## Dict default kwargs
        net_kwargs = net_kwargs or {}

        ## Save the input and output dimensions of the entire graph block
        self.inpt_dim = inpt_dim
        self.pooled_node_dim = pooled_node_dim

        ## Save the configuration for how the globals are to be updated
        self.use_net = use_net
        self.rsdl_type = rsdl_type

        ## Residual updates require both the option and existing input globals
        self.do_rsdl = rsdl_type != "none" and inpt_dim[2]

        ## Calculate the size global input/output information
        self.g_inpt_dim, self.g_outp_dim, self.ctxt_dim = self._get_dims()

        ## If using a dense netwok to update the globals
        if self.use_net:
            self.dense_net = DenseNetwork(
                inpt_dim=self.g_inpt_dim, ctxt_dim=self.ctxt_dim, **net_kwargs
            )
            self.g_outp_dim += self.dense_net.outp_dim

        ## Include a resizing layer if rsdl adding and change of dim
        self.do_rsdl_lin = (
            self.do_rsdl and rsdl_type == "add" and (inpt_dim[2] != self.g_outp_dim)
        )
        if self.do_rsdl_lin:
            self.rsdl_lin = nn.Linear(inpt_dim[2], self.g_outp_dim)

    def _get_dims(self) -> Tuple[int, int]:
        """Calculates the global input and output and context sizes based on config"""

        ## Without a network the global inputs are made up of just pooled info
        g_inpt_dim = self.pooled_node_dim
        g_outp_dim = g_inpt_dim

        ## With a network, input can include more info and output is set to 0 for now
        if self.use_net:
            g_inpt_dim += self.inpt_dim[2]
            g_outp_dim = 0  ## Increased by net output and residual connection

        ## If using the shortcut method for residual connections add the previous global dimensions
        if self.do_rsdl and self.rsdl_type == "cat":
            g_outp_dim += self.inpt_dim[2]

        ## Context dimensions come frrom and conditional information only
        ctxt_dim = self.inpt_dim[3]

        return g_inpt_dim, g_outp_dim, ctxt_dim

    def forward(self, graph: GraphBatch, pooled_nodes: T.Tensor) -> T.Tensor:
        """
        args:
            graph: The batched graph object to be passed through the block
            pooled_nodes: The pooled information from all nodes in the graph

        returns:
            The new global tensor for the graph
        """

        ## If using a network the pooled info is combined with others and processed
        if self.use_net:
            pooled_nodes = self.dense_net.forward(
                inputs=smart_cat([pooled_nodes, graph.globs], dim=-1), ctxt=graph.cndts
            )

        ## Add the old globs if using residual updates (through a resizing layer if needed)
        if self.do_rsdl:
            rsdl = self.rsdl_lin(graph.globs) if self.do_rsdl_lin else graph.globs
            pooled_nodes = apply_residual(self.rsdl_type, rsdl, pooled_nodes)

        return pooled_nodes


class GNBlock(nn.Module):
    """A message passing Graph Network Block

    Comprised of an several blocks which act in sequence
    - MssgPoolBlock: Combines current node features to get edge contributions
    - EdgeBlock: Updates graph.edges using pooled messages and graph info
    - EdgePoolBlock: Pools the edges over the sender node dimension (can use attention)
    - NodeBlock: Updates graph.nodes using pooled edges and graph info
    - NodePoolBlock: Pools the nodes over the whole graph (can use attention)
    - GlobBlock: Updates graph.globs using pooled nodes and graph info
        - These last two only exist if the argument do_glob is true
    """

    def __init__(
        self,
        inpt_dim: list,
        pers_edges: bool = True,
        do_glob: bool = True,
        norm_new_features: bool = False,
        mgpl_blk_kwargs: dict = None,
        edge_blk_kwargs: dict = None,
        egpl_blk_kwargs: dict = None,
        node_blk_kwargs: dict = None,
        ndpl_blk_kwargs: dict = None,
        glob_blk_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c]
            pers_edges: If the GN block should save the edges of the graph
            do_glob:  If there will be a global output tensor in this block
            mgpl_blk_kwargs: kwargs for the mssg pooling block
            edge_blk_kwargs: kwargs for the edge block
            egpl_blk_kwargs: kwargs for the edge pooling block
            node_blk_kwargs: kwargs for the node block
            ndpl_blk_kwargs: kwargs for the node pooling block
            glob_blk_kwargs: kwargs for the glob block
        """
        super().__init__()

        ## Dict default kwargs
        mgpl_blk_kwargs = mgpl_blk_kwargs or {}
        edge_blk_kwargs = edge_blk_kwargs or {}
        egpl_blk_kwargs = egpl_blk_kwargs or {}
        node_blk_kwargs = node_blk_kwargs or {}
        ndpl_blk_kwargs = ndpl_blk_kwargs or {}
        glob_blk_kwargs = glob_blk_kwargs or {}

        ## Store the input dimension and other attributes
        self.inpt_dim = inpt_dim
        self.pers_edges = pers_edges
        self.do_glob = do_glob
        self.norm_new_features = norm_new_features

        ## Initialise each of the blocks that make up this module
        self.mspl_block = MssgPoolBlock(inpt_dim, **mgpl_blk_kwargs)
        self.edge_block = EdgeBlock(
            inpt_dim, self.mspl_block.outp_dim, **edge_blk_kwargs
        )
        self.egpl_block = EdgePoolBlock(
            inpt_dim,
            self.edge_block.e_outp_dim,
            self.mspl_block.outp_dim,
            **egpl_blk_kwargs,
        )
        self.node_block = NodeBlock(
            inpt_dim, self.egpl_block.outp_dim, **node_blk_kwargs
        )

        ## For the global updates
        if do_glob:
            self.ndpl_block = NodePoolBlock(
                inpt_dim,
                self.node_block.n_outp_dim,
                self.egpl_block.outp_dim,
                **ndpl_blk_kwargs,
            )
            self.glob_block = GlobBlock(
                inpt_dim, self.ndpl_block.outp_dim, **glob_blk_kwargs
            )

        ## Calculate the output dimension of the final graph
        self.outp_dim = [
            self.edge_block.e_outp_dim if pers_edges else 0,
            self.node_block.n_outp_dim,
            self.glob_block.g_outp_dim if do_glob else 0,
            inpt_dim[-1],
        ]

        if norm_new_features:
            self.edge_norm = nn.LayerNorm(self.edge_block.e_outp_dim)
            self.node_norm = nn.LayerNorm(self.node_block.n_outp_dim)
            self.glob_norm = nn.LayerNorm(self.glob_block.g_outp_dim)

        ## Check that at least one of the blocks has a network
        if not any(
            [
                self.edge_block.use_net,
                self.node_block.use_net,
                do_glob and self.glob_block.use_net,
            ]
        ):
            raise ValueError(
                "None of the submodules in this GN block have any learnable parameters!"
            )

    def forward(
        self, graph: GraphBatch, locked_nodes: T.BoolTensor = None
    ) -> GraphBatch:
        """Return an updated graph with the same structure, but new features"""

        ## Pool the information from each send-recv nodes
        pooled_mssgs = self.mspl_block(graph)

        ## Update the edges of the graph and pool for each receiver node
        new_edges = self.edge_block(graph, pooled_mssgs)
        if self.norm_new_features:
           new_edges = self.edge_norm(new_edges)
        pooled_edges = self.egpl_block(new_edges, graph, pooled_mssgs)
        graph.edges = new_edges  ## Delay update due to attn needing old and new
        del pooled_mssgs

        ## Update the nodes of the graph
        new_nodes = self.node_block(graph, pooled_edges, locked_nodes)
        if self.norm_new_features:
           new_nodes = self.node_norm(new_nodes)

        ## If there are global outputs, pool the nodes and update globs
        if self.do_glob:
            pooled_nodes = self.ndpl_block(new_nodes, graph, pooled_edges)
            graph.nodes = new_nodes
            del pooled_edges
            graph.globs = self.glob_block(graph, pooled_nodes)
        else:
            del pooled_edges
            graph.nodes = new_nodes
            graph.globs = empty_0dim_like(graph.globs)

        ## If we are not doing persistant edges, ensure they are empty
        if not self.pers_edges:
            graph.edges = empty_0dim_like(graph.edges)

        ## Return the graph
        return graph

    def __repr__(self):
        """A way to print the block config on one line for quick review

        We DONT overload a repr method for each block as the default print method
        is clean and verbose enough when looking at them individually!!!
        """
        string = str(self.inpt_dim)

        ## Message Pooling
        string += f"->MssPool({self.mspl_block.msg_type})"

        ## Edge updating
        if self.edge_block.use_net:
            string += f"->EdgeNet[{self.edge_block.dense_net.one_line_string()}]"
        if self.edge_block.do_rsdl:
            string += f"({self.edge_block.rsdl_type})"

        ## Edge pooling
        string += f"->{self.egpl_block.pool_type}"
        if self.egpl_block.pool_type == "attn":
            string += f"({self.egpl_block.attn_net.outp_dim})"

        ## Node updating
        if self.node_block.use_net:
            string += f"->NodeNet[{self.node_block.dense_net.one_line_string()}]"
        if self.node_block.do_rsdl:
            string += f"({self.node_block.rsdl_type})"

        if self.do_glob:

            ## Node pooling
            string += f"->{self.ndpl_block.pool_type}"
            if self.ndpl_block.pool_type == "attn":
                string += f"({self.ndpl_block.attn_net.outp_dim})"

            ## Global updating
            if self.glob_block.use_net:
                string += f"->GlobNet[{self.glob_block.dense_net.one_line_string()}]"
            if self.glob_block.do_rsdl:
                string += f"({self.glob_block.rsdl_type})"

        string += f"->{self.outp_dim}"
        return string
