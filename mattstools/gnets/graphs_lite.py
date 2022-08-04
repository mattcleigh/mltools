"""
Defines the lightweight and streamlined graph object and operations
"""

from typing import Iterable, Union

import torch as T
from torch.utils.data._utils.collate import default_collate

from mattstools.torch_utils import pass_with_mask, sel_device
from mattstools.modules import DenseNetwork

import torch as T
import torch.nn as nn
import torch.nn.functional as F



class GraphLite:
    """The base class for the custom graph lite object
    - Without compressed edges
    - Without conditional features

    A collection of 6 tensors:
    - 4 describe the attributes of the edges, nodes, globals
    - 2 describe the structure, providing masks for the edges (adjmat) and nodes (mask)
    """

    def __init__(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        globs: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
        dev: str = "cpu",
    ) -> None:
        """
        args:
            edges: Compressed edge features (num_edges x Ef)
            nodes: Node features (N x Nf)
            globs: Global features (Gf)
            adjmat: The adjacency matrix, a mask for the edge features (N x N)
            mask: The node mask (N)
        kwargs:
            dev: A string indicating the device on which to store the tensors
        """

        ## Save the device where the graph will be stored
        self.device = T.device(dev)

        ## Save each of the component tensors onto the correct device
        self.edges = edges.to(self.device)
        self.nodes = nodes.to(self.device)
        self.globs = globs.to(self.device)
        self.adjmat = adjmat.to(self.device)
        self.mask = mask.to(self.device)

    @property
    def dtype(self):
        """Inherits the dtype from the node tensor"""
        return self.nodes.dtype

    def dim(self):
        """Return the dimensions of the graph"""
        return [self.edges.shape[-1], self.nodes.shape[-1], self.globs.shape[-1]]

    def __len__(self):
        """Return the masking length of the graph"""
        return len(self.mask)

    def to(self, dev: str):
        """Move the graph to a selected device"""
        return Graph(
            self.edges, self.nodes, self.globs, self.adjmat, self.mask, dev=dev
        )


def gcoll_lite(batch: Iterable) -> Union[GraphLite, tuple]:
    """A custom collation function which allows us to batch together multiple graphs
    - Wraps the pytorch default collation function to allow for all the memory tricks
    - Looks at the first element of the batch for instructions on what to do

    args:
        batch: An iterable list/tuple containing graphs or other iterables of graphs
    returns:
        Batched graph object
    """

    ## Get the first element object type
    elem = batch[0]

    ## If we are dealing with a graph object then we apply the customised collation
    if isinstance(elem, GraphLite):
        edges = default_collate([g.edges for g in batch])
        nodes = default_collate([g.nodes for g in batch])
        globs = default_collate([g.globs for g in batch])
        adjmat = default_collate([g.adjmat for g in batch])
        mask = default_collate([g.mask for g in batch])

        return GraphLite(edges, nodes, globs, adjmat, mask, dev=mask.device)

    ## If we have a tuple, we must run the function for each object
    if isinstance(elem, tuple):
        return tuple(gcoll_lite(samples) for samples in zip(*batch))

    ## If we are dealing with any other type we must run the normal collation function
    return default_collate(batch)


class EdgeBlockLite(nn.Module):
    """The edge updating and pooling step of a graph network block"""

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        n_heads: int = 1,
        feat_kwargs: dict = None,
        attn_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            outp_dim: The dimensions of the output graph [e,n,g] of the gn block
            ctxt_dim: The size of the contextual information
            n_heads: Number of attention heads
            feat_kwargs: The dictionary of kwargs for the feature dense network
            attn_kwargs: The dictionary of kwargs for the attention dense network
        """
        super().__init__()

        ## Number of attention heads must divide dimension
        assert inpt_dim[0] % n_heads == 0
        self.head_dim = inpt_dim[0] // n_heads

        ## Dict default kwargs
        feat_kwargs = feat_kwargs or {}
        attn_kwargs = attn_kwargs or {}

        ## Useful dimensions
        edge_inpt_dim = inpt_dim[0] + 2 * inpt_dim[2]
        ctxt_inpt_dim = inpt_dim[2] + ctxt_dim
        self.same_size = inpt_dim[0] == outp_dim[0]

        ## The dense network to update messsages
        self.feat_net = DenseNetwork(
            inpt_dim=edge_inpt_dim,
            outp_dim=outp_dim[0],
            ctxt_dim=ctxt_inpt_dim,
            **feat_kwargs,
        )

        ## The attention network for pooling
        self.attn_net = DenseNetwork(
            inpt_dim=edge_inpt_dim,
            outp_dim=n_heads,
            ctxt_dim=ctxt_inpt_dim,
            **attn_kwargs,
        )

        ## The pre-post layernormalisation layer
        self.pre_ln = nn.LayerNorm(edge_inpt_dim)
        self.post_ln = nn.LayerNorm(outp_dim[0])

    def forward(self, graph: GraphLite, ctxt: T.Tensor = None) -> T.Tensor:
        """
        args:
            graph: The batched graph object
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_edges: The new edge features of the graph
        """

        ## Create the inputs for the edge networks
        ex_size = (*graph.adjmat.shape, -1)
        edges = T.cat(
            [
                graph.nodes.unsqueeze(-2).expand(ex_size),
                graph.nodes.unsqueeze(-3).expand(ex_size),
                graph.edges,
            ],
            dim=-1,
        )
        edges = pass_with_mask(edges, self.pre_ln, graph.adjmat)

        ## Pass them through the feature network
        new_edges = pass_with_mask(
            edges, self.feat_net, graph.adjmat, context=[graph.globs, ctxt]
        )
        new_edges = pass_with_mask(new_edges, self.post_ln, graph.adjmat)
        if self.same_size:
            new_edges = new_edges + graph.edges

        ## Pass them through the attention network
        edge_weights = F.softmax(
            pass_with_mask(
                edges, self.attn_net, graph.adjmat, context=[graph.globs, ctxt]
            ),
            dim=-3,
        )

        ## Broadcast the attention to get the multiple poolings and sum
        edge_weights = edge_weights.expand(new_edges.shape)
        edge_weights = (new_edges * edge_weights).sum(dim=-3)

        return new_edges, edge_weights


class NodeBlockLite(nn.Module):
    """The node updating and pooling step of a graph network block"""

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        n_heads: int = 1,
        feat_kwargs: dict = None,
        attn_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            outp_dim: The dimensions of the output graph [e,n,g] of the gn block
            ctxt_dim: The size of the contextual information
            n_heads: Number of attention heads
            feat_kwargs: The dictionary of kwargs for the feature dense network
            attn_kwargs: The dictionary of kwargs for the attention dense network
        """
        super().__init__()

        ## Number of attention heads must divide dimension
        assert inpt_dim[0] % n_heads == 0
        self.head_dim = inpt_dim[0] // n_heads

        ## Dict default kwargs
        feat_kwargs = feat_kwargs or {}
        attn_kwargs = attn_kwargs or {}

        ## Useful dimensions
        node_inpt_dim = inpt_dim[0] + inpt_dim[1]
        ctxt_inpt_dim = inpt_dim[2] + ctxt_dim
        self.same_size = inpt_dim[1] == outp_dim[1]

        ## The dense network to update messsages
        self.feat_net = DenseNetwork(
            inpt_dim=node_inpt_dim,
            outp_dim=outp_dim[1],
            ctxt_dim=ctxt_inpt_dim,
            **feat_kwargs,
        )

        ## The attention network for pooling
        self.attn_net = DenseNetwork(
            inpt_dim=node_inpt_dim,
            outp_dim=n_heads,
            ctxt_dim=ctxt_inpt_dim,
            **attn_kwargs,
        )

        ## The pre-post layernormalisation layer
        self.pre_ln = nn.LayerNorm(node_inpt_dim)
        self.post_ln = nn.LayerNorm(outp_dim[1])

    def forward(
        self, graph: GraphLite, pooled_edges: T.Tensor, ctxt: T.Tensor = None
    ) -> T.Tensor:
        """
        args:
            graph: The batched graph object
            pooled_edges: The pooled information per receiver node
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_nodes: The new node features of the graph
        """

        ## Create the inputs for the node networks
        nodes = T.cat([graph.nodes, pooled_edges], dim=-1)
        nodes = pass_with_mask(nodes, self.pre_ln, graph.mask)

        ## Pass them through the feature network
        new_nodes = pass_with_mask(
            nodes, self.feat_net, graph.mask, context=[graph.globs, ctxt]
        )
        new_nodes = pass_with_mask(new_nodes, self.pre_ln, graph.mask)
        if self.same_size:
            new_nodes = new_nodes + graph.nodes

        ## Pass them through the attention network
        node_weights = F.softmax(
            pass_with_mask(
                nodes, self.attn_net, graph.mask, context=[graph.globs, ctxt]
            ),
            dim=-2,
        )

        ## Broadcast the attention to get the multiple poolings and sum
        node_weights = node_weights.expand(new_nodes.shape)
        node_weights = (new_nodes * node_weights).sum(dim=-2)

        return new_nodes, node_weights


class GlobBlockLite(nn.Module):
    """The global updating step of a graph network block"""

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        feat_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            outp_dim: The dimensions of the output graph [e,n,g] of the gn block
            ctxt_dim: The size of the contextual information
            feat_kwargs: The dictionary of kwargs for the feature dense network
        """
        super().__init__()

        ## Dict default kwargs
        feat_kwargs = feat_kwargs or {}

        ## Useful dimensions
        glob_inpt_dim = inpt_dim[1] + inpt_dim[2]

        ## The dense network to update messsages
        self.feat_net = DenseNetwork(
            inpt_dim=glob_inpt_dim,
            outp_dim=outp_dim[2],
            ctxt_dim=ctxt_dim,
            **feat_kwargs,
        )

        ## The pre-post layernormalisation layer
        self.pre_ln = nn.LayerNorm(glob_inpt_dim)
        self.post_ln = nn.LayerNorm(outp_dim[2])

    def forward(
        self, graph: GraphLite, pooled_nodes: T.Tensor, ctxt: T.Tensor = None
    ) -> T.Tensor:
        """
        args:
            graph: The batched graph object
            pooled_nodes: The pooled information across the graph
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_globs: The new global features of the graph
        """
        return (
            self.post_ln(
                self.feat_net(
                    self.pre_ln(T.cat[graph.globs, pooled_nodes], dim=-1), ctxt=ctxt
                )
            )
            + graph.globs
        )


class GNBlockLite(nn.Module):
    """A message passing Graph Network Block
    - Lite implies that the coding and variability between models is minimal
    - Does not use compressed edges
    """

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        edge_block_kwargs: dict = None,
        node_block_kwargs: dict = None,
        glob_block_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g]
            outp_dim: The dimensions of the output graph [e,n,g]
            has_edges: If the input graph already has edges
            has_globs: If the input graph already has globals
        kwargs:
            edge_block_kwargs: kwargs for the edge block
            node_block_kwargs: kwargs for the node block
            glob_block_kwargs: kwargs for the glob block
        """
        super().__init__()

        ## Dict default kwargs
        edge_block_kwargs = edge_block_kwargs or {}
        node_block_kwargs = node_block_kwargs or {}
        glob_block_kwargs = glob_block_kwargs or {}

        ## Store the input dimensions
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        ## Define the update blocks
        self.edge_block = EdgeBlockLite(
            inpt_dim, outp_dim, ctxt_dim, **edge_block_kwargs
        )
        self.node_block = NodeBlockLite(
            inpt_dim, outp_dim, ctxt_dim, **node_block_kwargs
        )
        self.glob_block = GlobBlockLite(
            inpt_dim, outp_dim, ctxt_dim, **glob_block_kwargs
        )

    def forward(
        self, graph: GraphLite, ctxt: T.Tensor = None
    ) -> GraphLite:
        """Return an updated graph with the same structure, but new features"""
        graph.edges, pooled_edges = self.edge_block(graph, ctxt)
        graph.nodes, pooled_nodes = self.node_block(graph, pooled_edges, ctxt)
        del pooled_edges ## Saves alot of memory if we delete right away
        graph.globs = self.node_block(graph, pooled_nodes, ctxt)
        return graph

    def __repr__(self):
        """A way to print the block config on one line for quick review"""
        string = str(self.inpt_dim)
        string += f"->EdgeNet[{self.edge_block.feat_net.one_line_string()}]"
        if self.edge_block.same_size:
            string += "(add)"
        string += f"->EdgePool[{self.edge_block.attn_net.one_line_string()}]"
        string += f"->NodeNet[{self.node_block.feat_net.one_line_string()}]"
        if self.node_block.same_size:
            string += "(add)"
        string += f"->NodePool[{self.node_block.attn_net.one_line_string()}]"
        string += f"->GlobNet[{self.glob_block.feat_net.one_line_string()}]"
        if self.glob_block.same_size:
            string += "(add)"
        return string


class GNBStack(nn.Module):
    """A stack of N many identical GNBlockLite(s)
    Graph to Graph
    """

    def __init__(
        self,
        inpt_dim: list,
        model_dim: list,
        num_blocks: int,
        ctxt_dim: int = 0,
        edge_block_kwargs: dict = None,
        node_block_kwargs: dict = None,
        glob_block_kwargs: dict = None,
    ) -> None:
        """
        args:
            num_blocks: The number of blocks in the stack
            inpt_dim: The dimensions of the input graph [e,n,g] (unchanging)
        kwargs:
            edge_block_kwargs: kwargs for the edge block
            node_block_kwargs: kwargs for the node block
            glob_block_kwargs: kwargs for the glob block
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.inpt_dim = inpt_dim
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim
        self.blocks = nn.ModuleList(
            [
                GNBlockLite(
                    inpt_dim if i == 0 else model_dim,
                    model_dim,
                    ctxt_dim,
                    edge_block_kwargs,
                    node_block_kwargs,
                    glob_block_kwargs
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, graph: GraphLite, ctxt: T.Tensor = None) -> T.Tensor:
        """Pass the input through all layers sequentially"""
        for blocks in self.blocks:
            graph = blocks(graph, ctxt)
        return graph
