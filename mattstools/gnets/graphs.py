"""
Defines the graph object type and other operations specific to handing them
"""

from typing import Iterable, Union

import torch as T
from torch.utils.data._utils.collate import default_collate

from mattstools.torch_utils import sel_device


class Graph:
    """The base class for the custom graph object

    A collection of 6 tensors:
    - 4 describe the attributes of the edges, nodes, globals and conditionals
    - 2 describe the structure, providing masks for the edges (adjmat) and nodes (mask)
    """

    def __init__(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        globs: T.Tensor,
        cndts: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
        dev: str = "cpu",
    ) -> None:
        """
        args:
            edges: Compressed edge features (num_edges x Ef)
            nodes: Node features (N x Nf)
            globs: Global features (Gf)
            cndts: Conditional features (Cf)
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
        self.cndts = cndts.to(self.device)
        self.adjmat = adjmat.to(self.device)
        self.mask = mask.to(self.device)

    @property
    def dtype(self):
        """Inherits the dtype from the node tensor"""
        return self.nodes.dtype

    def dim(self):
        """Return the dimensions of the graph"""
        return [
            self.edges.shape[-1],
            self.nodes.shape[-1],
            self.globs.shape[-1],
            self.cndts.shape[-1],
        ]

    def __len__(self):
        """Return the masking length of the graph"""
        return len(self.mask)

    def to(self, dev: str):
        """Move the graph to a selected device"""
        return Graph(
            self.edges,
            self.nodes,
            self.globs,
            self.cndts,
            self.adjmat,
            self.mask,
            dev=dev,
        )


class GraphBatch:
    """A batch of graph objects

    Batching the nodes, globs, cndts, adjmat and mask are simple as they just
    receive an extra batch dimension.

    Batching the edges however requires more steps as the edges are in compressed form
    This means that only the nonzero edges in the graph are stored such that
    full_edges[adjmat] = edges

    """

    def __init__(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        globs: T.Tensor,
        cndts: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
        dev: str = "cpu",
    ) -> None:
        """
        args:
            edges: Compressed edge features (Num_edges x Ef)
            nodes: Node features (B x N x Nf)
            globs: Global features (B x Gf)
            cndts: Conditional features (B x Cf)
            adjmat: The adjacency matrix, a mask for the edge features (B x N x N)
            mask: The node mask (B x N)
        kwargs:
            dev: A string indicating the device on which to store the tensors
        """

        ## Save the device where the graph will be stored
        self.device = sel_device(dev)

        ## Save each of the component tensors onto the correct device
        self.edges = edges.to(self.device)
        self.nodes = nodes.to(self.device)
        self.globs = globs.to(self.device)
        self.cndts = cndts.to(self.device)
        self.adjmat = adjmat.to(self.device)
        self.mask = mask.to(self.device)

    @property
    def dtype(self):
        """Inherits the dtype from the node tensor"""
        return self.nodes.dtype

    def __getitem__(self, idx: int):
        """Retrieve a particular graph from within the graph batch using an index"""

        ## Work out the indexes of the edge tensor
        start = self.adjmat[:idx].sum()
        end = self.adjmat[idx].sum()

        return Graph(
            self.edges[start:end],
            self.nodes[idx],
            self.globs[idx],
            self.cndts[idx],
            self.adjmat[idx],
            self.mask[idx],
            dev=self.device,
        )

    def __len__(self):
        """Return the length of the graph batch"""
        return len(self.mask)

    def max_n(self):
        """Return the number of nodes that the batch can hold"""
        return self.mask.shape[-1]

    def dim(self):
        """Return the dimensions of the graph object starting with the batch length"""
        return (
            len(self),
            [
                self.edges.shape[-1],
                self.nodes.shape[-1],
                self.globs.shape[-1],
                self.cndts.shape[-1],
            ],
        )

    def to(self, dev: str):
        """Move the graph to a selected device"""
        return GraphBatch(
            self.edges,
            self.nodes,
            self.globs,
            self.cndts,
            self.adjmat,
            self.mask,
            dev=dev,
        )

    def has_nan(self):
        """Check if there is any nan values in the graph's tensors"""
        result = [
            T.isnan(self.edges).any().item(),
            T.isnan(self.nodes).any().item(),
            T.isnan(self.globs).any().item(),
            T.isnan(self.cndts).any().item(),
            T.isnan(self.adjmat).any().item(),
            T.isnan(self.mask).any().item(),
        ]
        return result

    def batch_select(self, b_mask: T.BoolTensor):
        """Returns a batched graph object made from the subset of another batched graph
        This function needs to exist to account for the edges which have no batch
        dimension

        Operation returns a new graph batch
        """

        assert self.adjmat.sum() == len(self.edges)

        return GraphBatch(
            self.edges[b_mask.repeat_interleave(self.adjmat.sum((-1, -2))).bool()],
            self.nodes[b_mask],
            self.globs[b_mask],
            self.cndts[b_mask],
            self.adjmat[b_mask],
            self.mask[b_mask],
        )

    def batch_replace(self, graph_2, b_mask: T.BoolTensor) -> None:
        """Replace samples with those from graph_2 following a mask
        Number of graphs in graph_2 must be smaller!
        Operation modifies the current graph batch
        """
        self.adjmat[b_mask] = graph_2.adjmat
        self.nodes[b_mask] = graph_2.nodes
        self.globs[b_mask] = graph_2.globs
        self.cndts[b_mask] = graph_2.cndts
        self.mask[b_mask] = graph_2.mask

        ## This step kills all persistant edges until I work out how to do this
        ## TODO Work out how to batch replace without killing edges!
        self.edges = T.zeros(
            (self.adjmat.sum(), 0),
            dtype=T.float,
            device=self.device,
        )

    def __repr__(self):
        """Return the name of the graph and its dimension for printing"""
        return f"GraphBatch({self.dim()})"

    def clone(self):
        """Returns a copy of itself so the GNBlock does not make inplace changes"""

        return GraphBatch(
            self.edges,
            self.nodes,
            self.globs,
            self.cndts,
            self.adjmat,
            self.mask,
            dev=self.device,
        )


def graph_collate(batch: Iterable) -> Union[GraphBatch, tuple]:
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
    if isinstance(elem, Graph):
        edges = T.cat([g.edges for g in batch])  ## Input edges should be compressed
        nodes = default_collate([g.nodes for g in batch])
        globs = default_collate([g.globs for g in batch])
        cndts = default_collate([g.cndts for g in batch])
        adjmat = default_collate([g.adjmat for g in batch])
        mask = default_collate([g.mask for g in batch])

        return GraphBatch(edges, nodes, globs, cndts, adjmat, mask, dev=mask.device)

    ## If we have a tuple, we must run the function for each object
    elif isinstance(elem, tuple):
        return tuple(graph_collate(samples) for samples in zip(*batch))

    ## If we are dealing with any other type we must run the normal collation function
    else:
        return default_collate(batch)


def blank_graph_batch(
    dim: list, max_nodes: int, b_size: int = 1, dev: str = "cpu"
) -> GraphBatch:
    """Create an empty graph of a certain shape
    - All attributes are zeros
    - All masks/adjmats are false

    args:
        dim: The dimensions of the desired graph [e,n,g,c]
        max_nodes: The max number of nodes to allow for
        b_size: The batch dimension
    kwargs:
        dev: The device on which to store the graph
    returns:
        Empty graph object
    """
    dev = sel_device(dev)
    edges = T.zeros((0, dim[0]), device=dev)
    nodes = T.zeros((b_size, max_nodes, dim[1]), device=dev)
    globs = T.zeros((b_size, dim[2]), device=dev)
    cndts = T.zeros((b_size, dim[3]), device=dev)
    adjmat = T.zeros((b_size, max_nodes, max_nodes), device=dev).bool()
    mask = T.zeros((b_size, max_nodes), device=dev).bool()

    return GraphBatch(edges, nodes, globs, cndts, adjmat, mask, dev=dev)
