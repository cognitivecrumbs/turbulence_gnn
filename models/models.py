import torch
from torch import nn
from torch_scatter import scatter_sum
from typing import Optional

class ResidualBlock(torch.nn.Module):
    def __init__(self, 
                 ndim: int = 256, 
                 activation: nn.modules.activation = nn.ReLU(), 
                 dropout_rate: int=0.1, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = nn.Sequential(
            nn.Linear(ndim, ndim),
            activation,
            nn.Linear(ndim, ndim),
            activation,
            nn.LayerNorm(ndim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)

class MeshGraphNet(torch.nn.Module):
    def __init__(self, 
                 npasses: int = 4, 
                 ndim: int = 128, 
                 node_fc_depth: int = 6, 
                 edge_fc_depth: int = 6, 
                 node_input_dim: int = 5, 
                 edge_input_dim: int = 0, 
                 activation: nn.modules.activation = nn.ReLU(),
                 dropout_rate: float = 0.1,
                 concat: bool = False,
                 global_dim: int = 0,
                 *args, 
                 **kwargs) -> None:
                 
        super().__init__(*args, **kwargs)
        self.npasses = npasses
        self.concat = concat
        self.global_dim = global_dim
        self.encoder_node = nn.Linear(node_input_dim, ndim)
        if edge_input_dim > 0:
            self.encoder_edge = nn.Linear(edge_input_dim, ndim)
        if self.global_dim > 0:
            self.encoder_global = nn.Linear(ndim,global_dim)
            
        self.decoder_node = nn.Linear(ndim, node_input_dim)

        self.node = Node(node_fc_depth,ndim,activation=activation,dropout_rate=dropout_rate)
        self.edge = Edge(edge_fc_depth,ndim,activation=activation,dropout_rate=dropout_rate)

    def forward(self, v: torch.Tensor, ij: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        # edge -> node -> ... -> edge -> node
        v = self.encoder_node(v)

        if e is not None: 
            if len(e.shape) == 2 and self.concat:
                e = e.unsqueeze(0).repeat(v.shape[0],1,1)
            e = self.encoder_edge(e)

        for _ in range(self.npasses):
            e = self.edge(v, ij, e)
            v = self.node(v, ij, e)

        v = self.decoder_node(v)
        return v
    
    

class MeshGraphNetGlobal(torch.nn.Module):
    def __init__(self, 
                 npasses: int = 4, 
                 ndim: int = 128, 
                 node_fc_depth: int = 6, 
                 edge_fc_depth: int = 6, 
                 node_input_dim: int = 5, 
                 edge_input_dim: int = 0, 
                 activation: nn.modules.activation = nn.ReLU(),
                 dropout_rate: float = 0.1,
                 *args, 
                 **kwargs) -> None:
                 
        super().__init__(*args, **kwargs)
        self.npasses = npasses
        self.encoder_node = nn.Linear(node_input_dim, ndim)
        if edge_input_dim > 0:
            self.encoder_edge = nn.Linear(edge_input_dim, ndim)
        self.decoder_node = nn.Linear(ndim, node_input_dim)

        self.node = Node(node_fc_depth,ndim,activation=activation,dropout_rate=dropout_rate)
        self.edge = Edge(edge_fc_depth,ndim,activation=activation,dropout_rate=dropout_rate)

    def forward(self, v: torch.Tensor, ij: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        # edge -> node -> ... -> edge -> node
    
        v = self.encoder_node(v)

        if e is not None: 
            e = self.encoder_edge(e)

        for _ in range(self.npasses):
            e = self.edge(v, ij, e)
            v = self.node(v, ij, e)

        v = self.decoder_node(v)
        return v
    
# class PIGraph(torch.nn.Module):
#     def __init__(self, npasses: int = 4, ndim: int = 128, fc_depth: int = 6, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.npasses = 4 # placeholder
#         self.encoder_node = nn.Linear(2, ndim)
#         self.encoder_edge = nn.Linear(2, ndim)
#         self.decoder_node = nn.Linear(ndim, 2)

#         self.node = Node(fc_depth,ndim)
#         self.edge = Edge(fc_depth,ndim)

#     def least_squares_gradient_init()
#         mask = ij[0] == i
#         # d = e[ij[0,mask]]
#         d = e[mask]
#         w = torch.diag_embed(weighting[mask])
#         g = d.T@w.T@w@d
#         # print(g)

#         g_inv = torch.linalg.inv(g)
#         mat = g_inv@d.T@w.T@w
#         inv_square_mesh_mat[i] = mat

#     def forward(self, v: torch.Tensor, e: torch.Tensor, ij: torch.Tensor) -> torch.Tensor:
#         # calculate convective flux
#         # v = [u,v,p]
#         # determine surface area and normal vector flux is applied to
#         # delaunay triangularization used
#         surface_areas
#         normal_vectors
#         # compute convective fluxes
#         convective_momentum_change

#         # edge -> node -> ... -> edge -> node 
#         v = self.encoder_node(v)
#         e = self.encoder_edge(e)
#         for _ in range(self.npasses):
#             e = self.edge(v,ij)
#             v = self.node(v, e, ij)
            
#         v = self.decoder_node(v)

#         return v + convective_momentum_change # NN based correction to convective fluxes (dissipation and other higher order effects from turbulence and longer time steps)


class Node(torch.nn.Module):
    def __init__(self, 
                 depth:int=6, 
                 ndim:int=128, 
                 activation: nn.modules.activation = nn.ReLU(), 
                 dropout_rate: float = 0.1, 
                 concat: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff = nn.ModuleList([
            ResidualBlock(ndim=ndim, activation=activation, dropout_rate=dropout_rate)
            for _ in range(depth)
        ])
        self.concat = concat
        if self.concat:
            self.reduction = nn.Linear(ndim*2, ndim)
    
    def forward(self, v, ij, e=None):
        eij = scatter_sum(e, ij[1], dim=1)
        # eij = eij.unsqueeze(0)

        if self.concat:
            x = self.reduction(torch.cat([v,eij],dim=-1))
        else:
            x = v+eij if e is not None else v
        for layer in self.ff:
            x = layer(x)

        return x

class Edge(torch.nn.Module):
    def __init__(self, 
                 depth:int=6, 
                 ndim:int=128, 
                 activation: nn.modules.activation = nn.ReLU(), 
                 dropout_rate: float = 0.1, 
                 concat: bool = False,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ff = nn.ModuleList([
            ResidualBlock(ndim=ndim, activation=activation, dropout_rate=dropout_rate)
            for _ in range(depth)
        ])
        self.concat = concat
        if self.concat:
            self.reduction = nn.Linear(ndim*3, ndim)
    
    def forward(self, v, ij, e=None):
        # if len(e.shape) == 2:
        #     e = e.unsqueeze(0)
        if self.concat:
            x = self.reduction(torch.cat([v[:,ij[0]],v[:,ij[1]],e],dim=-1))
        else:
            x = v[:,ij[0]] - v[:,ij[1]] + e if e is not None else v[:,ij[0]] - v[:,ij[1]]

        for layer in self.ff:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = MeshGraphNet(edge_input_dim=2)
    x = torch.randn((100,5))
    ij = torch.randint(0,100,(2,300))
    e = torch.randn((300,2))
    out = model(x,ij,e)
    print(out.shape)