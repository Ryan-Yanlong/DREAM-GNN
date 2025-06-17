import dgl
import math
import torch as th
import torch.nn as nn
try:
    import dgl.fn as fn
except ImportError:
    import dgl.function as fn

import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

from torch.nn import init
from utils import get_activation, to_etype_name
from torch.nn.parameter import Parameter


class GCMCLayer(nn.Module):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}
    """
    def __init__(self, rating_vals,
                 user_in_units, 
                 movie_in_units, 
                 msg_units, 
                 out_units, 
                 dropout_rate=0.1,  
                 agg='stack',
                 agg_act=None,
                 ini=True, 
                 share_user_item_param=False, 
                 basis_units=2, 
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals  
        self.agg = agg 
        self.share_user_item_param = share_user_item_param 
        self.user_in_units = user_in_units

        # First calculate effective message dimension
        effective_msg_units = msg_units
        if agg == 'stack':
            assert effective_msg_units % len(rating_vals) == 0
            effective_msg_units = effective_msg_units // len(rating_vals)
        if ini:
            effective_msg_units = effective_msg_units // 3 
        self.msg_units = effective_msg_units

        # Construct linear mapping layer with effective_msg_units as input features
        self.ufc = nn.Linear(effective_msg_units, out_units)  
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(effective_msg_units, out_units)

        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = {}
        subConv = {}
        self.basis_units = basis_units 
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units)) 
        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, effective_msg_units))
        for i, rating in enumerate(rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                effective_msg_units, 
                                                weight=False, 
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    effective_msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                effective_msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    effective_msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Move parameters except W_r to specified device"""
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if not self.share_user_item_param:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, drug_feat=None, dis_feat=None, Two_Stage=False):
            in_feats = {'drug': drug_feat, 'disease': dis_feat}
            mod_args = {}
            self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
            self.W = self.W.view(-1, self.user_in_units, self.msg_units)
            for i, rating in enumerate(self.rating_vals):
                rating = to_etype_name(rating)
                rev_rating = 'rev-%s' % rating

                mod_args[rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)
                mod_args[rev_rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)

            out_feats = self.conv(graph, in_feats, mod_args=mod_args)
            drug_feat = out_feats['drug']
            dis_feat = out_feats['disease']

            # Activation and dropout
            drug_feat = self.agg_act(drug_feat)
            drug_feat = self.dropout(drug_feat)

            dis_feat = self.agg_act(dis_feat)
            dis_feat = self.dropout(dis_feat)

            drug_feat = self.ifc(drug_feat)
            dis_feat = self.ufc(dis_feat)

            return drug_feat, dis_feat

    
class GCMCGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.1):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats 
        self._out_feats = out_feats 
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None, Two_Stage=False):
        """Compute graph convolution

        Use local scope to prevent data contamination.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used
            # Ensure input feat is on specified device (if device is specified)
            if self.device is not None:
                feat = feat.to(self.device)
            
            # Get cj and ci
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
                
            # Check and handle size mismatch
            num_src_nodes = graph.number_of_src_nodes()
            
            # Ensure feat and cj size match
            if feat.size(0) != num_src_nodes:
                print(f"Warning: feat size ({feat.size(0)}) doesn't match source node count ({num_src_nodes})")
                # If feat is too big, truncate it
                if feat.size(0) > num_src_nodes:
                    feat = feat[:num_src_nodes]
                # If feat is too small, pad it (by repeating last row)
                else:
                    padding = feat[-1].unsqueeze(0).repeat(num_src_nodes - feat.size(0), 1)
                    feat = th.cat([feat, padding], dim=0)
            
            # Ensure cj size matches feat
            if cj.size(0) != feat.size(0):
                print(f"Warning: cj size ({cj.size(0)}) doesn't match feat size ({feat.size(0)})")
                # If cj is too big, truncate it
                if cj.size(0) > feat.size(0):
                    cj = cj[:feat.size(0)]
                # If cj is too small, pad it
                else:
                    padding = th.ones(feat.size(0) - cj.size(0), 1, device=cj.device)
                    cj = th.cat([cj, padding], dim=0)
            
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight provided but module also has its own weight parameter, please set weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            # Ensure dropout(cj) has same device and size as feat
            cj_dropout = self.dropout(cj).view(-1, 1)
            feat = feat * cj_dropout
            
            graph.srcdata['h'] = feat
            # Use new API: copy_u instead of copy_src
            graph.update_all(
                fn.copy_u('h', 'm'),
                fn.sum('m', 'h')
            )
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst

class GCN(nn.Module):
    def __init__(self, features, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(features, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class FGCN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout):
        super(FGCN, self).__init__()
        self.FGCN_drug = GCN(fdim_drug, nhid1, nhid2, dropout)
        self.FGCN_disease = GCN(fdim_disease, nhid1, nhid2, dropout)
        self.dropout = dropout # Store dropout rate
        self.drug_fusion = nn.Linear(nhid2 * 2, nhid2)
        self.disease_fusion = nn.Linear(nhid2 * 2, nhid2)

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat,
                drug_feature_graph=None, disease_feature_graph=None):

        emb1_sim = self.FGCN_drug(drug_sim_feat, drug_graph)
        emb2_sim = self.FGCN_disease(disease_sim_feat, dis_graph)

        emb1_feat, emb2_feat = None, None # Initialize in case graph is None

        if drug_feature_graph is not None and disease_feature_graph is not None:
            emb1_feat = self.FGCN_drug(drug_sim_feat, drug_feature_graph)
            emb2_feat = self.FGCN_disease(disease_sim_feat, disease_feature_graph)

            fused_drug = th.relu(self.drug_fusion(th.cat([emb1_sim, emb1_feat], dim=1)))
            fused_disease = th.relu(self.disease_fusion(th.cat([emb2_sim, emb2_feat], dim=1)))

            # Add dropout
            # Use self.training to ensure dropout is only active during training
            emb1 = F.dropout(fused_drug, p=self.dropout, training=self.training)
            emb2 = F.dropout(fused_disease, p=self.dropout, training=self.training)

        else:
            # If only one type of graph, no fusion dropout needed, just pass through
            emb1, emb2 = emb1_sim, emb2_sim

        # Return all embeddings, including intermediate ones if they exist
        return emb1, emb2, emb1_sim, emb1_feat, emb2_sim, emb2_feat

class GraphConvolution(nn.Module):
    """Simple GCN layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        device = input.device
        if adj.device != device:
            adj = adj.to(device)
            
        support = th.mm(input, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16, dropout_rate=0.1):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        w = self.project(z)
        beta = th.softmax(w, dim=1)
        beta = self.dropout(beta)  # Add Dropout
        return (beta * z).sum(1), beta


class MLPDecoder(nn.Module):
    def __init__(self,
                 in_units,
                 dropout_rate=0.1):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.lin1 = nn.Linear(2 * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']

            out = F.relu(self.lin1(out))
            out = self.dropout(out)

            out = F.relu(self.lin2(out))
            out = self.dropout(out)

            out = self.lin3(out)

        return out
    

def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}


def dot_or_identity(A, B, device=None):
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1).to(device)
    else:
        # For dense embedding case, perform linear transformation
        return th.matmul(A, B)