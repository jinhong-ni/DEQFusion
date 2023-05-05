""" 
    DEQ fusion module
    Modified based on the DEQ repo (https://github.com/locuslab/deq)
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from .solver import anderson, broyden, weight_tie
from .jacobian import jac_loss_estimate, power_method


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

def list2vec(z1_list):
    """Convert list of tensors to a vector"""
    bsz = z1_list[0].size(0)
    return torch.cat([elem.reshape(bsz, -1) for elem in z1_list], dim=1)

def vec2list(z1, cutoffs):
    """Convert a vector back to a list, via the cutoffs specified"""
    bsz = z1.shape[0]
    z1_list = []
    start_idx, end_idx = 0, cutoffs[0]
    for i in range(len(cutoffs)):
        z1_list.append(z1[:, start_idx:end_idx].view(bsz, cutoffs[i]))
        if i < len(cutoffs)-1:
            start_idx = end_idx
            end_idx += cutoffs[i + 1]
    return z1_list


class SimpleResidualBlock(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(SimpleResidualBlock, self).__init__()

        self.out_dim = out_dim

        self.conv1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.gn1 = nn.GroupNorm(4, self.out_dim, affine=True)
        self.gn2 = nn.GroupNorm(4, self.out_dim, affine=True)
        self.gn3 = nn.GroupNorm(4, self.out_dim, affine=True)
        
    def forward(self, x, injection_feature):
        residual = x

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.conv2(out) + injection_feature
        out = self.gn2(out)

        out += residual
        out = self.gn3(F.relu(out))
        
        return out


class DEQFusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        Purified-then-combined fusion block.
        """
        super(DEQFusionBlock, self).__init__()

        self.out_dim = num_out_dims[-1]
        
        self.gate = nn.Linear(num_out_dims[0], self.out_dim)
        self.fuse = nn.Linear(self.out_dim, self.out_dim)
            
        self.gn3 = nn.GroupNorm(4, self.out_dim, affine=True)
            
    def forward(self, x, injection_features, residual_feature):
        extracted_feats = []
        for i, inj_feat in enumerate(injection_features):
            extracted_feats.append(torch.mul(x, self.gate(inj_feat + x)))
        
        out = self.fuse(torch.stack(extracted_feats, dim=0).sum(dim=0)) + residual_feature
#         out = self.gn3(F.relu(out))
        out = F.relu(self.gn3(out))
        
        return out

    
class DEQFusionLayer(nn.Module):
    def __init__(self, num_out_dims):
        super(DEQFusionLayer, self).__init__()

        self.num_branches = len(num_out_dims)
        self.block = SimpleResidualBlock
        self.fusion_block = DEQFusionBlock
        self.branches = self._make_branches(self.num_branches, num_out_dims)

    def _make_one_branch(self, branch_index, num_out_dims):
        out_dim = num_out_dims[branch_index]
        return self.block(out_dim, deq_expand=1, num_groups=1, dropout=0)
    
    def _make_fusion_branch(self, branch_index, num_out_dims):
        #out_dim = num_out_dims[branch_index]
        return self.fusion_block(num_out_dims, deq_expand=2, dropout=0)

    def _make_branches(self, num_branch, num_out_dims):
        branch_layers = [self._make_one_branch(i, num_out_dims) for i in range(num_branch - 1)]
        branch_layers.append(self._make_fusion_branch(num_branch - 1, num_out_dims))
        return nn.ModuleList(branch_layers)

    def forward(self, x, injection):
        x_block_out = []
        for i in range(self.num_branches - 1):
            out = self.branches[i](x[i], injection[i])
            x_block_out.append(out)
        x_block_out.append(self.branches[self.num_branches - 1](x[self.num_branches - 1], x_block_out, injection[-1]))

        return x_block_out


class DEQFusion(nn.Module):
    def __init__(self, channel_dim, num_modals, f_thres=55, b_thres=56, stop_mode="abs", deq=True, num_layers=1, solver='anderson'):
        super(DEQFusion, self).__init__()
        self.f_thres = f_thres
        self.b_thres = b_thres
        self.stop_mode = stop_mode
        self.func_ = DEQFusionLayer([channel_dim for _ in range(num_modals + 1)])
        self.f_solver = anderson if solver == 'anderson' else broyden
        self.b_solver = anderson if solver == 'anderson' else broyden
        self.deq = deq
        self.num_layers = num_layers
        
        self.weights = nn.ParameterList()
        for i in range(num_modals):
            self.weights.append(nn.Parameter(torch.FloatTensor(1), requires_grad=True))
            self.weights[i].data.fill_(1)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias.data.fill_(0)
        
    def featureFusion(self, features, fusion_feature, compute_jac_loss=True):
        batch_size = features[0].shape[0]
        feature_dim = features[0].shape[1]

        x_list = [f for f in features] + [fusion_feature]
        out_dim_list = [f.shape[1] for f in features] + [fusion_feature.shape[1]]
        z_list = [torch.zeros(batch_size, dim_size).cuda() for dim_size in out_dim_list]
        cutoffs = [elem.size(1) for elem in z_list]
        z1 = list2vec(z_list)
        #self.func_._reset(z_list)
        func = lambda z: list2vec(self.func_(vec2list(z, cutoffs), x_list))

        deq = self.deq
        jac_loss = torch.tensor(0.0).to(fusion_feature)

        if not deq:
            result = {'rel_trace':[]}
            for layer_ind in range(self.num_layers): 
                z1 = func(z1)
                print(z1.min(),z1.max())
            new_z1 = z1
#             if self.training:
#                 if compute_jac_loss:
#                     z2 = z1.clone().detach().requires_grad_()
#                     new_z2 = func(z2)
#                     jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(func, z1, threshold=self.f_thres, stop_mode=self.stop_mode)
                z1 = result['result']
                print(result['nstep'], end=" ")
            new_z1 = z1
            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    if self.hook is not None:
                        # To avoid infinite loop
                        self.hook.remove()
                        torch.cuda.synchronize()
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, \
                                                torch.zeros_like(grad), threshold=self.b_thres)['result']
                    return new_grad
                self.hook = new_z1.register_hook(backward_hook)
        net = vec2list(new_z1, cutoffs)

        return net[-1], jac_loss.view(1,-1), result
    
    def forward(self, features):
        """ 
            features: List[Tensor], features from different modalities
        """
        # Naive weighted sum fusion: x_fuse
        fusion_feature = torch.stack([self.weights[i] * f for i, f in enumerate(features)], dim=0).sum(dim=0) + self.bias
        
        fused_feat, jacobian_loss, trace = self.featureFusion(features, fusion_feature)
        
        return fused_feat, jacobian_loss, trace
