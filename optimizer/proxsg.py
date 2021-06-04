import time
import os
import csv
import torch
import numpy as np
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import argparse
import time

class ProxSG(Optimizer):

    def __init__(self, params, lr=required, lmbda=required, momentum=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if lmbda is not required and lmbda < 0.0:
            raise ValueError("Invalid lambda: {}".format(lmbda))

        if momentum is not required and momentum < 0.0:
            raise ValueError("Invalid momentum: {}".format(momentum))

        defaults = dict(lr=lr, lmbda=lmbda, momentum=momentum)
        super(ProxSG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ProxSG, self).__setstate__(state)
    
    def get_xs_grad_fs(self, params):
        xs = []
        grad_fs = []
        for p in params:
            if p.grad is None:
                continue
            if len(p.data.shape) == 1:
                xs.append(p.data.unsqueeze(1))
                grad_fs.append(p.grad.data.unsqueeze(1))
            elif len(p.data.shape) == 4: # conv layer
                xs.append(p.data.view(p.data.shape[0], -1))
                grad_fs.append(p.grad.data.view(p.grad.data.shape[0], -1))

            else:
                xs.append(p.data)
                grad_fs.append(p.grad.data)    
        return xs, grad_fs    

    def get_xs(self, params):
        xs = []
        for p in params:
            if len(p.data.shape) == 1:
                xs.append(p.data.unsqueeze(1))
            elif len(p.data.shape) == 4: # conv layer
                xs.append(p.data.view(p.data.shape[0], -1))
            else:
                xs.append(p.data)
        return xs    

    def update_xs_given_new_xs(self, params, xs, new_flatten_xs, shapes=None):
        if shapes is not None: # conv layer
            left_pointer = 0
            right_pointer = 0
            for i, x in enumerate(xs):
                right_pointer += shapes[i][1:].numel()
                params[i].data.copy_(new_flatten_xs[:, left_pointer:right_pointer].view(shapes[i]))
                left_pointer = right_pointer
        else:
            left_pointer = 0
            right_pointer = 0
            for i, x in enumerate(xs):
                right_pointer += x.shape[1]
                if right_pointer - left_pointer == 1:
                    params[i].data.copy_(new_flatten_xs[:, left_pointer:right_pointer].squeeze(1))
                else:
                    params[i].data.copy_(new_flatten_xs[:, left_pointer:right_pointer])
                left_pointer = right_pointer      

    def grad_descent_update(self, x, lr, grad):
        return x - lr * grad

    def reorg_multi_head(self, x, num_groups, num_heads):
        return x.view(num_heads, num_groups, -1).permute(1, 0, 2).contiguous().view(num_groups, -1)

    def reverse_reorg_multi_head(self, x, num_groups, num_heads):
        return x.view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous().view(num_heads * num_groups, -1)

    def get_momentum_grad(self, param_state, key, momentum, grad):
        if momentum > 0:
            if key not in param_state:
                buf = param_state[key] = grad
            else:
                buf = param_state[key]
                buf.mul_(momentum).add_(grad)
            return buf
        else:
            return grad 
    
    def prox_project(self, hat_x, alpha, lmbda):
        num_groups = hat_x.shape[0]
        denoms = torch.norm(hat_x, p=2, dim=1)
        numer = alpha * lmbda
        coeffs = 1.0 - numer / (denoms + 1e-6) 
        coeffs[coeffs<=0] = 0.0
        hat_x = coeffs.unsqueeze(1) * hat_x
        return hat_x

    def proxsg_step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['group_type'] == 0:
                """Params that do not need update"""
                continue
            elif group['group_type'] == 1:
                """Params that are not included in the regularization"""
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_f = p.grad.data
                    grad_f = self.get_momentum_grad(self.state[p], 'momentum_buffer_half_space', group['momentum'], grad_f)
                    p.data.add_(-group['lr'], grad_f)        
                    continue
        
            elif group['group_type'] == 2:
                """Group for multi-head linear layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)

                num_groups = flatten_x.shape[0] // group['num_heads']
                flatten_x = self.reorg_multi_head(flatten_x, num_groups, group['num_heads'])
                flatten_grad_f = self.reorg_multi_head(flatten_grad_f, num_groups, group['num_heads'])

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0

                # compute trial iterate
                flatten_hat_x = self.grad_descent_update(flatten_x, group['lr'], flatten_grad_f)

                # do half space projection
                flatten_hat_x =  self.prox_project(flatten_hat_x, group['lr'], group['lmbda'])

                # recover shape
                flatten_hat_x = self.reverse_reorg_multi_head(flatten_hat_x, num_groups, group['num_heads'])

                self.update_xs_given_new_xs(group['params'], xs, flatten_hat_x)
                
            elif group['group_type'] == 3:
                """Group for standard linear layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)
                
                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0

                # compute trial iterate
                flatten_hat_x = self.grad_descent_update(flatten_x, group['lr'], flatten_grad_f)

                # do half space projection
                flatten_hat_x = self.prox_project(flatten_hat_x, group['lr'], group['lmbda'])

                self.update_xs_given_new_xs(group['params'], xs, flatten_hat_x)

            elif group['group_type'] == 4:
                """Group for Conv layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)
                
                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0

                # compute trial iterate
                flatten_hat_x = self.grad_descent_update(flatten_x, group['lr'], flatten_grad_f)

                # do half space projection
                flatten_hat_x = self.prox_project(flatten_hat_x, group['lr'], group['lmbda'])

                self.update_xs_given_new_xs(group['params'], xs, flatten_hat_x, shapes=group['shapes'])
            else:
                raise("some parameters are not in any group type, please check")
        
        return loss

    def compute_group_sparsity_omega(self):
        total_num_groups = torch.zeros(5) + 1e-6
        total_num_zero_groups = torch.zeros(5)

        omega = 0.0
        for group in self.param_groups:
            if group['group_type'] == 0:
                """Params that do not need update"""
                total_num_zero_groups[group['group_type']] = 0
                pass
            elif group['group_type'] == 1:
                """Params that are not included in the regularization"""
                total_num_zero_groups[group['group_type']] = 0
                pass
            elif group['group_type'] == 2:
                xs = self.get_xs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)

                num_groups = flatten_x.shape[0] // group['num_heads']
                flatten_x = self.reorg_multi_head(flatten_x, num_groups, group['num_heads'])

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += num_groups
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
            elif group['group_type'] == 3:
                """Group for standard linear layer"""
                xs = self.get_xs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)                       

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += flatten_x.shape[0]
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
            elif group['group_type'] == 4:
                xs = self.get_xs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)                       

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += flatten_x.shape[0]
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)

        
        overall_group_sparsity = torch.sum(total_num_zero_groups) / torch.sum(total_num_groups)
        group_sparsities = total_num_zero_groups = total_num_zero_groups / total_num_groups
        return total_num_zero_groups.cpu().numpy(), total_num_groups.cpu().numpy(), group_sparsities.cpu().numpy(), overall_group_sparsity.cpu().numpy(), omega.cpu().numpy()
