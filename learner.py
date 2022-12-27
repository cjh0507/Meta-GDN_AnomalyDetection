import  torch
from    torch import nn
from    torch.nn import functional as F
# import  numpy as np


class Learner(nn.Module):

    def __init__(self, config):
        super(Learner, self).__init__()

        self.config = config # Model architecture information이 (name, param) 형태로 들어있음. 이 때, param은 linear의 경우 (ch_out, ch_in)의 형태이다.
        self.vars = nn.ParameterList() # 각 layer 들의 W, b가 담겨 있음. [W1, b1, W2, b2, W3, b3, ...]의 형태 
        self.vars_bn = nn.ParameterList() # ! <- 현재 세팅으로는 사용되지 않음. Batch normalization에서 사용되는 mean, variance를 담고 있음.
        for i, (name, param) in enumerate(self.config): # from modelArch() in getConfig.py
            if name is 'linear': # ! <- Fully connected layer. 뭔가 여러 가지 내부 모델을 넣어보려한 것 같은데 이것만 쓰임
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'bn': # Batch Normalization을 의미하는 듯
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError
            
    def extra_repr(self):
        info = ''
        for name, param in self.config:
            if name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        '''
        :param x: 이 network에 넣을 input
        :param vars: 기본값은 None이고, vars가 None이면 learner 내부의 vars를 사용함.
        :param bn_training: Batch Normalization을 할 때만 의미 있음.
        :return:
        '''
        if vars is None:
            vars = self.vars
        idx = 0 # vars 탐색용
        idx_bn = 0 # vars_bn 탐색용
        for name, param in self.config:
            if name is 'linear': # ! <- 이것만 사용됨
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b) # F.linear(x, w, b) = xW^T + b <- W의 transpose를 사용하기 때문에 위에서 param을 (ch_out, ch_in)의 형태로 정의해서 쓴 것 같음.
                idx += 2
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[idx_bn], self.vars_bn[idx_bn+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training) # 이 때, running_mean, running_var 변화함
                idx += 2
                idx_bn += 2
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            else:
                raise NotImplementedError
        assert idx == len(vars)
        assert idx_bn == len(self.vars_bn)

        return x
    
    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars