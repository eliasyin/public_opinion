from torch import nn


class mySequential(nn.Sequential):
    """
    Due to nn.Sequential can not handle modules with different num of inputs,
    this module is created to solve the problem
    reference:
        https://github.com/pytorch/pytorch/issues/19808#issuecomment-487291323
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
