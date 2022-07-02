import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat.view(y.shape),y) + self.eps)
        return loss


class DotProductLoss(nn.Module):

    def __init__(self):
        super(DotProductLoss, self).__init__()

    def forward(self, output, target):
        return -torch.dot(target.view(-1), output.view(-1)) / target.nelement()


class WeightedCrossEntropy(nn.Module):

    def __init__(self, weights=None, size_average=True):
        super(WeightedCrossEntropy, self).__init__()
        self.weights = weights
        self.size_average = size_average
        assert (self.size_average == True)  # Not implemented for the other case

    def forward(self, output, target):
        loss = nn.CrossEntropyLoss(self.weights, self.size_average)
        output_one = output.view(-1)
        output_zero = 1 - output_one
        output_converted = torch.stack([output_zero, output_one], 1)
        target_converted = target.view(-1).long()
        return loss(output_converted, target_converted)
    

class MultiLabelCrossEntropyLoss(nn.Module):

    def __init__(self, weights, size_average=True):
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.criterions = [
            nn.CrossEntropyLoss(weight, size_average)

            for weight in weights.cuda()
        ]
        self.size_average = size_average

    def forward(self, output, target):
        assert output.dim() == 4
        assert target.dim() == 3
        if output.size(0) != target.size(0):
            raise ValueError("Inequal batch size ({} vs. {})".format(
                output.size(0), target.size(0)))
        if output.size(1) != target.size(1):
            raise ValueError("Inequal sequence length ({} vs. {})".format(
                output.size(1), target.size(1)))
        if output.size(2) != target.size(2):
            raise ValueError("Inequal number of labels ({} vs. {})".format(
                output.size(2), target.size(2)))
        if output.size(2) != len(self.criterions):
            raise ValueError("Unexpected number of labels ({} vs. {})".format(
                output.size(2), len(self.criterions)))
##        print("OUTPUT LENGTH",output.size())
##        print("TARGET LENGTH",target.size())
        loss = 0
    
        for i, criterion in enumerate(self.criterions):
            # Merge sequence length to batch_size
            label_output = output[:, :, i, :].contiguous().view(
                -1, output.size(3))

            label_target = target[:, :, i].contiguous().view(-1)
            label_target = label_target.long()
##            print(label_output.type())
##            print(label_target.type())
            loss += criterion(label_output, label_target-1)
        if self.size_average:
            loss /= len(self.criterions)
        return loss
    
