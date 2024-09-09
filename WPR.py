import torch
import numpy as np
from torchsummary import summary


class Aconv1d(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(Aconv1d, self).__init__()

        assert activate in ['sigmoid', 'tanh']

        self.dilation = dilation
        self.activate = activate

        self.dilation_conv1d = torch.nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=7, dilation=self.dilation, bias=False)
        self.bn = torch.nn.BatchNorm1d(channel_out)


    def forward(self, inputs):
        # padding number = (kernel_size - 1) * dilation / 2
        inputs = torch.nn.functional.pad(inputs, (3*self.dilation, 3*self.dilation))
        outputs = self.dilation_conv1d(inputs)
        outputs = self.bn(outputs)

        if self.activate == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.tanh(outputs)

        return outputs


class ResnetBlock(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(ResnetBlock, self).__init__()
        self.conv_filter = Aconv1d(dilation, channel_in, channel_out, activate='tanh')
        self.conv_gate = Aconv1d(dilation, channel_in, channel_out, activate='sigmoid')

        self.conv1d = torch.nn.Conv1d(channel_out, out_channels=128, kernel_size=7,padding=3,stride=1, bias=False)
        self.bn = torch.nn.BatchNorm1d(128)

    def forward(self, inputs):
        out_filter = self.conv_filter(inputs)
        out_gate = self.conv_gate(inputs)
        outputs = out_filter * out_gate

        outputs = torch.tanh(self.bn(self.conv1d(outputs)))
        out = outputs + inputs
        return out, outputs

class WPR(torch.nn.Module):
    def __init__(self, num_classes, channels_in=1, channels_out=128, num_layers=3, dilations=[1,2,4,8,16]): # dilations=[1,2,4]
        super(WPR, self).__init__()
        self.num_layers = num_layers
        self.con1d1 = torch.nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=1,bias=False)
        self.bn = torch.nn.BatchNorm1d(channels_out)
        self.resnet_block_0 = torch.nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_1 = torch.nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_2 = torch.nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.conv1d_out = torch.nn.Conv1d(channels_out, channels_out, kernel_size=7, padding=3,stride=1,bias=False)
        self.bn2 = torch.nn.BatchNorm1d(channels_out)
        self.get_logits = torch.nn.Conv1d(in_channels=channels_out, out_channels=num_classes, kernel_size=7, padding=3,stride=1,bias=False)

    def forward(self, inputs):
        x = torch.tanh(self.con1d1(inputs))
        outs = 0.0
        for layer in self.resnet_block_0:
            x, out = layer(x)
            outs += out
        for layer in self.resnet_block_1:
            x, out = layer(x)
            outs += out
        for layer in self.resnet_block_2:
            x, out = layer(x)
            outs += out

        outs = torch.relu(self.bn2(self.conv1d_out(outs)))
        logits = torch.tanh(self.get_logits(outs))
        return logits