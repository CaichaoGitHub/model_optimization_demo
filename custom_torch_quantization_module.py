# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 3:45 PM
# @Author  : cc
# @Email   :
# @File    : custom_torch_quantization_module.py
# @Software: PyCharm
# Copyright @ 2021 cc. All rights reserved.

import torch
from torch import Tensor
import onnx
import onnxruntime
import time
import copy
from torch.quantization.qconfig import QConfig,HistogramObserver,default_per_channel_qconfig,default_per_channel_weight_observer
from torch.nn.quantized.modules.conv import Conv2d
import numpy as np


class MyActivation(torch.nn.Module):
    def __init__(self):
        super(MyActivation,self).__init__()
        self.relu = torch.nn.ReLU6(inplace=False)

    def forward(self, x):
        return x * self.relu(x + 3) / 6

#A quantized custom module class should accept quantized input and return quantized output
class MyActivationQuant(torch.nn.Module):
    def __init__(self,scale=1.0,zeropoint=0):
        super(MyActivationQuant,self).__init__()
        self.scale = float(scale)
        self.zeropoint = int(zeropoint)
        self.relu = torch.nn.ReLU6(inplace=False)
        #self.observer = HistogramObserver.with_args(reduce_range=True)

    def forward(self, x):
        #fx = sx * (qx -zx)
        #y = fx * relu(fx + 3) / 6 =>
        #y = (sx^2 * (qx - zx) ^ 2 + sx * 3 * (qx - zx)) / 6
        sx = x.q_scale() # < 1 ?
        zx = x.q_zero_point()
        x_uint8 = x.int_repr().to(int) # convert to int in case of accuracy loss

        # qz - zx
        a = x_uint8 - zx
        # relu(fx + 3)
        mutipler = 536870912 # 2^29
        a1 = int(mutipler * sx) * a #sx * a
        # fix-point mutipler , using float16 instead float32 ?
        a2 = a1 / mutipler #a1 >>29 #a1 / mutipler , rhs operation will result in loss of accuracy  (round-to-nearest right-shift) ??
        a3 = torch.clamp(a2 + 3,0,3)

        # fx / 6
        a4 = int(mutipler * (sx / 6)) * a
        a5 = a4 / mutipler #a3 >> 29

        y = a5 * a3
        y_quantized = torch.quantize_per_tensor(y, scale=self.scale, zero_point=self.zeropoint, dtype=torch.quint8)
        #x1 = x.dequantize()
        #y1 = x1 * self.relu(x1 + 3) / 6


        #torch.quantize_per_tensor(torch.tensor([1,2,3],dtype=torch.float),scale=0.1,zero_point=0,dtype=torch.uint8)
        return y_quantized

    # `quantized_custom_module_class` will have a `from_observed` classmethod,
    # which will return an quantized custom module instance given
    # a observed custom module instance.
    # This will be used in prepare step of post training static quantization or
    # quantization aware training

    @staticmethod
    def from_observed(mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        ac = MyActivationQuant(scale,zero_point)
        #ac.qconfig = mod.qconfig
        return ac

    @staticmethod
    def from_float(mod): # float module => observerd module , observer some qparams
        #scale, zero_point = mod.activation_post_process.calculate_qparams()
        #ac = MyActivationQuant()
        #ac.qconfig = mod.qconfig
        return mod

class testHSwish(torch.nn.Module):
    def __init__(self):
        super(testHSwish,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = torch.nn.Conv2d(3,13,3)
        self.conv2 = torch.nn.Conv2d(13, 3, 3)
        self.hswish = MyActivation()#torch.nn.ReLU() #
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.hswish(x)
        return self.dequant(x)



def testQuantHSwishActivation():

    x = torch.randn(1, 3, 224, 224)
    m = testHSwish()
    m.eval()
    res0 = m(x)
    observer  = HistogramObserver.with_args(reduce_range=True)
    qconfig = QConfig(activation=observer,
                      weight=default_per_channel_weight_observer)

    torch.quantization.register_observed_custom_module_mapping(MyActivation,MyActivationQuant)
    torch.quantization.register_quantized_custom_module_mapping(MyActivation,MyActivationQuant)
    # torch.quantization.register_activation_post_process_hook()

    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.quantization.prepare(m) #转换
    res1 = model_fp32_prepared(x)
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    res2 = model_int8(x)
    print("succeed")

if __name__ == '__main__':
    testQuantHSwishActivation()