'''
Edited according to https://zhuanlan.zhihu.com/p/662569090
for CPU peft ft

'''
import torch
from .quantization import *




class LoraQuantizedLinear(torch.nn.Module):

    def __init__(self, q_linear, lora_r=32, lora_alpha=32, lora_dropout_rate=0.0):
        super().__init__()

        # 保存原始参数和Lora配置
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.weight_bit_width = q_linear.weight_bit_width
        self.weight = q_linear.weight
        self.weight_scale = q_linear.weight_scale
        self.bias = q_linear.bias

        # 冻结原始参数
        self.weight.requires_grad = False
        self.weight_scale.requires_grad = False
        if self.bias is not None: self.bias.requires_grad = False

        # 创建 Lora 参数，FP16
        out_dim, in_dim = self.weight.shape
        # INT4 模型下，InDim 是原始大小的一半
        if self.weight_bit_width == 4: in_dim *= 2
        # LoraA 正态初始化
        self.lora_a = torch.nn.Parameter(torch.empty(
            [self.lora_r, in_dim],
            device=self.weight.device,
            dtype=torch.float16,
        ))
        torch.nn.init.kaiming_normal_(self.lora_a)
        # LoraB 全零初始化
        self.lora_b = torch.nn.Parameter(torch.zeros(
            [out_dim, self.lora_r],
            device=self.weight.device,
            dtype=torch.float16,
        ))
        self.lora_dropout = torch.nn.Dropout(self.lora_dropout_rate)
        self.lora_scale = self.lora_alpha / self.lora_r

    def forward(self, input):
        ori_output = QuantizedLinear.forward(self, input)
        lora_output = (
            self.lora_dropout(input.half()) @
            self.lora_a.transpose(0, 1) @
            self.lora_b.transpose(0, 1) *
            self.lora_scale
        )
        return ori_output + lora_output.to(ori_output.dtype)

    def merge(self):
        # H = XW + b + XAB * s => H = X(W + AB * s) + b
        # 将 int 原始参数转成 fp16
        weight = extract_weight_to_half(self.weight, self.weight_scale, self.weight_bit_width)
        # 合并 lora 参数
        weight += self.lora_b @ self.lora_a * self.lora_scale
        # 再转回 int
        weight, weight_scale = half_weight_to_int(weight, self.weight_bit_width)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        # 重新初始化 lora 两个矩阵
        torch.nn.init.kaiming_normal_(self.lora_a)
        torch.nn.init.zeros_(self.lora_b)



# 将所有的QuantizedLinear改成LoraQuantizedLinear
def attach_lora(model, lora_r=32, lora_alpha=32, lora_dropout_rate=0.0):
    if model.lora_attached: return model
    lora_conf = dict(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout_rate=lora_dropout_rate)
    for mod in model.modules():
        for name in dir(mod):
            submod = getattr(mod, name, None)
            if not isinstance(submod, QuantizedLinear):
                continue
            new_submod = LoraQuantizedLinear(submod, **lora_conf)
            setattr(mod, name, new_submod)

    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    model.lora_attached = True
    return model


# 导出所有 Lora 参数。
def lora_state_dict(model):
    return {
        k:v
        for k, v in model.state_dict().items()
        if 'lora_' in k
    }


def base_state_dict(model):
    return {   
        k:v
        for k, v in model.state_dict().items()
        if 'lora_' not in k
    }


# 将所有LoraQuantizedLinear的参数合并。
def merge_lora(model):
    for mod in model.modules():
        if isinstance(mod, LoraQuantizedLinear):
            mod.merge()
    return model


# 搜索模型的所有模块，再搜索它的直接子模块，如果任何东西是LoraQuantizedLinear，就把它替换回QuantizedLinear。
def detach_lora(model):
    if not model.lora_attached: return model

    for mod in model.modules():
        for name in dir(mod):
            submod = getattr(mod, name, None)
            if not isinstance(submod, LoraQuantizedLinear):
                continue
            new_submod = QuantizedLinear.from_params(
                submod.weight_bit_width,
                submod.weight,
                submod.weight_scale,
                submod.bias,
            )
            setattr(mod, name, new_submod)

    model.lora_attached = False
    return model


