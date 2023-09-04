import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qat.ops import QuantizedTensor
from operations.amm_conv2d import AMMConv2d
import torchsnooper
from einops import rearrange


# +
class AMMLinear(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_features,
        out_features,
        bias,
        k=16
    ):
        super().__init__()
        self.ncodebooks = ncodebooks
        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % self.ncodebooks == 0
        self.subvec_len = self.in_features // self.ncodebooks
        self.k = k

        self.register_parameter(
            "centroids",
            nn.Parameter(torch.randn(self.ncodebooks, self.k, self.subvec_len))
        )
        self.register_parameter(
            "weight",
            nn.Parameter(torch.randn(
                self.ncodebooks, self.subvec_len, self.out_features
            ))
        )
        self.register_parameter(
            "inverse_temperature_logit",
            nn.Parameter(torch.randn(1))
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_features))
            )
        else:
            self.register_parameter('bias', None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    def _quantize_lut(lut: torch.Tensor) -> QuantizedTensor:
        # Quantize weight to -128 ~ 127
        a = lut.min()
        b = lut.max()
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))

        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))

        q = torch.clamp(
            lut / s + z,
            torch.tensor(-128).to(lut.device),
            torch.tensor(127).to(lut.device)
        ).round().to(torch.int8)
        return QuantizedTensor(q, s, z)
    # @torchsnooper.snoop()
    def _forward(self, x, quantized_lut):
        # batch_size = x.shape[0]
        # x = x.reshape(batch_size, self.ncodebooks, self.subvec_len)
        
        shape = x.shape[:-1]
        # x = x.reshape(np.prod(shape), self.ncodebooks, self.subvec_len)
        x = rearrange(x, 'b n (c v) -> (b n) c v', c=self.ncodebooks, v=self.subvec_len)
        x = x.permute(1, 0, 2)
        # x = rearrange(x, 'b c v -> c b v')
        
        dist = torch.cdist(x, self.centroids) # (c, b, v)å(c, k, v)->(c, b, k)
        # similarity = torch.bmm(x, self.centroids.transpose(1, 2)) # angle-based (c, b, k)
        
        
        lut = torch.bmm(self.centroids, self.weight) # (c, k, v)@(c, v, m)->(c, k, m)
        attention = F.softmax(-dist * (F.softplus(self.inverse_temperature_logit) + 1), dim=-1) # (c, b, k)
        # attention = F.softmax(similarity * (F.softplus(self.inverse_temperature_logit) + 1), dim=-1)
        real_output = torch.bmm(attention, lut).sum(0) # (c, b, k)@(c, k, m)->(c, b, m)->(b, m)
        
        one_hot = F.one_hot(dist.argmin(dim=-1), num_classes=self.k).float()
        # one_hot = F.one_hot(similarity.argmax(dim=-1), num_classes=self.k).float()
        quantized_output = torch.bmm(one_hot, quantized_lut.dequantize()).sum(0)
        
        
        output = real_output - (real_output - quantized_output).detach()
        # (b, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)
        # return output.reshape(batch_size, self.out_features)

    def forward(self, x):
        with torch.no_grad():
            fused_lut = torch.bmm(self.centroids, self.weight) # fused_lut is the same as
            quantized_lut = AMMConv2d._quantize_lut(fused_lut)

        return self._forward(x, quantized_lut)


# -

class TrivenLinear(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_features,
        out_features,
        bias,
        k=16
    ):
        super().__init__()
        self.ncodebooks = ncodebooks
        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % self.ncodebooks == 0
        self.subvec_len = self.in_features // self.ncodebooks
        self.k = k

        self.register_parameter(
            "centroids",
            nn.Parameter(torch.randn(self.ncodebooks, self.k, self.subvec_len)) # (c, k, v)
        )
        self.register_parameter(
            "weight",
            nn.Parameter(torch.randn(
                # self.ncodebooks, self.subvec_len, self.out_features # (c, v, m)
                self.out_features, self.in_features
            ))
        )
        self.register_parameter(
            "inverse_temperature_logit",
            nn.Parameter(torch.randn(1))
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_features))
            )
        else:
            self.register_parameter('bias', None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = tensor<(32, 197, 384), (B, N, D)>
        shape = x.shape[:-1]
        x = rearrange(x, 'b n (c v) -> (b n) c v', c=self.ncodebooks, v=self.subvec_len)
        x = rearrange(x, 'b c v -> c b v') # BN, c, v -> c, BN, v
        dist = torch.cdist(x, self.centroids) # (c, BN, v)å(c, k, v)->(c, BN, k)
        attention = F.softmax(
            -dist * (F.softplus(self.inverse_temperature_logit) + 1),
            dim=-1
        ) # (c, BN, k)
        a = torch.bmm(attention, self.centroids) # (c, BN, k)@(c, k, v)->(c, BN, v)
        x = rearrange(a, 'c b v -> b c v') # BN, c, v -> c, BN, v
        x = rearrange(x, 'b c v -> b (c v)')
        
        
        x = F.linear(x, self.weight, self.bias) # (BN, D)@(D, D)-> (BN, D)
        return x.reshape(*shape, self.out_features) # (B, N, D)

class PQLinear(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_features,
        out_features,
        bias,
        k=16
    ):
        super().__init__()
        self.ncodebooks = ncodebooks
        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % self.ncodebooks == 0
        self.subvec_len = self.in_features // self.ncodebooks
        self.k = k

        self.register_parameter(
            "centroids",
            nn.Parameter(torch.randn(self.ncodebooks, self.k, self.subvec_len))
        )
        self.register_parameter(
            "weight",
            nn.Parameter(torch.randn(
                self.ncodebooks, self.subvec_len, self.out_features
            ))
        )
        self.register_parameter(
            "inverse_temperature_logit",
            nn.Parameter(torch.randn(1))
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_features))
            )
        else:
            self.register_parameter('bias', None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(np.prod(shape), self.ncodebooks, self.subvec_len)
        x = x.permute(1, 0, 2)
        dist = torch.cdist(x, self.centroids)
        # (ncodebooks, b, k)
        attention = F.softmax(
            -dist * (F.softplus(self.inverse_temperature_logit) + 1),
            dim=-1
        )
        # (ncodebooks, b, k)
        lut = torch.bmm(self.centroids, self.weight)
        # (ncodebooks, k, out_features)
        training_output = torch.bmm(attention, lut).sum(0)
        one_hot = F.one_hot(dist.argmin(dim=-1), num_classes=self.k).float()
        inference_output = torch.bmm(one_hot, lut).sum(0)
        output = training_output - (training_output - inference_output).detach()
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)
