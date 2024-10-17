from torch import nn
import torch
import math
import torch.nn.functional as F


class MLP(nn.Module):
    """The implementation of simple multi-layer perceptron layer
    without dropout and identity connection.

    The feature process order follows `Linear -> ReLU -> Linear -> ReLU -> ...`.

    Args:
        input_dim (int): The input feature dimension.
        hidden_dim (int): The hidden dimension of MLPs.
        output_dim (int): the output feature dimension of MLPs.
        num_layer (int): The number of FC layer used in MLPs.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, return_intermediate=False,
    ) -> torch.Tensor:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.return_intermediate = return_intermediate

    def forward(self, x):
        """Forward function of `MLP`.

        Args:
            x (torch.Tensor): the input tensor used in `MLP` layers.

        Returns:
            torch.Tensor: the forward results of `MLP` layer
        """
        intermediate_res = []
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            intermediate_res.append(x)
        if self.return_intermediate:
            return torch.stack(intermediate_res,dim=0)
        return x

class PositionEmbeddingSine1D(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Forward function for `PositionEmbeddingSine1D`.

        Args:
            text (torch.Tensor): Input text tensor. Shape as `(bs, seq_len, emb_dim)`.

        Returns:
            torch.Tensor: Returned position embedding with shape `(bs, seq_len, num_pos_feats * 2)`.
        """
        pos_len, dim = text.shape[1:]
        assert dim % 2 == 0, "wrong dimension!"
        position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
        # i矩阵
        i_matrix = torch.arange(dim//2, dtype=torch.float)
        i_matrix /= dim / 2
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1 / i_matrix
        i_matrix = i_matrix.to(torch.long)
        # pos矩阵
        pos_vec = torch.arange(pos_len).to(torch.long)
        # 矩阵相乘，pos变成列向量，i_matrix变成行向量
        out = pos_vec[:, None] @ i_matrix[None, :]
        # 奇/偶数列
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        # 赋值
        position_emb[:, 0::2] = emb_sin
        position_emb[:, 1::2] = emb_cos
        return position_emb
    
