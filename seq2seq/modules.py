import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
        equation:
        \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
        \end{array}
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.einsum("bod, bid->boi", [output, context])

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn, dim=1)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        scored_output = torch.einsum("boi,bid->bod", [attn, context])

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((scored_output, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
