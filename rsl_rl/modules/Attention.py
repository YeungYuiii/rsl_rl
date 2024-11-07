import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F

# Define the Attention mechanism
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        Q = self.query_layer(x)  # (batch_size, sequence_length, hidden_dim)
        K = self.key_layer(x)    # (batch_size, sequence_length, hidden_dim)
        V = self.value_layer(x)  # (batch_size, sequence_length, hidden_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, sequence_length, sequence_length)

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, sequence_length, hidden_dim)
        #return attention_output.mean(dim=1)  # Reduce to (batch_size, hidden_dim)
        return attention_output

# Multi-Head Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear layers for query, key, and value
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # Linear projections
        Q = self.query_layer(x)  # (batch_size, sequence_length, hidden_dim)
        K = self.key_layer(x)    # (batch_size, sequence_length, hidden_dim)
        V = self.value_layer(x)  # (batch_size, sequence_length, hidden_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, -1)

        # Final linear projection
        output = self.fc_out(attention_output)
        return output