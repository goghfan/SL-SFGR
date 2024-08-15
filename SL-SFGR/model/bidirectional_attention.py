import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossAttention3DModule(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention3DModule, self).__init__()

        # Assuming in_channels is the number of input channels for A and B
        self.conv_query_a = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_key_b = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_value_b = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.conv_query_b = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_key_a = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_value_a = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, input_a, input_b):
        # Calculate attention weights when A is query and B is key-value
        query_a = self.conv_query_a(input_a)
        key_b = self.conv_key_b(input_b)
        value_b = self.conv_value_b(input_b)

        attention_weights_a_to_b = F.softmax(torch.matmul(query_a.view(query_a.size(0), -1), key_b.view(key_b.size(0), -1).T), dim=-1)
        output_a_to_b = torch.matmul(attention_weights_a_to_b, value_b.view(value_b.size(0), -1)).view(value_b.size())

        # Calculate attention weights when B is query and A is key-value
        query_b = self.conv_query_b(input_b)
        key_a = self.conv_key_a(input_a)
        value_a = self.conv_value_a(input_a)

        attention_weights_b_to_a = F.softmax(torch.matmul(query_b.view(query_b.size(0), -1), key_a.view(key_a.size(0), -1).T), dim=-1)
        output_b_to_a = torch.matmul(attention_weights_b_to_a, value_a.view(value_a.size(0), -1)).view(value_a.size())

        return output_a_to_b.to(torch.float32) , output_b_to_a.to(torch.float32) 



if __name__ == '__main__':
    # Example usage
    in_channels = 4 # You can change this based on your input channels
    cross_attention_module = CrossAttention3DModule(in_channels).to(torch.float16).to('cuda:4')

    # Assuming input_a and input_b are your input tensors with the same shape
    input_a = torch.randn(2, 4, 192,160,192).to(torch.float16).to('cuda:4')
    input_b = torch.randn(2, 4, 192,160,192).to(torch.float16).to('cuda:4')
    output_a_to_b, output_b_to_a = cross_attention_module(input_a, input_b)


    in_channels = 4  # You can change this based on your input channels
    cross_attention_module2 = CrossAttention3DModule(in_channels).to(torch.float16).to('cuda:4')

    # Assuming input_a and input_b are your input tensors with the same shape
    input_a = torch.randn(2, in_channels, 96,80,96).to(torch.float16).to('cuda:4')
    input_b = torch.randn(2, in_channels, 96,80,96).to(torch.float16).to('cuda:4')

    output_a_to_b, output_b_to_a = cross_attention_module2(input_a, input_b)

    print("Output shape (A to B):", output_a_to_b.shape)
    print("Output shape (B to A):", output_b_to_a.shape)
