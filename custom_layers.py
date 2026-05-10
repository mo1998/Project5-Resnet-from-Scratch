import torch
import torch.nn as nn
import math

class CustomLinearFunction(torch.autograd.Function):
    """
    Hand-written forward and backward pass for a Linear layer.
    Satisfies the 'AI Neutralization' requirement.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Save input and weight for backward pass
        ctx.save_for_backward(input, weight, bias)
        
        # Linear transformation: y = x * W^T + b
        output = input.mm(weight.t())
        if bias is not None:
            # Add bias with broadcasting
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None

        # Chain rule:
        # 1. grad_input = grad_output * W
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        
        # 2. grad_weight = grad_output^T * input
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        
        # 3. grad_bias = sum(grad_output, dim=0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight and bias as Parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Standard Kaiming Uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weight, self.bias)

if __name__ == "__main__":
    # Verification script
    print("Testing CustomLinear Layer...")
    x = torch.randn(8, 16, requires_grad=True)
    custom_layer = CustomLinear(16, 10)
    
    # Forward Pass
    out_custom = custom_layer(x)
    
    # Backward Pass
    loss = out_custom.pow(2).sum()
    loss.backward()
    
    print(f"Forward pass output shape: {out_custom.shape}")
    print(f"Weight gradient shape: {custom_layer.weight.grad.shape}")
    
    # Compare with standard PyTorch implementation
    std_layer = nn.Linear(16, 10)
    std_layer.weight.data = custom_layer.weight.data.clone()
    std_layer.bias.data = custom_layer.bias.data.clone()
    
    x_std = x.detach().clone()
    x_std.requires_grad = True
    out_std = std_layer(x_std)
    out_std.pow(2).sum().backward()
    
    # Validation
    fwd_diff = (out_custom - out_std).abs().max().item()
    grad_w_diff = (custom_layer.weight.grad - std_layer.weight.grad).abs().max().item()
    grad_x_diff = (x.grad - x_std.grad).abs().max().item()
    
    print(f"Max Forward Diff: {fwd_diff:.2e}")
    print(f"Max Weight Grad Diff: {grad_w_diff:.2e}")
    print(f"Max Input Grad Diff: {grad_x_diff:.2e}")
    
    if fwd_diff < 1e-6 and grad_w_diff < 1e-6:
        print("SUCCESS: Custom layer matches PyTorch implementation!")
    else:
        print("FAILURE: Differences too large.")
