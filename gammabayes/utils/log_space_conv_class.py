import torch
import torch.nn as nn
import torch.nn.functional as F

class LogSumExpConv3D(nn.Module):
    """
    A PyTorch module for performing 3D convolution using the log-sum-exp operation
    without explicit input/output channels. This operates directly on 3D volumes.

    This module requires a kernel tensor to be provided during initialization.
    The aggregation function at each spatial step is logsumexp(input_window_element + kernel_element).

    Args:
        kernel_size (int or tuple): Size of the convolutional kernel.
                                     Can be a single integer for a cubic kernel,
                                     or a tuple (D, H, W) for depth, height, and width.
        kernel_tensor (torch.Tensor): The kernel tensor to be used for convolution.
                                      Its shape must be (K_D, K_H, K_W).
                                      If you want the kernel to be learnable, pass it as an nn.Parameter.
                                      Otherwise, pass a regular torch.Tensor for a fixed kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    """
    def __init__(self, kernel_size, kernel_tensor, stride=1, padding=0):
        super(LogSumExpConv3D, self).__init__()

        # Ensure kernel_size is a tuple for consistent handling
        if isinstance(kernel_size, int):
            _kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            _kernel_size = kernel_size

        # Ensure stride is a tuple
        if isinstance(stride, int):
            _stride = (stride, stride, stride)
        else:
            _stride = stride

        # Ensure padding is a tuple
        if isinstance(padding, int):
            _padding = (padding, padding, padding)
        else:
            _padding = padding

        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding

        # --- Mandatory Kernel Handling ---
        # Validate the shape of the provided kernel tensor
        expected_kernel_shape = _kernel_size
        if kernel_tensor.shape != expected_kernel_shape:
            raise ValueError(
                f"Kernel tensor shape mismatch. Expected {expected_kernel_shape}, "
                f"but got {kernel_tensor.shape}"
            )

        # Register the kernel. If it's an nn.Parameter, it remains learnable.
        # Otherwise, it's registered as a fixed buffer.
        if isinstance(kernel_tensor, nn.Parameter):
            self.weight = kernel_tensor
            print("Using provided learnable kernel (nn.Parameter).")
        else:
            self.register_buffer('weight', kernel_tensor)
            print("Using provided fixed kernel (torch.Tensor).")

        # Bias is removed for this channel-less version for simplicity.
        # If a scalar bias is needed, it can be added manually after the forward pass.


    def _calc_output_dim(self, input_dim, kernel_dim, padding_dim, stride_dim):
        """Calculates a single output dimension."""
        return (input_dim + 2 * padding_dim - kernel_dim) // stride_dim + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the log-sum-exp 3D convolution.

        Args:
            x (torch.Tensor): The input tensor to the convolution.
                              Expected shape: (Batch_size, Depth, Height, Width).

        Returns:
            torch.Tensor: The output tensor after log-sum-exp convolution.
                          Shape: (Batch_size, Output_Depth, Output_Height, Output_Width).
        """
        if x.dim()==3:
            x = x.unsqueeze(0)

        batch_size, in_D, in_H, in_W = x.shape

        out_D = self._calc_output_dim(in_D, self.kernel_size[0], self.padding[0], self.stride[0])
        out_H = self._calc_output_dim(in_H, self.kernel_size[1], self.padding[1], self.stride[1])
        out_W = self._calc_output_dim(in_W, self.kernel_size[2], self.padding[2], self.stride[2])

        # Pad the input tensor. 'constant' padding with -infinity ensures logsumexp works correctly.
        padded_x = F.pad(x, (self.padding[2], self.padding[2],
                              self.padding[1], self.padding[1],
                              self.padding[0], self.padding[0]),
                         mode='constant', value=float('-inf')) # Use -inf for log-domain padding

        s_d, s_h, s_w = padded_x.stride()[-3:] # Get strides for D, H, W from input

        # Target shape for the 'unfolded' tensor:
        # (Batch, out_D, out_H, out_W, K_D, K_H, K_W)
        target_shape = (batch_size, out_D, out_H, out_W,
                        self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])

        # Target strides for the unfolded tensor:
            # For batch: s_batch
            # For out_D, out_H, out_W: This is the input stride * stride
            # For K_D, K_H, K_W: This is the input stride for the kernel
        target_strides = (padded_x.stride()[0], # Batch stride
                          s_d * self.stride[0], # Output D stride
                          s_h * self.stride[1], # Output H stride
                          s_w * self.stride[2], # Output W stride
                          s_d, s_h, s_w)        # Kernel D, H, W strides


        all_windows = padded_x.as_strided(target_shape, target_strides)

        sum_terms = all_windows + self.weight

        output = torch.logsumexp(sum_terms, dim=[-1, -2, -3])

        return output


    def peek(self, norm='log', vmin=None, vmax=None, *args, **kwargs):
        import numpy as np
        from matplotlib import pyplot as plt
        
        real_space_kernel = self.weight.exp()


        fig, ax = plt.subplots(2, 2)
        ax = np.array(ax).flatten()

        ax[0].plot(real_space_kernel.sum(dim=(1,2)))
        ax[0].set(
            xlabel = "Energy [TeV]",
            yscale=norm,
            ylim=[vmin, vmax],
            )
        pcm = ax[1].pcolormesh(real_space_kernel.sum(dim=0).T, norm=norm, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax[1])
        ax[1].set(
            xlabel= "Longitude [deg]",
            ylabel= "Latitude [deg]",
            aspect='equal',
            )

        ax[2].plot(real_space_kernel.sum(dim=(0,2)))
        ax[2].set(
            xlabel = "Longitude [deg]",
            yscale = norm,
            ylim=[vmin, vmax],
            )

        ax[3].plot(real_space_kernel.sum(dim=(0,1)))
        ax[3].set(
            xlabel = "Latitude [deg]",
            yscale=norm,
            ylim=[vmin, vmax],
            )

        plt.tight_layout()
        return fig, ax
