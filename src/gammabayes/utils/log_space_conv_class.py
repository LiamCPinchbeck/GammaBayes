import torch
import torch.nn as nn
import torch.nn.functional as F
from gammabayes.core import GammaBinning

class LogSumExpConv3D(nn.Module):
    def __init__(self, kernel_size, kernel_tensor:torch.Tensor, binnning_geometry:GammaBinning, stride:int=1, padding:int=0):
        super(LogSumExpConv3D, self).__init__()

        # Ensure kernel_size is a tuple for consistent handling
        if isinstance(kernel_size, int):
            _kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            _kernel_size = kernel_size

        # Make sure that the stride is a tuple
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

        expected_kernel_shape = _kernel_size
        if kernel_tensor.shape != expected_kernel_shape:
            raise ValueError(
                f"Kernel tensor shape mismatch. Expected {expected_kernel_shape}, "
                f"but got {kernel_tensor.shape}"
            )

        # Register the kernel. If it's an nn.Parameter, it remains learnable.
            # Otherwise, it's registered as a fixed buffer. 
            # Would potentially be useful if an analysis wants to investigate
            # systematics involving response functions

        if isinstance(kernel_tensor, nn.Parameter):
            self.weight = kernel_tensor
            print("Using provided learnable kernel (nn.Parameter).")
        else:
            self.register_buffer('weight', kernel_tensor)
            print("Using provided fixed kernel (torch.Tensor).")


        # The actual values of the padding here really don't matter as the 
            # values in the padded inputs should be 0 and hence multiplication with
            # them should result in 0, but just incase someone for whatever reason
            # wants a realtively good padded energy axis here ya go.
            # Presumes that the energy axis in linear in log10
        base_log10energy_axis = torch.log10(binnning_geometry.energy_axis)
        diff = base_log10energy_axis[1]-base_log10energy_axis[0]
        
        begin_energy_pad = torch.tensor([base_log10energy_axis[0] - num*diff for num in range(self.padding[0], 0, -1)])
        end_energy_pad = torch.tensor([base_log10energy_axis[-1] + num*diff for num in range(1, self.padding[0]+1, 1)])

        self.padded_energy_axis = 10**torch.concatenate((begin_energy_pad, base_log10energy_axis, end_energy_pad))

        print("self.padded_energy_axis shape: ", self.padded_energy_axis.shape)
        
        # Estimate Delta E based on the center and the constant log10 width
        self.log_width_correction = (self.padded_energy_axis * torch.log(torch.tensor(10.0)) * diff).log()

        # The correction term for the input
        self.register_buffer('log_delta_E', self.log_width_correction) 


    def _calc_output_dim(self, input_dim, kernel_dim, padding_dim, stride_dim):
        """Calculates a single output dimension."""
        return (input_dim + 2 * padding_dim - kernel_dim) // stride_dim + 1

    def oldforward(self, x: torch.Tensor) -> torch.Tensor:
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial steps remain
        x = x.squeeze()
        in_D, in_H, in_W = x.shape

        out_D = self._calc_output_dim(in_D, self.kernel_size[0], self.padding[0], self.stride[0])
        out_H = self._calc_output_dim(in_H, self.kernel_size[1], self.padding[1], self.stride[1])
        out_W = self._calc_output_dim(in_W, self.kernel_size[2], self.padding[2], self.stride[2])

        padded_x = F.pad(x.unsqueeze(0).unsqueeze(0), # Add N and C dimensions (1, 1, D, H, W)
                         (self.padding[2], self.padding[2],
                          self.padding[1], self.padding[1],
                          self.padding[0], self.padding[0]),
                         mode='constant', value=float('-inf')).squeeze() # Remove N, C

        conved_output = torch.empty(size=(out_D, out_H, out_W), device=x.device, dtype=x.dtype)
        
        k_d, k_h, k_w = self.kernel_size
        s_d, s_h, s_w = self.stride
        
        
        for convi in range(out_D): 
            
            d_start = convi * s_d
            # Slice correction factor
            log_delta_E_slice = self.log_width_correction[d_start : d_start + k_d]
            log_delta_E_broadcast = log_delta_E_slice[:, None, None]
            precomp_weightmat = log_delta_E_broadcast + self.weight

            precomp_weightmat -= torch.logsumexp(precomp_weightmat, dim=(0, 1, 2))
            
            # Reshape precomp_weightmat for broadcasting (k_d * k_h * k_w)
            precomp_weightmat_flat = precomp_weightmat.flatten()[:, None] 

            # --- Extract Patches for this D-slice over H and W ---
            padded_x_slice = padded_x[d_start : d_start + k_d, :, :] # (k_d, H_padded, W_padded)
            
            view_shape = (k_d, out_H, k_h, out_W, k_w)
            
            s_p = padded_x_slice.stride()
            view_strides = (s_p[0], s_p[1] * s_h, s_p[1], s_p[2] * s_w, s_p[2])
            
            all_patches_view = padded_x_slice.as_strided(view_shape, view_strides)
            all_patches = all_patches_view.permute(0, 2, 4, 1, 3).flatten(0, 2).flatten(1) # (K^3, out_H * out_W)

            sum_terms = all_patches + precomp_weightmat_flat
            
            logsumexp_result = torch.logsumexp(sum_terms, dim=0) # (out_H * out_W)
            
            # Store result: (out_H * out_W) -> (out_H, out_W)
            conved_output[convi, :, :] = logsumexp_result.view(out_H, out_W)


        return conved_output

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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
