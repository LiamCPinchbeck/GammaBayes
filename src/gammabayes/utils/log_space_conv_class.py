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

    def superoldforward(self, x: torch.Tensor) -> torch.Tensor:
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
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        elif x.dim() == 4:
            x = x.unsqueeze(1) # (N, 1, D, H, W)        in_D, in_H, in_W = x.shape
            
        batch_size, channels, in_D, in_H, in_W = x.shape

        out_D = self._calc_output_dim(in_D, self.kernel_size[0], self.padding[0], self.stride[0])
        out_H = self._calc_output_dim(in_H, self.kernel_size[1], self.padding[1], self.stride[1])
        out_W = self._calc_output_dim(in_W, self.kernel_size[2], self.padding[2], self.stride[2])

        # 3. Padding: Use F.pad on (N, C, D, H, W) input
        pad_spec = (self.padding[2], self.padding[2], # W
                    self.padding[1], self.padding[1], # H
                    self.padding[0], self.padding[0]) # D
        padded_x = F.pad(x, pad_spec, mode='constant', value=float('-inf'))
        # We will work with the squeezed version (N, D_pad, H_pad, W_pad)
        padded_x_squeezed = padded_x.squeeze(1)

        # ── Position-based validity volume (the fix for clearance-dependence) ──
        # 0 (= log 1) on real input bins, -inf on padding. Each output bin is
        # renormalised by the kernel weight that actually lands on IN-DOMAIN
        # input (the "live" weight), NOT by the full kernel sum. Near the FoV
        # boundary part of the kernel falls on padding; normalising by the full
        # sum (the old behaviour) silently dropped that weight, and the size of
        # the dropped fraction grew with the kernel — i.e. with the clearance —
        # so changing the IRF clearance shifted the forward model. Renormalising
        # by the live weight makes interior bins identical to before and makes
        # the boundary a well-defined, clearance-stable weighted average.
        #
        # This MUST be position-based: a genuine zero-rate bin is log(0) = -inf
        # in the data, identical in value to padding, so a value-based mask would
        # wrongly drop real empty bins from the normaliser and inflate them.
        valid_vol = x.new_zeros((1, 1, in_D, in_H, in_W))
        padded_valid = F.pad(valid_vol, pad_spec, mode='constant', value=float('-inf'))[0, 0]
        # (D_pad, H_pad, W_pad)

        conved_output = torch.empty(size=(batch_size, out_D, out_H, out_W), 
                                    device=x.device, dtype=x.dtype)
        
        k_d, k_h, k_w = self.kernel_size
        s_d, s_h, s_w = self.stride        

        # Strides for the strided patch views (constant across the D-loop)
        s_n, s_d_pad, s_h_pad, s_w_pad = padded_x_squeezed.stride()
        vs_d, vs_h, vs_w = padded_valid.stride()

        for convi in range(out_D): 
            
            d_start = convi * s_d
            
            # Precompute matrix
            # This step is the log-domain integral and must be done inside the loop.
            log_delta_E_slice = self.log_width_correction[d_start : d_start + k_d]
            # (k_d, 1, 1) + (k_d, k_h, k_w) -> (k_d, k_h, k_w)
            log_delta_E_broadcast = log_delta_E_slice[:, None, None] 
            
            # Apply the log(Delta E) correction. NOTE: the kernel is left
            # UN-normalised here; normalisation is now per-output-bin against the
            # live weight (see below) instead of a single full-window logsumexp.
            precomp_weightmat = log_delta_E_broadcast + self.weight 

            # Reshape precomp_weightmat for broadcasting: (k_d * k_h * k_w)
            # add an N dimension (N, K^3, 1) and will broadcast over the batch
            precomp_weightmat_flat = precomp_weightmat.flatten()[None, :, None] # (1, K^3, 1)

            # Extract Patches for this D-slice over H and W for *all* batches
            # (N, k_d, H_padded, W_padded)
            padded_x_slice = padded_x_squeezed[:, d_start : d_start + k_d, :, :] 
            
            # Target View Shape: (N, k_d, out_H, k_h, out_W, k_w)
            view_shape = (batch_size, k_d, out_H, k_h, out_W, k_w)
            
            # Target Strides: (N stride, D stride, Output H stride, Kernel H stride, Output W stride, Kernel W stride)
            view_strides = (s_n, s_d_pad, 
                            s_h_pad * s_h, s_h_pad, 
                            s_w_pad * s_w, s_w_pad)
            
            all_patches_view = padded_x_slice.as_strided(view_shape, view_strides)
            
            # Reshape to (N, K_D, K_H, K_W, out_H, out_W) -> (N, K^3, out_H * out_W)
            all_patches = all_patches_view.permute(0, 1, 3, 5, 2, 4).flatten(1, 3).flatten(2, 3)
            # Output Shape: (N, K^3, out_H * out_W)

            # Matching validity patches (no batch dim): (1, K^3, out_H * out_W)
            valid_slice = padded_valid[d_start : d_start + k_d, :, :]
            v_view_shape = (k_d, out_H, k_h, out_W, k_w)
            v_view_strides = (vs_d, vs_h * s_h, vs_h, vs_w * s_w, vs_w)
            valid_patches = valid_slice.as_strided(v_view_shape, v_view_strides) \
                                       .permute(0, 2, 4, 1, 3).flatten(0, 2).flatten(1, 2)[None]
            # (1, K^3, out_H * out_W)  -- 0 where the tap is in-domain, -inf on padding

            # 6. Log-Sum-Exp Operation
            # Numerator:  log Σ_j W_j x_{i+j}.  Padding (and genuine zero-rate)
            # taps are -inf in all_patches, so they contribute exp(-inf)=0.
            numerator = torch.logsumexp(all_patches + precomp_weightmat_flat, dim=1)  # (N, out_H*out_W)

            # Denominator: log Σ_{j in-domain} W_j  (live weight; position-based,
            # so real zero-rate bins still count, only padding is excluded).
            log_live_weight = torch.logsumexp(precomp_weightmat_flat + valid_patches, dim=1)  # (1, out_H*out_W)

            logsumexp_result = numerator - log_live_weight  # (N, out_H*out_W)

            # Guard a fully-padded column (no live weight): -inf - -inf -> nan.
            # Set to -inf (zero rate). Cannot occur for valid outputs when
            # padding = kernel_size // 2, but is kept for general stride/padding.
            logsumexp_result = torch.where(
                torch.isneginf(log_live_weight),
                torch.full_like(logsumexp_result, float('-inf')),
                logsumexp_result)
            
            # 7. Store Result: (N, out_H * out_W) -> (N, out_H, out_W)
            conved_output[:, convi, :, :] = logsumexp_result.view(batch_size, out_H, out_W)


        return conved_output.squeeze(1) # Squeeze the single channel dimension if desired



    # def forward(self, x: torch.Tensor, spatial_chunk_size=5) -> torch.Tensor:
    #     # NOTE: if re-enabling this memory-chunked variant, apply the SAME
    #     # live-weight (position-based) normalisation as forward() above:
    #     # build a padded validity volume (0 in-domain, -inf on padding), take the
    #     # matching patches per chunk, and divide the numerator logsumexp by
    #     # logsumexp(precomp_weightmat + valid_patches) instead of subtracting a
    #     # single full-window logsumexp. Otherwise boundary bins become
    #     # clearance-dependent again.
    #     if x.dim() == 3:
    #         x = x.unsqueeze(0).unsqueeze(0)
    #     elif x.dim() == 4:
    #         x = x.unsqueeze(1)
    #     ...
        
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