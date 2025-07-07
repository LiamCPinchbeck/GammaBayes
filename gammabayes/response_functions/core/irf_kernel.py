import torch
from torch.distributions import Normal


def make_normal_kernel_with_std_and_bins(linear_axis, std, confidence_clearance=5):

    bin_width = linear_axis[1]-linear_axis[0]

    num_bins = int(confidence_clearance*std/bin_width)

    kernel_pseudo_axis = torch.arange(-num_bins*bin_width, (num_bins+1)*bin_width, bin_width)

    base_normal_dist = Normal(loc=torch.tensor(0.), scale=std)


    return base_normal_dist.log_prob(kernel_pseudo_axis).exp()



def make_irf_kernel(geom, log10energy_std, lon_std, lat_std):

    log10_energy_kernel_axis = make_normal_kernel_with_std_and_bins(
        torch.log10(geom.energy_axis), log10energy_std/torch.log10(torch.tensor(torch.e)), confidence_clearance=5)

    longitude_kernel_axis = make_normal_kernel_with_std_and_bins(
        geom.lon_axis, lon_std, confidence_clearance=5)

    latitude_kernel_axis = make_normal_kernel_with_std_and_bins(
        geom.lat_axis, lat_std, confidence_clearance=5)

    irf_kernel = log10_energy_kernel_axis[:, None, None]*longitude_kernel_axis[None, :, None]*latitude_kernel_axis[None, None, :]

    return irf_kernel




