from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)

import inspect




class DMProfiles:
    """Class containing predefined dark matter profiles."""
    NFW         = profiles.NFWProfile()
    Einasto     = profiles.EinastoProfile()
    Burkert     = profiles.BurkertProfile()
    Isothermal  = profiles.IsothermalProfile()
    Moore       = profiles.MooreProfile()


str_to_gammapy_profile_dict = {
    'einasto':DMProfiles.Einasto,
    'nfw':DMProfiles.NFW,
    'burkert':DMProfiles.Burkert,
    'isothermal':DMProfiles.Isothermal,
    'moore':DMProfiles.Moore,
}

def convert_str_profile_to_DMProfiles(profile_string):
    """
    Converts a string representing a profile name to a DMProfiles class instance.

    Args:
        profile_string (str): The name of the profile to convert.

    Returns:
        profiles.DMProfile: The corresponding Gammapy dark matter profile.

    Raises:
        KeyError: If the profile string does not match any known profile.
    """
    profile_string = profile_string.lower()

    return str_to_gammapy_profile_dict[profile_string]

def check_profile_module(profile):
    """
    Checks if the provided profile is a valid Gammapy dark matter profile.

    Args:
        profile (str or profiles.DMProfile): The profile to check.

    Returns:
        profiles.DMProfile: The validated dark matter profile.

    Raises:
        Exception: If the profile is not a valid Gammapy dark matter profile.
    """
    try:
        if type(profile)==str:
            profile = convert_str_profile_to_DMProfiles(profile)

        assert inspect.getmodule(profile) == profiles

        return profile

    except:
        raise Exception(f"Invalid dark matter density profile. \nPlease provide a string contained in the following list {str_to_gammapy_profile_dict.keys()} \n or a profile from the gammapy.astro.darkmatter.profiles module.")


