import warnings
import numpy as np
from ._data import (
    get_disinfection_table,
    plot_disinfection_data,
    get_full_disinfection_table,
)


class DisinfectionCalculator:
    """
    manage calculation of disinfection rates for a Room object
    """

    def __init__(self, room):
        self.room = room

    def get_disinfection_data(self, zone_id):
        """return the dataframe and violin plot"""
        df, fluence_dict = self.get_disinfection_table(zone_id)
        fig = plot_disinfection_data(df, fluence_dict, self.room)
        return df, fig

    def get_disinfection_plot(self, zone_id):
        """
        Return a violin plot of expected disinfection rates
        """
        df, fluence_dict = self.get_disinfection_table(zone_id)
        fig = plot_disinfection_data(df, fluence_dict, self.room)
        return fig

    def get_disinfection_table(self, zone_id):
        """
        Return a table of expected disinfection rates
        """
        # check if zone_id is in room.calc_zones
        if zone_id in self.room.calc_zones.keys():
            zone = self.room.calc_zones[zone_id]
        else:
            raise KeyError(f"Calc zone {zone_id} is not in the room state.")

        fluence_dict = self._get_fluence_dict(zone)
        if len(fluence_dict) > 0:
            df = get_disinfection_table(fluence=fluence_dict, room=self.room)
            if len(fluence_dict) == 1:  # drop wavelength if unnecessary
                df = df.drop(columns="wavelength [nm]")
            # move some keys arounds
            new_keys = ["Link"] + [key for key in df.keys() if "Link" not in key]
            df = df[new_keys]
        else:
            msg = "Fluence value not available; returning full disinfection data table."
            warnings.warn(msg, stacklevel=3)
            df = get_full_disinfection_table()

        return df, fluence_dict

    def _get_fluence_dict(self, zone):
        """
        get a dict of all the wavelengths and the fluences they contribute
        to in a given calculation zone
        """
        lamp_wavelengths = self._get_lamp_wavelength_dict(zone)
        fluence_dict = {}
        for label, lamp_ids in lamp_wavelengths.items():
            vals = np.zeros(zone.values.shape)
            for lamp_id in lamp_ids:
                if lamp_id in zone.lamp_cache.keys():
                    vals += zone.lamp_cache[lamp_id].values.mean()
            fluence_dict[label] = vals.mean()
        return fluence_dict

    def _get_lamp_wavelength_dict(self, zone):
        """assign lamps to each unique wavelength contributing to the zone"""
        lamp_types = self._get_lamp_types(zone)
        if len(zone.lamp_cache) == 0:
            msg = f"Calc zone {zone.zone_id} has no associated lamps."
            warnings.warn(msg)
        elif len(lamp_types) == 0:
            msg = f"Calc zone {zone.zone_id} has no associated lamps with an associated wavelength."

        lamp_wavelengths = {}
        for wavelength in lamp_types:
            val = [
                lamp.lamp_id
                for lamp in self.room.lamps.values()
                if lamp.wavelength == float(wavelength)
            ]
            lamp_wavelengths[wavelength] = val
        return lamp_wavelengths

    def _get_lamp_types(self, zone):
        """fetch a list of unique wavelengths contributing to the zone"""
        wavelengths = []
        for lamp_id in zone.lamp_cache.keys():
            if lamp_id in self.room.lamps.keys():
                lamp = self.room.lamps[lamp_id]
                if lamp.wavelength is not None:
                    wavelengths.append(lamp.wavelength)
                else:
                    msg = f"{lamp.name} ({lamp_id}) has an undefined wavelength. Its fluence contribution will not be counted."
                    warnings.warn(msg, stacklevel=3)
            else:
                msg = f"{lamp_id} has been removed from the room"
                warnings.warn(msg, stacklevel=3)
        return np.unique(wavelengths)
