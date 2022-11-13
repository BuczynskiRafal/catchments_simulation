"""Package include method for simulate subcatchment with different
features values from Storm Water Management Model"""
import os
import swmmio
import pandas as pd
import numpy as np
from pyswmm import Simulation, Subcatchments


class FeaturesSimulation:
    """Class FeaturesSimulation contains all methods."""

    def __init__(self, subcatchment_id, raw_file):
        self.raw_file = raw_file
        self.subcatchment_id = subcatchment_id
        self.file = FeaturesSimulation.copy_file(self, copy=self.raw_file)
        self.model = swmmio.Model(self.file)

    def copy_file(self, copy=None, suffix="copy"):
        """
        > This function takes a SWMM input file and creates a copy of it with a suffix added to the end of the file name

        :param copy: the path to the file you want to copy. If you don't specify one, it will use the raw_file
        :param suffix: the suffix to add to the end of the file name, defaults to copy (optional)
        :return: The new path of the copied file.
        """
        if copy is None:
            copy = self.raw_file
        baseline = swmmio.Model(copy)
        new_path = os.path.join(baseline.inp.name + "_" + suffix + ".inp")
        baseline.inp.save(new_path)
        return new_path

    def get_section(self, section: str = "subcatchments") -> pd.DataFrame:
        """
        > The function `get_section` takes a SWMM input file and returns a pandas dataframe of the specified section

        :param section: the name of the section you want to get, defaults to subcatchments (optional)
        :type section: str (optional)
        :return: The section of the inp file.
        """
        return getattr(swmmio.Model(self.file).inp, section)

    def calculate(self) -> dict:
        """
        > The function `calculate` takes a `self` argument, and returns the
        statistics of the subcatchment with the ID `self.subcatchment_id` in the SWMM model file.
        :return: The statistics of the subcatchment.
        """
        with Simulation(self.file) as sim:
            subcatchment = Subcatchments(sim)[self.subcatchment_id]
            for _ in sim:
                pass
            return subcatchment.statistics

    def simulate_subcatchment(
        self, feature, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        > This function takes a SWMM model, a subcatchment ID,
        and a feature (e.g. percent impervious) and runs the model
        for a range of values for that feature
        The function returns a `pandas.DataFrame` with catchment statistics and analysed feature

        :param feature: the name of the parameter you want to change
        :param start: the starting value of the parameter, defaults to 0 (optional), defaults to 0
        :type start: float (optional)
        :param stop: the maximum value of the parameter you want to test, defaults to 100 (optional), defaults to 100
        :type stop: float (optional)
        :param step: the step size for the simulation, defaults to 10 (optional), defaults to 10
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        self.file = self.copy_file(self.raw_file)
        catchment_data = {
            "runoff": [],
            "peak_runoff_rate": [],
            "infiltration": [],
            "evaporation": [],
        }
        percent_impervious = np.arange(start, stop + 0.001, step)

        for percent in percent_impervious:
            # subcatchment = swmmio.utils.dataframes.dataframe_from_inp(self.file, '[SUBCATCHMENTS]')
            subcatchment = self.model.inp.subcatchments
            subcatchment.loc[self.subcatchment_id, feature] = percent
            swmmio.utils.modify_model.replace_inp_section(
                self.model.inp.path, "[SUBCATCHMENTS]", subcatchment
            )
            subcatchemnt_stats = self.calculate()
            for key in catchment_data:
                catchment_data[key].append(subcatchemnt_stats[key])
        catchment_data[feature] = percent_impervious
        return pd.DataFrame(data=catchment_data)

    def simulate_area(
        self, start: float = 1, stop: float = 10, step: float = 1
    ) -> pd.DataFrame:
        """
        This function simulates the area of the subcatchment in selected range of area

        :param start: the starting value of the parameter to be varied, defaults to 1 (optional), defaults to 1
        :type start: float (optional)
        :param stop: The maximum value of the parameter to be simulated, defaults to 10 (optional), defaults to 10
        :type stop: float (optional)
        :param step: the step size for the simulation, defaults to 1 (optional), defaults to 1
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="Area", start=start, stop=stop, step=step
        )

    def simulate_percent_impervious(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        This function simulates the percent impervious of a subcatchment and returns a dataframe with the results

        :param start: the starting value for the parameter, defaults to 0 (optional), defaults to 0
        :type start: float (optional)
        :param stop: the maximum value of the parameter to simulate, defaults to 100 (optional), defaults to 100
        :type stop: float (optional)
        :param step: the step size for the simulation, defaults to 10 (optional), defaults to 10
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="PercImperv", start=start, stop=stop, step=step
        )

    def simulate_percent_slope(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        > This function simulates the subcatchment's percent slope in default
        from 0 to 100 percent in increments of 10 percent

        :param start: the starting value of the parameter, defaults to 0
        :type start: float (optional)
        :param stop: the maximum value of the parameter to simulate, defaults to 100
        :type stop: float (optional)
        :param step: the step size for the range of values to simulate, defaults to 10
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="PercSlope", start=start, stop=stop, step=step
        )

    def simulate_width(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        > This function simulates the width of the subcatchment and returns a dataframe with the results

        :param start: the starting value of the parameter, defaults to 0
        :type start: float (optional)
        :param stop: the maximum value of the parameter to be simulated, defaults to 100
        :type stop: float (optional)
        :param step: the step size for the simulation, defaults to 10
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="Width", start=start, stop=stop, step=step
        )

    def simulate_curb_length(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        > This function simulates the curb length of a subcatchment

        :param start: the starting value of the parameter, defaults to 0
        :type start: float (optional)
        :param stop: the maximum value of the parameter to simulate, defaults to 100
        :type stop: float (optional)
        :param step: the step size for the simulation, defaults to 10
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="CurbLength", start=start, stop=stop, step=step
        )

    def simulate_manning_n(self, param: str) -> pd.DataFrame:
        """
        > This function takes a SWMM model file and a subcatchment ID and returns a dataframe with the results
        of running the model with different Manning's n values

        :param param: type feature name
        :type param: str
        :return: A dataframe with the results of the simulation.
        """
        self.file = self.copy_file(self.raw_file)
        # Source: McCuen, R. et al. (1996), Hydrology, FHWA-SA-96-067, Federal Highway Administration, Washington, DC.
        manning_n = np.sort(
            [
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
                0.024,
                0.05,
                0.06,
                0.17,
                0.13,
                0.15,
                0.24,
                0.41,
                0.4,
                0.8,
            ]
        )

        catchment_data = {
            "runoff": [],
            "peak_runoff_rate": [],
            "infiltration": [],
            "evaporation": [],
        }
        for n in manning_n:
            subareas = self.model.inp.subareas
            subareas.loc[self.subcatchment_id, "N-" + param] = n
            self.model.inp.subareas = subareas
            swmmio.utils.modify_model.replace_inp_section(
                self.file, "[SUBAREAS]", subareas
            )
            catchment_stats = self.calculate()
            for key in catchment_data:
                catchment_data[key].append(catchment_stats[key])
        catchment_data["N-" + param] = manning_n
        return pd.DataFrame(data=catchment_data)

    def simulate_n_imperv(self) -> pd.DataFrame:
        """
        > This function simulates the Manning's n for impervious area
        :return: A dataframe with the simulated values of Manning's n for the impervious area.
        """
        return self.simulate_manning_n(param="Imperv")

    def simulate_n_perv(self) -> pd.DataFrame:
        """
        > This function simulates the Manning's n for the pervious area.
        :return: A dataframe with the simulated values of Manning's n for the pervious area.
        """
        return self.simulate_manning_n(param="Perv")

    def simulate_destore(self, param: str) -> pd.DataFrame:
        """
        > This function takes a parameter name for depth of depression storage on area, and then runs the model
        for each typical value.

        :param param: "N-Imperv"
        :type: str
        :return: A dataframe with the following columns:
            - runoff
            - peak_runoff_rate
            - infiltration
            - evaporation
            - Destore-param
        """
        self.file = self.copy_file(self.raw_file)
        """Source: ASCE,(1992), Design & Construction of Urban Stormwater Management Systems, New York, NY."""
        typical_values = [0.05, 0.1, 0.2, 0.3]  # Inches
        typical_values = [val * 25.4 for val in typical_values]  # SI units [mm]
        catchment_data = {
            "runoff": [],
            "peak_runoff_rate": [],
            "infiltration": [],
            "evaporation": [],
        }
        for n in typical_values:
            subareas = self.model.inp.subareas
            subareas.loc[self.subcatchment_id, "S-" + param] = n
            self.model.inp.subareas = subareas
            swmmio.utils.modify_model.replace_inp_section(
                self.file, "[SUBAREAS]", subareas
            )
            catchment_stats = self.calculate()
            for key in catchment_data:
                catchment_data[key].append(catchment_stats[key])
        df = pd.DataFrame(data=catchment_data)
        df["Destore-" + param] = typical_values
        return df

    def simulate_s_imperv(self) -> pd.DataFrame:
        """
        > This function simulates the impervious depth of depression storage on area.
        :return: A dataframe with the simulated values of the impervious surface area.
        """
        return self.simulate_destore(param="Imperv")

    def simulate_s_perv(self) -> pd.DataFrame:
        """
        > This function simulates the pervious depth of depression storage on area.
        :return: A dataframe with the simulated values of the S_perv variable.
        """
        return self.simulate_destore(param="Perv")

    def simulate_percent_zero_imperv(
        self, start: float = 0, stop: float = 100, step: float = 10
    ):
        """
        > This function runs a series of simulations, each with a different
        percent impervious area with no depression,
        and returns a dataframe with the results

        :param start: the starting value for the percent impervious, defaults to 0
        :type start: float (optional)
        :param stop: the maximum percent impervious to test, defaults to 100
        :type stop: float (optional)
        :param step: the step size for the percent impervious, defaults to 10
        :type step: float (optional)
        :return: A dataframe with the results of the simulation.
        """
        self.file = self.copy_file(self.raw_file)
        percent_impervious = np.arange(start, stop + 0.001, step)
        catchment_data = {
            "runoff": [],
            "peak_runoff_rate": [],
            "infiltration": [],
            "evaporation": [],
        }
        for percent in percent_impervious:
            subareas = self.model.inp.subareas
            subareas.loc[self.subcatchment_id, "PctZero"] = percent
            self.model.inp.subareas = subareas
            swmmio.utils.modify_model.replace_inp_section(
                self.file, "[SUBAREAS]", subareas
            )
            catchment_stats = self.calculate()
            for key in catchment_data:
                catchment_data[key].append(catchment_stats[key])
        catchment_data["Zero-Imperv"] = percent_impervious
        return pd.DataFrame(data=catchment_data)
