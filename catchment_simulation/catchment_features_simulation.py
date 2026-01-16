"""Package include method for simulate subcatchment with different
features values from Storm Water Management Model"""
import os
import swmmio
import pandas as pd
import numpy as np
from pyswmm import Simulation, Subcatchments


class FeaturesSimulation:
    """
    A class to simulate subcatchments with different features using the Storm Water Management Model (SWMM).

    Parameters
    ----------
    subcatchment_id : str
        The identifier of the subcatchment being simulated.
    raw_file : str
        The path to the raw SWMM input file.
    """

    RESULT_KEYS: tuple[str, ...] = ("runoff", "peak_runoff_rate", "infiltration", "evaporation")

    # Source: McCuen, R. et al. (1996), Hydrology, FHWA-SA-96-067, Federal Highway Administration, Washington, DC.
    MANNING_N_VALUES: tuple[float, ...] = tuple(sorted([
        0.011, 0.012, 0.013, 0.014, 0.015, 0.024, 0.05, 0.06,
        0.17, 0.13, 0.15, 0.24, 0.41, 0.4, 0.8,
    ]))

    # Source: ASCE (1992), Design & Construction of Urban Stormwater Management Systems, New York, NY.
    # Values in mm (converted from inches: 0.05, 0.1, 0.2, 0.3)
    DEPRESSION_STORAGE_VALUES: tuple[float, ...] = tuple(val * 25.4 for val in [0.05, 0.1, 0.2, 0.3])

    @staticmethod
    def _create_result_dict() -> dict[str, list]:
        """Create an empty result dictionary with standard keys."""
        return {key: [] for key in FeaturesSimulation.RESULT_KEYS}

    def __init__(self, subcatchment_id: str, raw_file: str) -> None:
        self.raw_file = raw_file
        self.subcatchment_id = subcatchment_id
        self._temp_files: list[str] = []
        self.file = self.copy_file(copy=self.raw_file)
        self.model = swmmio.Model(self.file)

    def __enter__(self) -> "FeaturesSimulation":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup_temp_files()

    def _cleanup_temp_files(self) -> None:
        """Remove all temporary files created during simulation."""
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        self._temp_files.clear()

    @staticmethod
    def _validate_simulation_params(start: float, stop: float, step: float) -> None:
        """Validate simulation parameters."""
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")
        if start > stop:
            raise ValueError(f"start ({start}) must be <= stop ({stop})")

    def copy_file(self, copy: str = None, suffix: str = "copy") -> str:
        """
        Create a copy of a SWMM input file with a suffix added to the end of the file name.

        Parameters
        ----------
        copy : str, optional
            The path to the file you want to copy. If not specified, it will use the raw_file.
        suffix : str, optional
            The suffix to add to the end of the file name, defaults to 'copy'.

        Returns
        -------
        str
            The new path of the copied file.
        """
        if copy is None:
            copy = self.raw_file
        baseline = swmmio.Model(copy)
        new_path = os.path.join(baseline.inp.name + "_" + suffix + ".inp")
        baseline.inp.save(new_path)
        self._temp_files.append(new_path)
        return new_path

    def get_section(self, section: str = "subcatchments") -> pd.DataFrame:
        """
        Get a specified section from a SWMM input file as a pandas DataFrame.

        Parameters
        ----------
        section : str, optional
            The name of the section you want to get, defaults to 'subcatchments'.

        Returns
        -------
        pd.DataFrame
            The section of the inp file as a pandas DataFrame.
        """
        return getattr(swmmio.Model(self.file).inp, section)

    def calculate(self) -> dict:
        """
        Run a simulation using the SWMM model and return the statistics of the subcatchment with the
        ID `self.subcatchment_id`.

        Returns
        -------
        dict
            The statistics of the subcatchment.
        """
        with Simulation(self.file) as sim:
            subcatchment = Subcatchments(sim)[self.subcatchment_id]
            for _ in sim:
                pass
            return subcatchment.statistics

    def simulate_subcatchment(
        self, feature: str, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        Simulate a subcatchment with varying feature values using the SWMM model.

        Parameters
        ----------
        feature : str
            The name of the parameter to be varied in the simulation.
        start : float, optional
            The starting value of the parameter, defaults to 0.
        stop : float, optional
            The maximum value of the parameter to be simulated, defaults to 100.
        step : float, optional
            The step size for the simulation, defaults to 10.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        self._validate_simulation_params(start, stop, step)
        self.file = self.copy_file(self.raw_file)
        catchment_data = self._create_result_dict()
        scope = np.arange(start, stop + 0.001, step)

        for percent in scope:
            # subcatchment = swmmio.utils.dataframes.dataframe_from_inp(self.file, '[SUBCATCHMENTS]')
            subcatchment = self.model.inp.subcatchments
            subcatchment.loc[self.subcatchment_id, feature] = percent
            swmmio.utils.modify_model.replace_inp_section(
                self.model.inp.path, "[SUBCATCHMENTS]", subcatchment
            )
            subcatchment_stats = self.calculate()
            for key in catchment_data:
                catchment_data[key].append(subcatchment_stats[key])
        catchment_data[feature] = scope
        return pd.DataFrame(data=catchment_data)

    def simulate_area(
        self, start: float = 1, stop: float = 10, step: float = 1
    ) -> pd.DataFrame:
        """
        Simulate the area of the subcatchment within a specified range of values.

        Parameters
        ----------
        start : float, optional
            The starting value of the area to be varied, defaults to 1.
        stop : float, optional
            The maximum value of the area to be simulated, defaults to 10.
        step : float, optional
            The step size for the simulation, defaults to 1.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="Area", start=start, stop=stop, step=step
        )

    def simulate_percent_impervious(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        Simulate the percent impervious of a subcatchment within a specified range of values.

        Parameters
        ----------
        start : float, optional
            The starting value of the percent impervious to be varied, defaults to 0.
        stop : float, optional
            The maximum value of the percent impervious to be simulated, defaults to 100.
        step : float, optional
            The step size for the simulation, defaults to 10.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="PercImperv", start=start, stop=stop, step=step
        )

    def simulate_percent_slope(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        Simulate the percent slope of a subcatchment within a specified range of values.

        Parameters
        ----------
        start : float, optional
            The starting value of the percent slope to be varied, defaults to 0.
        stop : float, optional
            The maximum value of the percent slope to be simulated, defaults to 100.
        step : float, optional
            The step size for the simulation, defaults to 10.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="PercSlope", start=start, stop=stop, step=step
        )

    def simulate_width(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        Simulate the width of a subcatchment within a specified range of values.

        Parameters
        ----------
        start : float, optional
            The starting value of the width to be varied, defaults to 0.
        stop : float, optional
            The maximum value of the width to be simulated, defaults to 100.
        step : float, optional
            The step size for the simulation, defaults to 10.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="Width", start=start, stop=stop, step=step
        )

    def simulate_curb_length(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        Simulate the curb length of a subcatchment within a specified range of values.

        Parameters
        ----------
        start : float, optional
            The starting value of the curb length to be varied, defaults to 0.
        stop : float, optional
            The maximum value of the curb length to be simulated, defaults to 100.
        step : float, optional
            The step size for the simulation, defaults to 10.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        return self.simulate_subcatchment(
            feature="CurbLength", start=start, stop=stop, step=step
        )

    def simulate_manning_n(self, param: str) -> pd.DataFrame:
        """
        Simulate a subcatchment using various Manning's n values.

        Parameters
        ----------
        param : str
            The name of the feature for which Manning's n values should be varied.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the simulation.
        """
        self.file = self.copy_file(self.raw_file)
        catchment_data = self._create_result_dict()
        for n in self.MANNING_N_VALUES:
            subareas = self.model.inp.subareas
            col = "N-" + param
            subareas[col] = subareas[col].astype(float)
            subareas.loc[self.subcatchment_id, col] = n
            self.model.inp.subareas = subareas
            swmmio.utils.modify_model.replace_inp_section(
                self.file, "[SUBAREAS]", subareas
            )
            catchment_stats = self.calculate()
            for key in catchment_data:
                catchment_data[key].append(catchment_stats[key])
        catchment_data["N-" + param] = self.MANNING_N_VALUES
        return pd.DataFrame(data=catchment_data)

    def simulate_n_imperv(self) -> pd.DataFrame:
        """
        Simulate Manning's n for impervious area.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the simulated values of Manning's n for the impervious area.
        """
        return self.simulate_manning_n(param="Imperv")

    def simulate_n_perv(self) -> pd.DataFrame:
        """
        Simulate Manning's n for pervious area.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the simulated values of Manning's n for the pervious area.
        """
        return self.simulate_manning_n(param="Perv")

    def simulate_destore(self, param: str) -> pd.DataFrame:
        """
        Simulate the model for various depths of depression storage on the given area.

        Parameters
        ----------
        param : str
            The name of the feature for which the depth of depression storage should be varied.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the following columns:
            - runoff
            - peak_runoff_rate
            - infiltration
            - evaporation
            - Destore-param
        """
        self.file = self.copy_file(self.raw_file)
        catchment_data = self._create_result_dict()
        for n in self.DEPRESSION_STORAGE_VALUES:
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
        df["Destore-" + param] = self.DEPRESSION_STORAGE_VALUES
        return df

    def simulate_s_imperv(self) -> pd.DataFrame:
        """
        Simulate the impervious depth of depression storage on area.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the simulated values of the impervious surface area.
        """
        return self.simulate_destore(param="Imperv")

    def simulate_s_perv(self) -> pd.DataFrame:
        """
        Simulate the pervious depth of depression storage on area.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the simulated values of the S_perv variable.
        """
        return self.simulate_destore(param="Perv")

    def simulate_percent_zero_imperv(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """
        Run a series of simulations with different percentages of impervious area with no depression storage.

        Parameters
        ----------
        start : float, optional
            The starting value for the percent impervious, defaults to 0.
        stop : float, optional
            The maximum percent impervious to test, defaults to 100.
        step : float, optional
            The step size for the percent impervious, defaults to 10.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the results of the simulation.
        """
        self._validate_simulation_params(start, stop, step)
        self.file = self.copy_file(self.raw_file)
        percent_impervious = np.arange(start, stop + 0.001, step)
        catchment_data = self._create_result_dict()
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
