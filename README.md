[![Documentation Status](https://readthedocs.org/projects/catchments-simulation/badge/?version=latest)](https://catchments-simulation.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/LICENSE)
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.org/project/catchment-simulation/)
[![GitHub Actions Build Status](https://github.com/BuczynskiRafal/catchments_simulation/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/BuczynskiRafal/catchments_simulation/actions/workflows/python-package.yml)
[![GitHub Actions Build Status](https://github.com/BuczynskiRafal/catchments_simulation/actions/workflows/django.yml/badge.svg?branch=master)](https://github.com/BuczynskiRafal/catchments_simulation/actions/workflows/django.yml)



# Catchment simulation
Package include method for simulate subcatchment with different features values from Storm Water Management Model.
Currently, some of the 'catchment simulation' functionality available in the app - [catchment simulation](https://catchment-simulations.onrender.com/)

## Examples of How To Use 

Creating SWMM object for analyse

#### Inslall `catchment_simulation` package using pip 

```bash
pip install catchment-simulation
```

#### Example of simulation subcatchment area in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_area(start=1, stop: = 10, step = 1)
```

More code examples at the end of the notebook.

# Catchment simulation app

The application was built in django share some of the functionality from the 'catchment simulation' package. 

It is designed to analyze and predict water flow behavior in catchments.  The application contain two main components: the Catchment Simulation package and Catchment Calculation.
On the main page you can find information of the 'catchments simulation' package and examples of use. 

Application at - https://catchment-simulations.onrender.com/

<div align="center">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/home.png">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/example.png">
</div>

## Simulations in a web application
The 'Simulations' tab allows the user to upload a file and select components for simulation. Once the simulation is executed, the window will display an interactive graph of the obtained data and a button to download the results in an excel spreadsheet. 

<div align="center">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/simulation_start.png">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/simulation_after.png">
</div>

### Warning
You will be asked to register and log in before performing the simulation. 

## Appendix - ANN and SWMM predictions
The 'Calculations' tab contains a neural network model trained to predict catchment area runoff. The user, after uploading the file, receives the results of calculations performed SWMM and ANN model prediction. 

<div align="center">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/calculations.png">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/ann_model.png">
</div>

### Warning
You will be asked to register and log in before performing the simulation. 

# More examples of package usage 

#### Simulate subcatchment percent impervious in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_percent_impervious(start=1, stop: = 10, step = 1)
```

#### Simulate subcatchment percent slope in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_percent_slope(start=1, stop: = 10, step = 1)
```

#### Simulate subcatchment width in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_width(start=1, stop: = 10, step = 1)
```
#### Simulate subcatchment curb length in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_curb_length(start=1, stop: = 10, step = 1)
```

#### Simulate subcatchment N-Imperv in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_n_imperv(param="Imperv")
```

#### Simulate subcatchment N-Perv in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_n_perv(param="Perv")
```

#### Simulate subcatchment Destore-Imperv in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_s_imperv(param="Imperv")
```

#### Simulate subcatchment Destore-Perv in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_s_imperv(param="Perv")
```

#### Simulate subcatchment Percent Zero Imperv in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_percent_zero_imperv(start=0, stop=100, step=10)
```


# Bugs

If you encounter any bugs or issues while using our software, please feel free to report them on the project's [issue tracker](https://github.com/BuczynskiRafal/catchments_simulation/issues). When reporting a bug, please provide as much information as possible to help us reproduce and resolve the issue, including:

* A clear and concise description of the issue
* Steps to reproduce the problem
* Expected behavior and actual behavior
* Any error messages or logs that may be relevant

Your feedback is invaluable and will help us improve the software for all users.

# Contributing

We welcome and appreciate contributions from the community! If you're interested in contributing to this project, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your changes.
3. Make your changes, including updates to documentation if needed.
4. Write tests to ensure your changes are working as expected.
5. Ensure all tests pass and there are no linting or code style issues.
6. Commit your changes and create a pull request, providing a detailed description of your changes.

We will review your pull request as soon as possible and provide feedback. Once your contribution is approved, it will be merged into the main branch.

For more information about contributing to the project, please see our [contributing guide](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/CONTRIBUTING.md).

# License

License
This project is licensed under the [MIT License](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/LICENSE). By using, distributing, or contributing to this project, you agree to the terms and conditions of the license. Please refer to the [LICENSE.md](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/LICENSE) file for the full text of the license.