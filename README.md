# Catchment simulation
Package include method for simulate subcatchment with different features values from Storm Water Management Model

## Examples of How To Use 

Creating object for analyse

#### Create object to run simulation

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
```

#### Simulate subcatchment area in selected range.

```python
from catchment_simulation.catchment_features_simulation import FeaturesSimulation

subcatchment_id = "S1"
raw_file = "catchment_simulation/example.inp"
model = FeaturesSimulation(subcatchment_id=subcatchment_id, raw_file=raw_file)
df = model.simulate_area(start=1, stop: = 10, step = 1)
```

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

# Catchment simulation app

The application was built in django share some of the functionality from the 'catchment simulation' package. 

It is designed to analyze and predict water flow behavior in catchments.  The application contain two main components: the Catchment Simulation package and Catchment Calculation.
On the main page you can find information of the 'catchments simulation' package and examples of use. 

<div align="center">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/home.png">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/example.png">
</div>

The 'Simulations' tab allows the user to upload a file and select components for simulation. Once the simulation is executed, the window will display an interactive graph of the obtained data and a button to download the results in an excel spreadsheet. 

<div align="center">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/simulation_start.png">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/simulation_after.png">
</div>

The 'Calculations' tab contains a neural network model trained to predict catchment area runoff. The user, after uploading the file, receives the results of calculations performed SWMM and ANN model prediction. 

<div align="center">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/calculations.png">
    <img src="https://github.com/BuczynskiRafal/catchments_simulation/blob/main/img/ann_model.png">
</div>