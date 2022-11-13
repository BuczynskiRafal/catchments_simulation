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