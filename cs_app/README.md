
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