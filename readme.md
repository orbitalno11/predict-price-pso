# Application of Particle Swarm Optimization Algorithm with Feed-Forward Neural Network for stock forecasting
The aim of this project is to compare ability of stock forecasting between simple Feed-Forward Neural Network and Feed-Forward Neural Network which using Particle Swarm Optimization to finding best value for Feed-Forward Neural Network weight.

## Technical indicator
 1. EMA 5 days
 2. EMA 10 days
 3. MACD
 4. RSI 14 days

## Algorithm and technic
 1. Feed-Forward Neural Network
 2. Particle Swarm Optimization
 3. Buy and Hold

## Related work

 - pyswarms Library ([see detail here)](https://pyswarms.readthedocs.io/en/latest/)

## How to run this project
- create conda environment.
 ```
conda create -n myenv python=3.7
```
- activate your environment
```
conda activate myenv
```
- using pip to install require library
```
pip install -r requirements.txt
```
- running simulator (default simulate day is 30)

`If you run the simulator in terminal you have to close the window that show the result graph for simulator next stock.`
```
python run_simulator.py``
```