# multi-armed-bandit

## Usage

To run all python files, please move to the `src` folder.

```
$ cd src
```

### Run Bandit Experiment
#### run Thompson Sampling using Gaussian Prior

```
$ python run_ts_gaussian.py
```

#### run Thompson Sampling using Gaussian Scaled Inverse Chi-Squared Prior

```
$ python run_ts_gaussian_sicq.py
```

#### run UCB1

```
$ python run_ucb1.py
```

#### run Hyperopt

```
$ python run_hyperopt.py
```

The example output is

```
$ python run_ts_gaussian.py
----------Run Exp----------
Run Exp0
iteration: 0, selected arm:　1, regret: 1.0
iteration: 5000, selected arm:　0, regret: 120.0
iteration: 10000, selected arm:　0, regret: 120.0
iteration: 15000, selected arm:　0, regret: 120.0
iteration: 20000, selected arm:　0, regret: 120.0
Finish Exp0
```

There are several options to change setting of experiment.

* exp_num: Number of times to run experiment. Defaults to 1.
* not_run_exp: Whether to run experiment. Defaults to False.
* save_log: Whether to save output to csv file. Defaults to False.
* show_log: Whether to graph output using matplotlib.pyplot. Defaults to False.
* summarize_log: Whether to calculate mean values of output and save it to csv file.

Options can be used like follows

```
$ python run_ts_gaussian.py --exp_num=100 --save_log
```

### Plot Output File
#### plot each output of each experiment in the specified folder

```
$ python plot_each_data.py
```

There are several options to change setting of plotting.

* y_min: Min value of y axis. Defaults to 0.0.
* y_max: Max value of y axis. Defaults to 100.0.
* folder_name: Name of folder where outputs of experiment are saved (Name of folder: `<Arm Name>/<Algorithm Name>`). Defaults to "N(1.0,9.0)N(0.0,0.09)/ThompsonSamplingGaussianPrior".
* file_name: Name of csv file to plot. Defaults to "regret".

#### plot summarized data of output of all experiments in the specified folder.

```
$ python plot_summarized_data.py
```

There are several options to change setting of plotting.

* y_min: Min value of y axis. Defaults to 0.0.
* y_max: Max value of y axis. Defaults to 100.0.
* folder_name: Name of folder where outputs of experiment are saved (Name of folder: `<Arm Name>`). Defaults to "N(1.0,9.0)N(0.0,0.09)".
* file_name: Name of csv file to plot. Defaults to "regret".
