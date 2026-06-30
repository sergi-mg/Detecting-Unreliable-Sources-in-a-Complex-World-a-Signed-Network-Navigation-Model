# Detecting Unreliable Sources in a Complex World: a Signed-Network Navigation Model

This repository contains the code developed as part of my Master's thesis, *Detecting Unreliable Sources in a Complex World: a Signed-Network Navigation Model*, completed as part of the Master's Degree in Complex Systems and Biophysics at the University of Barcelona.

## Repository Structure

This repository contains a single folder with the three Python scripts developed as part of the thesis. 
Additionally, other folders may be generated when running the code, as described below. Note that, since 
the code was progressively developed and reused throughout the thesis, some file, folder, or function names 
may not fully describe their current content. This mainly occurs in the statistical_analysis.py script, which was used throughout the development of the project to generate figures for analysing the results, and does not affect the functionality of the code.

### Code Folder

This folder contains the three different scripts developed:

#### optimized_code.py

This script contains the code used to simulate the system using different 
exploration strategies, node definition heuristic rules, and biases.
The corresponding strings must be used to identify each case. 
The following function is used to run the simulations and save the data.
The rest of the functions are internal, and their documentation can be found in the script.

```python
main_program(N, k, r, update_rule, N_i, rule, strategy, weight, GTN_network,
             c_BA="Random", M=0, p_r=-1)
```
where `N` is the number of nodes, `k` is the expected number of connections per node, and `r` is the noise value ($\in[0,0.5]$). The weight function must always be provided as an input, but it will only be used when the rule string corresponds to an order bias.

The heuristic rule is selected using the `rule` and `update_rule` parameters.

##### Majority rule

| Bias | String (rule) | Function (update_rule) | Additional parameters |
|---|---|---|---|
| No bias | `mr` | `update_majority` | |
| Anchoring bias | `mr_anchor` | `update_majority_anchor` | |
| Ambiguity bias | `mr_ambiguity` | `update_majority_ambiguity` | Threshold `M` $\in(0.5,1]$|
| Primacy effect (linear) | `mr_primacy_linear` | `update_majority_weighted` | Weight function `prim_lin(t)` |
| Recency effect (linear) | `mr_recency_linear` | `update_majority_weighted` | Weight function `rec_lin(t)` |

##### Random neighbour

| Bias | String (rule) | Function (update_rule) | Additional parameters |
|---|---|---|---|
| No bias | `rn` | `update_rn` | |
| Anchoring bias | `rn_anchor` | `update_rn_anchor` | |
| Primacy effect (linear) | `rn_primacy_linear` | `update_rn_weighted` | Weight function `prim_lin(t)` |
| Recency effect (linear) | `rn_recency_linear` | `update_rn_weighted` | Weight function `rec_lin(t)` |

The exploration strategy is selected using the `strategy` parameter.

| Strategy | String |
|---|---|
| Random Selection | `rs` |
| Ordered by Distance | `obd` |

The network topology is selected using the `GTN_network` parameter.

| Topology | Function | Additional parameters |
|---|---|---|
| Erdös-Rényi | `ground_truth_network_ER` | |
| Watts-Strogatz | `ground_truth_network_WS` | Rewiring probability `p_r` $\in[0,1]$|
| Barabási-Albert | `ground_truth_network_BA` | `c_BA`: `Min`, `Max` or `Random` (default)|

Additionally, when working with Barabási-Albert or Watts-Strogatz graphs (no biases applied) `_BA` or `_WS` must be added to the `rule` string.

Note that the parameters `M`, `p_r`, `c_BA`, and `weight` only affect the configurations where they are required. They are
included in the function call for consistency across simulations.

For each execution of the function `main_program` a .npz file is generated and saved in the `data/time_evo_simulations_biases/` folder. The name of each file is the following: 
```python
name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".npz"
```
If additional parameters are needed, they will be added at the end of the name (see the code for the detailed implementation).

#### statistical_analysis.py

This script contains the code used to create all the plots used to analyse the obtained results. The generated figures are saved in a new folder called `images/biases/`. For each plot type, an additional folder inside the biases one will be created (see the code for the detailed implementation). Each of those folders contains the folders `pdf/` and `png/` since the plots are saved in two different file formats. As before, the name of each plot is created using information regarding the corresponding parameters (see the code for the detailed implementation).

#### final_plots.py

This script contains the code used to create the figures displayed in the report. All figures will be saved in `.pdf` format in the folder: `images/final_plots/pdf/`. 
