# Detecting Unreliable Sources in a Complex World: a Signed-Network Navigation Model

This repository contains the code developed as part of my Master's thesis, *Detecting Unreliable Sources in a Complex World: a Signed-Network Navigation Model*, completed as part of the Master's Degree in Complex Systems and Biophysics at the University of Barcelona.

## Repository Structure

This repository contains a single folder with the three Python scripts developed as part of the thesis. 
Additionally, other folders may be generated when running the code, as described below. Note that, since 
the code was progressively developed and reused throughout the thesis, some file, folder, or function names 
may not fully describe their current content. This does not affect the functionality of the code.

### Code Folder

This folder contains the three different scripts developed:

#### optimized_code.py

This program provides the code to simulate the system with different 
exploration strategies, node definition heuristic rules and biases.
It is important to use the corresponding strings to identify each case. 
The following function is the one to be executed to simulate and save the data, 
the rest of the functions are internal and their documentation can be found in the script.

```python
main_program(N, k, r, update_rule, N_i, rule, strategy, weight, GTN_network,
             c_BA="Random", M=0, p_r=-1)
```

where:

| Parameter | Description | Type |
|---|---|---|
| `N` | Number of nodes in the network | Int |
| `k` | Expected node degree | Int |
| `r` | Noise value [0,0.5] | Float |
| `update_rule` | Function used to update node opinions | Function |
| `N_i` | Number of initially known nodes | Int |
| `rule` | String identifying the heuristic rule configuration | Str |
| `strategy` | Exploration strategy (`rs` or `obd`) | Str |
| `weight` | Temporal weighting function for primacy/recency effects | Function |
| `GTN_network` | Function used to generate the network topology |  Function |
| `c_BA` | Initial node selection criterion for Barabási-Albert networks | Str |
| `M` | Threshold parameter for the ambiguity bias (0.5,1] | Float |
| `p_r` | Rewiring probability for Watts-Strogatz networks [0,1]| Float |

The wight function must always to be introduced as an input but it will only be used where the rule string
corresponds to order biases.

The heuristic rule is selected using the `rule` parameter.

#### Majority rule

| Bias | String | Function | Additional parameters |
|---|---|---|---|
| No bias | `mr` | `update_majority` | |
| Anchoring bias | `mr_anchor` | `update_majority_anchor` | |
| Ambiguity bias | `mr_ambiguity` | `update_majority_ambiguity` | Threshold `M` |
| Primacy effect (linear) | `mr_primacy_linear` | `update_majority_weighted` | Weight function `prim_lin(t)` |
| Recency effect (linear) | `mr_recency_linear` | `update_majority_weighted` | Weight function `rec_lin(t)` |

#### Random neighbour

| Bias | String | Function | Additional parameters |
|---|---|---|---|
| No bias | `rn` | `update_rn` | |
| Anchoring bias | `rn_anchor` | `update_rn_anchor` | |
| Primacy effect (linear) | `rn_primacy_linear` | `update_rn_weighted` | Weight function `prim_lin(t)` |
| Recency effect (linear) | `rn_recency_linear` | `update_rn_weighted` | Weight function `rec_lin(t)` |

When working with Barabási-Albert and Watts-Strogatz graphs (no biases applied) `_BA` or `_WS` must be added


The exploration strategy is selected using the `strategy` parameter.

| Strategy | String |
|---|---|
| Random Selection | `rs` |
| Ordered by Distance | `obd` |

The network topology is selected using the `GTN_network` parameter.

| Topology | Function | Additional parameters |
|---|---|---|
| Erdös-Rényi | `ground_truth_network_ER` | |
| Watts-Strogatz | `ground_truth_network_WS` | Rewiring probability `p_r` |
| Barabási-Albert | `ground_truth_network_BA` | |

For Barabási-Albert networks, the initial node selection criterion can be modified using `c_BA` which can take the
additional values `Min`, `Max` and `Random`(default).

Note that the parameters `M`, `p_r`, and `weight` only affect the configurations where they are required. They are
included in the function call for consistency across simulations.
