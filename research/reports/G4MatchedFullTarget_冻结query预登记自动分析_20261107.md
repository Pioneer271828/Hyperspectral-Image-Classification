# Full54 preregistered query analysis

primary_benefit_not_supported

Three model seeds with one support set; no significance claim. Exact count-based predicates.

| Claim | Mean macro OA delta (pp) | Sample SD (pp) | All gates |
|---|---:|---:|---|
| residualFT_minus_capacityFT | 0.330635574 | 1.613333738 | False |
| residualFT_minus_noneFT | -0.182743026 | 0.741943501 | False |
| none_FT_minus_scratch | -0.268415616 | 0.788259680 | False |
| capacity_control_FT_minus_scratch | -1.161600975 | 1.435109093 | False |
| x_residual_FT_minus_scratch | -1.196307980 | 0.848458177 | False |

## residualFT_minus_capacityFT

Paired seed deltas (pp): [-1.0868777006523551, -0.007430522770089502, 2.0862149462351716]

- mean_equal_scene_macro_oa_positive: True
- all_three_model_seeds_positive: False
- every_scene_mean_aa_drop_at_most_1pp: True
- honghu_mean_oa_drop_at_most_1pp: True
- every_class_mean_recall_drop_at_most_2pp: False

| Scene | Mean OA delta (pp) | Mean AA delta (pp) | Failed recall class IDs |
|---|---:|---:|---|
| paviaU | 0.727781556 | 0.176483697 | [] |
| xuzhou | 0.005818859 | 0.260581643 | [] |
| whu-honghu | 0.258306308 | 1.255152326 | [8] |

## residualFT_minus_noneFT

Paired seed deltas (pp): [-0.8706751686283838, -0.28097847388210995, 0.6034245656715365]

- mean_equal_scene_macro_oa_positive: False
- all_three_model_seeds_positive: False
- every_scene_mean_aa_drop_at_most_1pp: True
- honghu_mean_oa_drop_at_most_1pp: True
- every_class_mean_recall_drop_at_most_2pp: False

| Scene | Mean OA delta (pp) | Mean AA delta (pp) | Failed recall class IDs |
|---|---:|---:|---|
| paviaU | 0.317378423 | -0.422243432 | [1, 6, 8] |
| xuzhou | -0.264273176 | -0.056764547 | [4] |
| whu-honghu | -0.601334324 | 0.421358008 | [4, 8] |

## none_FT_minus_scratch

Paired seed deltas (pp): [0.6103751382578262, -0.9131118218128562, -0.5025101641415703]

- mean_equal_scene_macro_oa_positive: False
- all_three_model_seeds_positive: False
- every_scene_mean_aa_drop_at_most_1pp: True
- honghu_mean_oa_drop_at_most_1pp: False
- every_class_mean_recall_drop_at_most_2pp: False

| Scene | Mean OA delta (pp) | Mean AA delta (pp) | Failed recall class IDs |
|---|---:|---:|---|
| paviaU | -0.523752570 | 0.951879397 | [2, 6] |
| xuzhou | 0.819974203 | 0.969698491 | [] |
| whu-honghu | -1.101468481 | 3.276189376 | [3, 4, 6, 17, 19] |

## capacity_control_FT_minus_scratch

Paired seed deltas (pp): [0.338749190810517, -1.3024730581286645, -2.5210790564700605]

- mean_equal_scene_macro_oa_positive: False
- all_three_model_seeds_positive: False
- every_scene_mean_aa_drop_at_most_1pp: True
- honghu_mean_oa_drop_at_most_1pp: False
- every_class_mean_recall_drop_at_most_2pp: False

| Scene | Mean OA delta (pp) | Mean AA delta (pp) | Failed recall class IDs |
|---|---:|---:|---|
| paviaU | -1.375827646 | -0.049560904 | [2, 6] |
| xuzhou | 0.565884030 | 0.719865584 | [] |
| whu-honghu | -2.674859308 | 1.817504000 | [4, 6, 18] |

## x_residual_FT_minus_scratch

Paired seed deltas (pp): [-0.800969819701702, -2.170290194157958, -0.617663924872496]

- mean_equal_scene_macro_oa_positive: False
- all_three_model_seeds_positive: False
- every_scene_mean_aa_drop_at_most_1pp: True
- honghu_mean_oa_drop_at_most_1pp: False
- every_class_mean_recall_drop_at_most_2pp: False

| Scene | Mean OA delta (pp) | Mean AA delta (pp) | Failed recall class IDs |
|---|---:|---:|---|
| paviaU | -0.978713757 | 0.205209626 | [2, 6] |
| xuzhou | 0.486359625 | 0.849174039 | [] |
| whu-honghu | -3.096569806 | 3.043539864 | [1, 3, 4, 6, 7, 17] |

All54 three-head/count/class records,18 groups and descriptive contrasts are preserved in analysis.json.
NLL is authenticated query output; it cannot be independently reconstructed from confusion counts.
Scratch/capacity descriptive contrasts and difference-in-differences do not replace the primary claim.
