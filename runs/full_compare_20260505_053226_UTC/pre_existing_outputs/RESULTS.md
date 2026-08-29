# VISTA-Net++ vs Baselines (MALT, Qwen2.5-Coder-7B-Instruct)

## Overall (n=1000 held-out test)

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 1000 | 0.2000 | 0.1670 | 0.2776 |
| baseline_zeroshot_cot | 1000 | 0.1390 | 0.2340 | 0.3887 |
| baseline_fewshot_base | 1000 | 0.5000 | 0.3580 | 0.2751 |
| baseline_fewshot_semantic | 1000 | 0.9960 | 0.5850 | 0.2667 |
| baseline_sft_only | 1000 | 1.0000 | 0.0000 | 0.2342 |
| vista_base | 1000 | 1.0000 | 0.7090 | 2.5617 |
| vista_sft | 1000 | 1.0000 | 0.7090 | 2.6078 |

### capacity planning, level-1, add

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 1.0000 | 1.0000 | 0.1970 |
| baseline_zeroshot_cot | 100 | 0.3900 | 0.7700 | 0.3827 |
| baseline_fewshot_base | 100 | 1.0000 | 1.0000 | 0.2742 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.2500 | 0.2377 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2188 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.2632 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.3513 |

### capacity planning, level-1, list

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.1860 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.2613 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.2418 |
| baseline_fewshot_semantic | 100 | 1.0000 | 1.0000 | 0.2070 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.1909 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.1890 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.2594 |

### capacity planning, level-1, rank

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.2000 | 0.3922 |
| baseline_zeroshot_cot | 100 | 0.0000 | 1.0000 | 0.4538 |
| baseline_fewshot_base | 100 | 1.0000 | 1.0000 | 0.2355 |
| baseline_fewshot_semantic | 100 | 1.0000 | 1.0000 | 0.2048 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.1826 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.2344 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.2842 |

### capacity planning, level-1, remove

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 1.0000 | 0.4700 | 0.1812 |
| baseline_zeroshot_cot | 100 | 1.0000 | 0.4700 | 0.2908 |
| baseline_fewshot_base | 100 | 1.0000 | 0.4700 | 0.2456 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.0200 | 0.1976 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.1849 |
| vista_base | 100 | 1.0000 | 0.4700 | 3.6195 |
| vista_sft | 100 | 1.0000 | 0.4700 | 3.6697 |

### capacity planning, level-2, remove-count

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.2580 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.3959 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.2794 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.5500 | 0.3084 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2664 |
| vista_base | 100 | 1.0000 | 0.5500 | 2.5803 |
| vista_sft | 100 | 1.0000 | 0.5500 | 2.5798 |

### capacity planning, level-2, remove-list

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.2580 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.1000 | 0.4280 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.2740 |
| baseline_fewshot_semantic | 100 | 0.9600 | 0.4300 | 0.2808 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2459 |
| vista_base | 100 | 1.0000 | 0.4700 | 2.4625 |
| vista_sft | 100 | 1.0000 | 0.4700 | 2.6034 |

### capacity planning, level-2, remove-rank

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.3212 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.4913 |
| baseline_fewshot_base | 100 | 1.0000 | 0.4600 | 0.2858 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.4600 | 0.2816 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2426 |
| vista_base | 100 | 1.0000 | 0.4600 | 2.5014 |
| vista_sft | 100 | 1.0000 | 0.4600 | 2.4646 |

### capacity planning, level-3, add-count

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.2964 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.3438 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.3065 |
| baseline_fewshot_semantic | 100 | 1.0000 | 1.0000 | 0.3459 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2940 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.6233 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.7340 |

### capacity planning, level-3, add-list

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.2624 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.3693 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.2944 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.4900 | 0.3053 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2624 |
| vista_base | 100 | 1.0000 | 0.4900 | 2.5217 |
| vista_sft | 100 | 1.0000 | 0.4900 | 2.5638 |

### capacity planning, level-3, add-rank

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.4239 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.4700 |
| baseline_fewshot_base | 100 | 1.0000 | 0.6500 | 0.3137 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.6500 | 0.2979 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2535 |
| vista_base | 100 | 1.0000 | 0.6500 | 2.6218 |
| vista_sft | 100 | 1.0000 | 0.6500 | 2.5684 |
