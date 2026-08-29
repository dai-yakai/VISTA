# VISTA-Net++ vs Baselines (MALT, Qwen2.5-Coder-7B-Instruct)

## Overall (n=1000 held-out test)

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 1000 | 0.5800 | 0.4640 | 0.2802 |
| baseline_zeroshot_cot | 1000 | 0.4180 | 0.4180 | 0.3918 |
| baseline_fewshot_base | 1000 | 0.8000 | 0.5540 | 0.2815 |
| baseline_fewshot_semantic | 1000 | 1.0000 | 0.5890 | 0.2710 |
| baseline_sft_only | 1000 | 1.0000 | 0.0000 | 0.2358 |
| vista_base | 1000 | 1.0000 | 0.7090 | 2.4601 |
| vista_sft | 1000 | 1.0000 | 0.7090 | 2.5041 |

### capacity planning, level-1, add

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 1.0000 | 1.0000 | 0.1986 |
| baseline_zeroshot_cot | 100 | 0.3900 | 0.7700 | 0.3875 |
| baseline_fewshot_base | 100 | 1.0000 | 1.0000 | 0.2790 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.2500 | 0.2418 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2204 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.1809 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.2586 |

### capacity planning, level-1, list

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 1.0000 | 1.0000 | 0.1873 |
| baseline_zeroshot_cot | 100 | 1.0000 | 1.0000 | 0.2632 |
| baseline_fewshot_base | 100 | 1.0000 | 1.0000 | 0.2559 |
| baseline_fewshot_semantic | 100 | 1.0000 | 1.0000 | 0.2107 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.1914 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.1077 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.1493 |

### capacity planning, level-1, rank

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.8200 | 0.3921 |
| baseline_zeroshot_cot | 100 | 0.0000 | 1.0000 | 0.4584 |
| baseline_fewshot_base | 100 | 1.0000 | 1.0000 | 0.2424 |
| baseline_fewshot_semantic | 100 | 1.0000 | 1.0000 | 0.2091 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.1831 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.1553 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.2099 |

### capacity planning, level-1, remove

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 1.0000 | 0.4700 | 0.1837 |
| baseline_zeroshot_cot | 100 | 1.0000 | 0.4700 | 0.2935 |
| baseline_fewshot_base | 100 | 1.0000 | 0.4700 | 0.2517 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.0200 | 0.2009 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.1865 |
| vista_base | 100 | 1.0000 | 0.4700 | 3.4391 |
| vista_sft | 100 | 1.0000 | 0.4700 | 3.5792 |

### capacity planning, level-2, remove-count

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.2621 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.3980 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.2843 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.5500 | 0.3124 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2672 |
| vista_base | 100 | 1.0000 | 0.5500 | 2.4473 |
| vista_sft | 100 | 1.0000 | 0.5500 | 2.4896 |

### capacity planning, level-2, remove-list

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.9600 | 0.4700 | 0.2649 |
| baseline_zeroshot_cot | 100 | 0.5800 | 0.3700 | 0.4327 |
| baseline_fewshot_base | 100 | 1.0000 | 0.4700 | 0.2787 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.4700 | 0.2853 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2483 |
| vista_base | 100 | 1.0000 | 0.4700 | 2.3984 |
| vista_sft | 100 | 1.0000 | 0.4700 | 2.3686 |

### capacity planning, level-2, remove-rank

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.8400 | 0.3900 | 0.3222 |
| baseline_zeroshot_cot | 100 | 0.0700 | 0.0700 | 0.4929 |
| baseline_fewshot_base | 100 | 1.0000 | 0.4600 | 0.2911 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.4600 | 0.2862 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2448 |
| vista_base | 100 | 1.0000 | 0.4600 | 2.4092 |
| vista_sft | 100 | 1.0000 | 0.4600 | 2.3878 |

### capacity planning, level-3, add-count

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.2971 |
| baseline_zeroshot_cot | 100 | 0.0000 | 0.0000 | 0.3461 |
| baseline_fewshot_base | 100 | 0.0000 | 0.0000 | 0.3141 |
| baseline_fewshot_semantic | 100 | 1.0000 | 1.0000 | 0.3519 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2960 |
| vista_base | 100 | 1.0000 | 1.0000 | 2.5432 |
| vista_sft | 100 | 1.0000 | 1.0000 | 2.6855 |

### capacity planning, level-3, add-list

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 1.0000 | 0.4900 | 0.2660 |
| baseline_zeroshot_cot | 100 | 1.0000 | 0.4900 | 0.3732 |
| baseline_fewshot_base | 100 | 1.0000 | 0.4900 | 0.2999 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.4900 | 0.3096 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2644 |
| vista_base | 100 | 1.0000 | 0.4900 | 2.4424 |
| vista_sft | 100 | 1.0000 | 0.4900 | 2.4812 |

### capacity planning, level-3, add-rank

| Method | n | Correctness | Safety | Avg Latency (s) |
|---|---:|---:|---:|---:|
| baseline_zeroshot_base | 100 | 0.0000 | 0.0000 | 0.4281 |
| baseline_zeroshot_cot | 100 | 0.1400 | 0.0100 | 0.4730 |
| baseline_fewshot_base | 100 | 1.0000 | 0.6500 | 0.3176 |
| baseline_fewshot_semantic | 100 | 1.0000 | 0.6500 | 0.3024 |
| baseline_sft_only | 100 | 1.0000 | 0.0000 | 0.2556 |
| vista_base | 100 | 1.0000 | 0.6500 | 2.4775 |
| vista_sft | 100 | 1.0000 | 0.6500 | 2.4316 |
