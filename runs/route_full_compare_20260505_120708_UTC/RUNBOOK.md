# Route Full Run Template

Use this only in a privileged environment.

Prepared benchmark config:
`/home/ubuntu/vista/runs/route_full_compare_20260505_120708_UTC/config/route_full_150_queries.json`

Example TOML for one method:

```toml
prompt_type = "zeroshot_base"
num_queries = 10
benchmark_path = "/home/ubuntu/vista/runs/route_full_compare_20260505_120708_UTC/config/route_full_150_queries.json"
regenerate_benchmark = false
output_dir = "/home/ubuntu/vista/runs/route_full_compare_20260505_120708_UTC/outputs"
max_iterations = 10
num_switches = 2
num_hosts_per_subnet = 1

[[agent_client_configs]]
name = "qwen-zeroshot-base"
base_url = "http://127.0.0.1:8000"
prompt_type = "zeroshot_base"
```

Run:

```bash
PYTHONPATH=/home/ubuntu/NetArena/src:/usr/lib/python3/dist-packages python /home/ubuntu/NetArena/app-route/run.py -c route_config.toml
```
