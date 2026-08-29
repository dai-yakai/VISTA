# Route Full Comparison Status

## Status

Route full comparison was prepared but could not be executed in the current container.

The blocker is environment-level: NetArena Route requires Mininet to create Linux network namespaces, veth links, and OVS datapaths. This container is root, but its capability bounding set does not include `CAP_NET_ADMIN` or `CAP_SYS_ADMIN`.

## Evidence

- `unshare -n true` fails with `Operation not permitted`.
- `ip link add route_test_dummy type dummy` fails with `RTNETLINK answers: Operation not permitted`.
- `mn --test pingall --topo single,2 --switch ovsk --controller none` hangs during host creation and times out.
- There is no Docker socket or Docker CLI available for launching a privileged NetArena container from this session.

Full diagnostic log: `/home/ubuntu/vista/runs/route_full_compare_20260505_120708_UTC/logs/environment_check.log`

## Prepared Benchmark Config

- Config path: `/home/ubuntu/vista/runs/route_full_compare_20260505_120708_UTC/config/route_full_150_queries.json`
- Total queries: `150`
- `num_errors_per_type`: `10`
- `num_switches`: `2`
- `num_hosts_per_subnet`: `1`

### Query Distribution

| Error type | Queries |
|---|---:|
| disable_interface | 10 |
| disable_interface+drop_traffic_to_from_subnet | 10 |
| disable_interface+remove_ip | 10 |
| disable_interface+wrong_routing_table | 10 |
| disable_routing | 10 |
| disable_routing+disable_interface | 10 |
| disable_routing+drop_traffic_to_from_subnet | 10 |
| disable_routing+remove_ip | 10 |
| disable_routing+wrong_routing_table | 10 |
| drop_traffic_to_from_subnet | 10 |
| drop_traffic_to_from_subnet+wrong_routing_table | 10 |
| remove_ip | 10 |
| remove_ip+drop_traffic_to_from_subnet | 10 |
| remove_ip+wrong_routing_table | 10 |
| wrong_routing_table | 10 |

## Planned Method Lines

For Route, NetArena exposes prompt strategies through `PromptType`:

| Family | Method |
|---|---|
| NetArena baseline | `zeroshot_base` |
| NetArena baseline | `zeroshot_cot` |
| NetArena baseline | `fewshot_base` |
| NetArena baseline | `fewshot_cot` |
| VISTA | Route-specific verifier/repair agent endpoint required |

Note: the existing VISTA code in `/home/ubuntu/vista/src` is MALT-specific; a Route VISTA adapter must expose an A2A endpoint that returns one JSON command per turn and applies VISTA-style command validation/repair. The official Route evaluator itself is ready once a privileged Mininet environment is available.

## How To Run In A Privileged Environment

Start from NetArena root in an environment with `CAP_NET_ADMIN` and `CAP_SYS_ADMIN` or Docker `--privileged`:

```bash
cd /home/ubuntu/NetArena
service openvswitch-switch start || true
ovs-vswitchd --pidfile --detach --log-file=/var/log/openvswitch/ovs-vswitchd.log || true
mn -c
mn --test pingall --topo single,2 --switch ovsk --controller none
```

For each agent endpoint, create a config TOML pointing `benchmark_path` to the prepared JSON and run:

```bash
cd /home/ubuntu/NetArena/app-route
PYTHONPATH=/home/ubuntu/NetArena/src:/usr/lib/python3/dist-packages python run.py -c /path/to/route_config.toml
```

## Current Outcome

No Route metric is reported from this container because the official Route benchmark could not start a Mininet network. Reporting simulated or non-Mininet numbers here would not be a valid NetArena Route result.
