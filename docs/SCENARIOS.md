To set a custom environment or to use emulation, you have custom scenarios that describe the environment.

# Version

NASimEmu loads scenarios in the V2 format; the filename has to end with `.v2.yaml`. It is a custom format that is not compatible with NASim and allows some variables to be random.

The original NASim format (V1) is no longer supported since version 2.0.0.

# Common fields
The V2 format builds on the original NASim format, see the [original documentation](https://networkattacksimulator.readthedocs.io/en/latest/tutorials/scenarios.html).

You can set :
- subnet
  - A list containing the number of host in each subnet
  - For example, [2, 1, 5] means there is 2 hosts in subnet (1, X), 1 in (2, X) and 5 in (3, X)
- topology
  - Define which subnet are connected together
  - Also, the topology should always be symmetrical (topology[x][y] = topology[y][x])
  - For example, topology[2][3] = 1 means that subnet (2, X) and (3, X) are connected.
- sensitive hosts
  - Set the value of hosts
- os
  - List of the os name
- services
  - List of the service name
- processes
  - List of the processes name
- exploits
  - List of the exploits
- privilege escalations
  - List of the privilege escalations
- service scan cost
  - The cost of service scan (int)
- os scan cost
  - The cost of os scan (int)
- subnet scan cost
  - The cost of subnet scan (int)
- process scan cost
  - The cost of process scan (int)
- host configuration
  - List of the host with their configuration
- firewall
  - Set services that can communicate between 2 subnet with a list of service or [_all] to allow all services

# V2 Scenario

On top of the common fields, a V2 scenario adds some randomness to the generation.

With this scenario, subnet can have variable size by setting the minimum and maximum number of hosts in the 'subnets' field (for example, 1-5 for 1 to 5 hosts).

With `address_space_bounds`, you set the max number of subnet and the max number of hosts per subnet. This is crucial if you run the agent in multiple scenarios at once. Use the maximum possible values for all scenarios to enforce the same vector size per host.

Sensitive hosts (hosts with value) are calculated differently. You set the percent of sensitive hosts per subnet.

You can set `firewall: _subnets` to set the firewall to allow all services according to the network topology.

You can set `host_configurations: _random` to generate random host configurations.

The field `sensitive_services` defines services which will **be installed on every sentitive hosts**. However, there is also 10% chance that this service will be installed in a non-sensitive host.

## Example


```yaml
subnets: [1, 3-8] # ranges of hosts in subnets
# address_space_bounds: (2, 8)  # max number of subnets & hosts in a subnet
address_space_bounds: (5, 10) # fix for the experiment - all scenarios have to have the same bounds
topology: [[ 1, 1, 0],
           [ 1, 1, 1],
           [ 0, 1, 1]]
sensitive_hosts:  # probabilities of sensitive hosts in specific subnets
  1: 0.  # DMZ
  2: 0.  # user subnet
os:
  - linux
  - windows
services:
  - 21_linux_proftpd
  - 80_linux_drupal
  - 80_linux_phpwiki
  - 9200_windows_elasticsearch
  - 80_windows_wp_ninja
  - 3306_any_mysql
sensitive_services:
  - 3306_any_mysql
processes:
  - ~
exploits:
  e_proftpd:
    service: 21_linux_proftpd
    os: linux
    prob: 1.0
    cost: 1
    access: user
  e_drupal:
    service: 80_linux_drupal
    os: linux
    prob: 1.0
    cost: 1
    access: user
  e_phpwiki:
    service: 80_linux_phpwiki
    os: linux
    prob: 1.0
    cost: 1
    access: user
  e_elasticsearch:
    service: 9200_windows_elasticsearch
    os: windows
    prob: 1.0
    cost: 1
    access: root
  e_wp_ninja:
    service: 80_windows_wp_ninja
    os: windows
    prob: 1.0
    cost: 1
    access: user
privilege_escalation:
  pe_kernel:
    process: ~
    os: linux
    prob: 1.0
    cost: 1
    access: root
service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
host_configurations: _random
firewall: _subnets
```