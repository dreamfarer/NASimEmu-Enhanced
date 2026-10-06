import enum
from typing import Any, Sequence, Tuple

import numpy as np
from itertools import permutations

INTERNET = 0

# (subnet, host) address of a host
Address = Tuple[int, int]


class OneHotBool(enum.IntEnum):
    NONE = 0
    TRUE = 1
    FALSE = 2

    @staticmethod
    def from_bool(b: bool) -> "OneHotBool":
        if b:
            return OneHotBool.TRUE
        return OneHotBool.FALSE

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class ServiceState(enum.IntEnum):
    # values for possible service knowledge states
    UNKNOWN = 0     # service may or may not be running on host
    PRESENT = 1     # service is running on the host
    ABSENT = 2      # service not running on the host

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class AccessLevel(enum.IntEnum):
    NONE = 0
    USER = 1
    ROOT = 2

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


def get_minimal_steps_to_goal(
    topology: Any, sensitive_addresses: Sequence[Address]
) -> int:
    """Get the minimum total number of steps required to reach all sensitive
    hosts in the network starting from outside the network (i.e. can only
    reach exposed subnets).

    Returns
    -------
    int
        minimum number of steps to reach all sensitive hosts
    """
    num_subnets = len(topology)
    max_value = np.iinfo(np.int16).max
    distance = np.full((num_subnets, num_subnets),
                       max_value,
                       dtype=np.int16)

    # set distances for each edge to 1
    for s1 in range(num_subnets):
        for s2 in range(num_subnets):
            if s1 == s2:
                distance[s1][s2] = 0
            elif topology[s1][s2] == 1:
                distance[s1][s2] = 1
    # find all pair minimum shortest path distance
    for k in range(num_subnets):
        for i in range(num_subnets):
            for j in range(num_subnets):
                if distance[i][k] == max_value \
                   or distance[k][j] == max_value:
                    dis = max_value
                else:
                    dis = distance[i][k] + distance[k][j]
                if distance[i][j] > dis:
                    distance[i][j] = distance[i][k] + distance[k][j]

    # get list of all subnets we need to visit
    subnets_to_visit = [INTERNET]
    for subnet, host in sensitive_addresses:
        if subnet not in subnets_to_visit:
            subnets_to_visit.append(subnet)

    # find minimum shortest path that visits internet subnet and all
    # sensitive subnets by checking all possible permutations
    shortest = max_value
    for pm in permutations(subnets_to_visit):
        pm_sum = 0
        for i in range(len(pm) - 1):
            pm_sum += distance[pm[i]][pm[i+1]]
        shortest = min(shortest, pm_sum)

    return shortest
