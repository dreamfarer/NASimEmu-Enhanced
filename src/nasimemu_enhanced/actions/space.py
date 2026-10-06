"""The action space of the NASim environment."""

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from .actions import (
    Action,
    Exploit,
    NoOp,
    OSScan,
    PrivilegeEscalation,
    ProcessScan,
    ServiceScan,
    SubnetScan,
    load_action_list,
)

if TYPE_CHECKING:
    from nasimemu_enhanced.scenarios.scenario import Scenario


class ParameterisedActionSpace(spaces.MultiDiscrete):
    """A parameterised action space for NASim environment.

    Inherits and implements the gym.spaces.MultiDiscrete action space, where
    each dimension corresponds to a different action parameter.

    The action parameters (in order) are:

    0. Action Type = [0, 5]

       Where:

         0=Exploit,

         1=PrivilegeEscalation,

         2=ServiceScan,

         3=OSScan,

         4=SubnetScan,

         5=ProcessScan,

    1. Subnet = [0, #subnets-1]

       -1 since we don't include the internet subnet

    2. Host = [0, max subnets size-1]
    3. OS = [0, #OS+1]

       Where 0=None.

    4. Service = [0, #services]
    5. Process = [0, #processes+1]

       Where 0=None.

    Note that OS, Service and Process are only importand for exploits and
    privilege escalation actions.

    ...

    Attributes
    ----------
    nvec : Numpy.Array
        vector of the of the size of each parameter
    actions : list of Actions
        the list of all the Actions in the action space
    """

    action_types: list[type[Action]] = [
        Exploit,
        PrivilegeEscalation,
        ServiceScan,
        OSScan,
        SubnetScan,
        ProcessScan
    ]

    def __init__(self, scenario: "Scenario") -> None:
        """Initialize ParameterisedActionSpace.

        Parameters
        ----------
        scenario : Scenario
            scenario description
        """
        self.scenario = scenario
        self.actions = load_action_list(scenario)

        nvec = [
            len(self.action_types),
            len(self.scenario.subnets)-1,
            max(self.scenario.subnets),
            self.scenario.num_os+1,
            self.scenario.num_services,
            self.scenario.num_processes
        ]

        super().__init__(nvec)

    def get_action(self, action_vec: Sequence[int] | NDArray[Any]) -> Action:
        """Get Action object corresponding to action vector.

        Parameters
        ----------
        action_vector : list of ints or tuple of ints or Numpy.Array
            the action vector

        Returns
        -------
        Action
            Corresponding Action object

        Notes
        -----
        1. if host# specified in action vector is greater than
           the number of hosts in the specified subnet, then host#
           will be changed to host# % subnet size.
        2. if action is an exploit and parameters do not match
           any exploit definition in the scenario description then
           a NoOp action is returned with 0 cost.
        """
        assert isinstance(action_vec, (list, tuple, np.ndarray)), \
            ("When using parameterised action space, action must be an Action"
             f" object, a list or a numpy array: {action_vec} is invalid")
        a_class = self.action_types[action_vec[0]]
        # need to add one to subnet to account for Internet subnet
        subnet = action_vec[1]+1
        host = action_vec[2] % self.scenario.subnets[subnet]

        target = (subnet, host)

        if a_class not in (Exploit, PrivilegeEscalation):
            # can ignore other action parameters
            kwargs = self._get_scan_action_def(a_class)
            return a_class(target=target, **kwargs)

        os = None if action_vec[3] == 0 else self.scenario.os[action_vec[3]-1]

        if a_class == Exploit:
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            service = self.scenario.services[action_vec[4]]
            a_def = self._get_exploit_def(service, os)
        else:
            # privilege escalation
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            proc = self.scenario.processes[action_vec[5]]
            a_def = self._get_privesc_def(proc, os)

        if a_def is None:
            return NoOp()
        return a_class(target=target, **a_def)

    def _get_scan_action_def(self, a_class: type[Action]) -> dict[str, Any]:
        """Get the constants for scan actions definitions."""
        if a_class == ServiceScan:
            cost = self.scenario.service_scan_cost
        elif a_class == OSScan:
            cost = self.scenario.os_scan_cost
        elif a_class == SubnetScan:
            cost = self.scenario.subnet_scan_cost
        elif a_class == ProcessScan:
            cost = self.scenario.process_scan_cost
        else:
            raise TypeError(f"Not implemented for Action class {a_class}")
        return {"cost": cost}

    def _get_exploit_def(
        self, service: str, os: str | None
    ) -> dict[str, Any] | None:
        """Check if exploit parameters are valid."""
        e_map = self.scenario.exploit_map
        if service not in e_map:
            return None
        if os not in e_map[service]:
            return None
        return e_map[service][os]

    def _get_privesc_def(
        self, proc: str, os: str | None
    ) -> dict[str, Any] | None:
        """Check if privilege escalation parameters are valid."""
        pe_map = self.scenario.privesc_map
        if proc not in pe_map:
            return None
        if os not in pe_map[proc]:
            return None
        return pe_map[proc][os]
