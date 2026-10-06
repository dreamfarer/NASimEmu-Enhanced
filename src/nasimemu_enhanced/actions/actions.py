"""Action classes for the NASim environment.

Every action inherits from the base :class:`Action` class, which defines
some common attributes and functions. Different types of actions
are implemented as subclasses of the Action class.

Action types implemented:

- :class:`Exploit`
- :class:`PrivilegeEscalation`
- :class:`ServiceScan`
- :class:`OSScan`
- :class:`SubnetScan`
- :class:`ProcessScan`
- :class:`NoOp`
"""

import math
from typing import TYPE_CHECKING, Any

from ..utils import AccessLevel, Address

if TYPE_CHECKING:
    from nasimemu_enhanced.scenarios.scenario import Scenario


def load_action_list(scenario: "Scenario") -> list["Action"]:
    """Load list of actions for environment for given scenario.

    Parameters
    ----------
    scenario : Scenario
        the scenario

    Returns
    -------
    list
        list of all actions in environment
    """
    action_list: list[Action] = []
    for address in scenario.address_space:
        action_list.append(
            ServiceScan(address, scenario.service_scan_cost)
        )
        action_list.append(
            OSScan(address, scenario.os_scan_cost)
        )
        action_list.append(
            SubnetScan(address, scenario.subnet_scan_cost)
        )
        action_list.append(
            ProcessScan(address, scenario.process_scan_cost)
        )
        for e_name, e_def in scenario.exploits.items():
            exploit = Exploit(e_name, address, **e_def)
            action_list.append(exploit)
        for pe_name, pe_def in scenario.privescs.items():
            privesc = PrivilegeEscalation(pe_name, address, **pe_def)
            action_list.append(privesc)
    return action_list


class Action:
    """The base abstract action class in the environment.

    There are multiple types of actions (e.g. exploit, scan, etc.), but every
    action has some common attributes.

    ...

    Attributes
    ----------
    name : str
        the name of action
    target : (int, int)
        the (subnet, host) address of target of the action. The target of the
        action could be the address of a host that the action is being used
        against (e.g. for exploits or targeted scans) or could be the host that
        the action is being executed on (e.g. for subnet scans).
    cost : float
        the cost of performing the action
    prob : float
        the success probability of the action. This is the probability that
        the action works given that it's preconditions are met. E.g. a remote
        exploit targeting a host that you cannot communicate with will always
        fail. For deterministic actions this will be 1.0.
    req_access : AccessLevel,
        the required access level to perform action. For for on host actions
        (i.e. subnet scan, process scan, and privilege escalation) this will
        be the access on the target. For remote actions (i.e. service scan,
        os scan, and exploits) this will be the access on a pivot host (i.e.
        a compromised host that can reach the target).
    """

    def __init__(self,
                 name: str,
                 target: Address,
                 cost: float,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize Action.

        Parameters
        ----------
        name : str
            name of action
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_exploit(self) -> bool:
        """Check if action is an exploit.

        Returns
        -------
        bool
            True if action is exploit, otherwise False
        """
        return isinstance(self, Exploit)

    def is_privilege_escalation(self) -> bool:
        """Check if action is privilege escalation action.

        Returns
        -------
        bool
            True if action is privilege escalation action, otherwise False
        """
        return isinstance(self, PrivilegeEscalation)

    def is_scan(self) -> bool:
        """Check if action is a scan.

        Returns
        -------
        bool
            True if action is scan, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, SubnetScan, ProcessScan))

    def is_remote(self) -> bool:
        """Check if action is a remote action.

        A remote action is one where the target host is a remote host (i.e. the
        action is not performed locally on the target)

        Returns
        -------
        bool
            True if action is remote, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, Exploit))

    def is_service_scan(self) -> bool:
        """Check if action is a service scan.

        Returns
        -------
        bool
            True if action is service scan, otherwise False
        """
        return isinstance(self, ServiceScan)

    def is_os_scan(self) -> bool:
        """Check if action is an OS scan.

        Returns
        -------
        bool
            True if action is an OS scan, otherwise False
        """
        return isinstance(self, OSScan)

    def is_subnet_scan(self) -> bool:
        """Check if action is a subnet scan.

        Returns
        -------
        bool
            True if action is a subnet scan, otherwise False
        """
        return isinstance(self, SubnetScan)

    def is_process_scan(self) -> bool:
        """Check if action is a process scan.

        Returns
        -------
        bool
            True if action is a process scan, otherwise False
        """
        return isinstance(self, ProcessScan)

    def is_noop(self) -> bool:
        """Check if action is a do nothing action.

        Returns
        -------
        bool
            True if action is a noop action, otherwise False
        """
        return isinstance(self, NoOp)

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}: "
                f"name={self.name}, "
                f"target={self.target}, "
                f"cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}, "
                f"req_access={self.req_access}")

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.target != other.target:
            return False
        if not (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob)):
            return False
        return self.req_access == other.req_access


class Exploit(Action):
    """An Exploit action in the environment.

    Inherits from the base Action Class.

    ...

    Attributes
    ----------
    service : str
        the service targeted by exploit
    os : str
        the OS targeted by exploit. If None then exploit works for all OSs.
    access : int
        the access level gained on target if exploit succeeds.
    """

    def __init__(self,
                 name: str,
                 target: Address,
                 cost: float,
                 service: str,
                 os: str | None = None,
                 access: int = 0,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize Exploit.

        Parameters
        ----------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        service : str
            the target service
        os : str, optional
            the target OS of exploit, if None then exploit works for all OS
            (default=None)
        access : int, optional
            the access level gained on target if exploit succeeds (default=0)
        prob : float, optional
            probability of success (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.os = os
        self.service = service
        self.access = access

    def __str__(self) -> str:
        return (f"{super().__str__()}, os={self.os}, "
                f"service={self.service}, access={self.access}")

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, Exploit)
        return self.service == other.service \
            and self.os == other.os \
            and self.access == other.access


class PrivilegeEscalation(Action):
    """A privilege escalation action in the environment.

    Inherits from the base Action Class.

    ...

    Attributes
    ----------
    process : str
        the process targeted by the privilege escalation. If None the action
        works independent of a process
    os : str
        the OS targeted by privilege escalation. If None then action works
        for all OSs.
    access : int
        the access level resulting from privilege escalation action
    """

    def __init__(self,
                 name: str,
                 target: Address,
                 cost: float,
                 access: int,
                 process: str | None = None,
                 os: str | None = None,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize PrivilegeEscalation.

        Parameters
        ----------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        access : int
            the access level resulting from the privilege escalation
        process : str, optional
            the target process, if None the action does not require a process
            to work (default=None)
        os : str, optional
            the target OS of privilege escalation action, if None then action
            works for all OS (default=None)
        prob : float, optional
            probability of success (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.access = access
        self.os = os
        self.process = process

    def __str__(self) -> str:
        return (f"{super().__str__()}, os={self.os}, "
                f"process={self.process}, access={self.access}")

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, PrivilegeEscalation)
        return self.process == other.process \
            and self.os == other.os \
            and self.access == other.access


class ServiceScan(Action):
    """A Service Scan action in the environment.

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target: Address,
                 cost: float,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize ServiceScan.

        Parameters
        ----------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("service_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class OSScan(Action):
    """An OS Scan action in the environment.

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target: Address,
                 cost: float,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize OSScan.

        Parameters
        ----------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("os_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class SubnetScan(Action):
    """A Subnet Scan action in the environment.

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target: Address,
                 cost: float,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize SubnetScan.

        Parameters
        ----------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("subnet_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class ProcessScan(Action):
    """A Process Scan action in the environment.

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target: Address,
                 cost: float,
                 prob: float = 1.0,
                 req_access: AccessLevel = AccessLevel.USER,
                 **kwargs: Any) -> None:
        """Initialize ProcessScan.

        Parameters
        ----------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("process_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class NoOp(Action):
    """A do nothing action in the environment.

    Inherits from the base Action Class
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="noop",
                         target=(1, 0),
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)
