"""The result of performing an action in the NASim environment."""

from typing import Any

from ..utils import Address


class ActionResult:
    """A dataclass for storing the results of an Action.

    These results are then used to update the full state and observation.

    ...

    Attributes
    ----------
    success : bool
        True if exploit/scan was successful, False otherwise
    value : float
        value gained from action. Is the value of the host if successfuly
        exploited, otherwise 0
    services : dict
        services identified by action.
    os : dict
        OS identified by action
    processes : dict
        processes identified by action
    access : dict
        access gained by action
    discovered : dict
        host addresses discovered by action
    connection_error : bool
        True if action failed due to connection error (e.g. could
        not reach target)
    permission_error : bool
        True if action failed due to a permission error (e.g. incorrect access
        level to perform action)
    undefined_error : bool
        True if action failed due to an undefined error (e.g. random exploit
        failure)
    newly_discovered : dict
        host addresses discovered for the first time by action
    """

    def __init__(self,
                 success: bool,
                 value: float = 0.0,
                 services: dict[str, Any] | None = None,
                 os: dict[str, Any] | None = None,
                 processes: dict[str, Any] | None = None,
                 access: Any = None,
                 discovered: dict[Address, bool] | None = None,
                 connection_error: bool = False,
                 permission_error: bool = False,
                 undefined_error: bool = False,
                 newly_discovered: dict[Address, bool] | None = None) -> None:
        """Initialize ActionResult.

        Parameters
        ----------
        success : bool
            True if exploit/scan was successful, False otherwise
        value : float, optional
            value gained from action (default=0.0)
        services : dict, optional
            services identified by action (default=None={})
        os : dict, optional
            OS identified by action (default=None={})
        processes : dict, optional
            processes identified by action (default=None={})
        access : dict, optional
            access gained by action (default=None={})
        discovered : dict, optional
            host addresses discovered by action (default=None={})
        connection_error : bool, optional
            True if action failed due to connection error (default=False)
        permission_error : bool, optional
            True if action failed due to a permission error (default=False)
        undefined_error : bool, optional
            True if action failed due to an undefined error (default=False)
        newly_discovered : dict, optional
            host addresses discovered for first time by action (default=None)
        """
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.processes = {} if processes is None else processes
        self.access = {} if access is None else access
        self.discovered = {} if discovered is None else discovered
        self.connection_error = connection_error
        self.permission_error = permission_error
        self.undefined_error = undefined_error
        if newly_discovered is not None:
            self.newly_discovered = newly_discovered
        else:
            self.newly_discovered = {}

    def info(self) -> dict[str, Any]:
        """Get results as dict of plain Python values.

        The dict only holds ``bool``, ``int``, ``float``, ``str``, lists and
        dicts with string keys, so it can be serialized, e.g. as JSON.

        Returns
        -------
        dict
            ``success``, ``connection_error``, ``permission_error`` and
            ``undefined_error`` as bool; ``value`` as float; ``access`` as
            int (0 = none, 1 = user, 2 = root); ``services``, ``os`` and
            ``processes`` as ``{name: float}``; ``discovered`` and
            ``newly_discovered`` as sorted lists of ``[subnet, host]``
            addresses
        """
        return dict(
            success=bool(self.success),
            value=float(self.value),
            services={k: float(v) for k, v in self.services.items()},
            os={k: float(v) for k, v in self.os.items()},
            processes={k: float(v) for k, v in self.processes.items()},
            access=0 if isinstance(self.access, dict) else int(self.access),  # {} if not set
            discovered=sorted([int(s), int(h)] for (s, h), d in self.discovered.items() if d),
            connection_error=bool(self.connection_error),
            permission_error=bool(self.permission_error),
            undefined_error=bool(self.undefined_error),
            newly_discovered=sorted([int(s), int(h)] for (s, h), d in self.newly_discovered.items() if d)
        )

    def __str__(self) -> str:
        output = ["ActionObservation:"]
        for k, val in self.info().items():
            output.append(f"  {k}={val}")
        return "\n".join(output)
