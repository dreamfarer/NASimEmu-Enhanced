"""Expose :class:`NASimEmuEnv` over the network via dm_env_rpc.

:func:`serve` starts a gRPC server that wraps the environment;
:class:`RemoteNASimEmuEnv` is a gymnasium client that drives it as if it were
local. Both need the optional ``[network]`` dependencies (``dm-env-rpc`` and
``grpcio``).
"""
from nasimemu_enhanced.network.client import RemoteNASimEmuEnv as RemoteNASimEmuEnv
from nasimemu_enhanced.network.server import NASimEmuServicer as NASimEmuServicer
from nasimemu_enhanced.network.server import serve as serve
