# This is an interface to pymetasploit3, with the currently supported modules as methods and some helper functions.
# By default, the msfrcpd password is 'msfpassword'.

from typing import Any
import requests
from retry import retry
from pymetasploit3.msfrpc import MsfRpcClient, MsfError, PayloadModule
import time, re
import logging

from .msf_modules import MsfModuleMixin, _group1


class _SharedSessionMsfRpcClient(MsfRpcClient):  # type: ignore[misc]
    """MsfRpcClient that reuses a single requests.Session across all RPC calls.

    The upstream client opens a fresh TCP connection for every RPC call; over a
    full episode that is a large amount of needless connection setup. Reusing one
    pooled session is one of NASimEmu's patches, moved here out of the vendored fork.
    """

    def __init__(self, password: str, **kwargs: Any) -> None:
        # Must exist before super().__init__, which logs in (and thus posts) immediately.
        self._session = requests.Session()
        super().__init__(password, **kwargs)

    @retry(tries=3, delay=1, backoff=2)
    def post_request(self, url: str, payload: bytes) -> "requests.Response":
        return self._session.post(url, data=payload, headers=self.headers, verify=False)


class MsfClient(MsfModuleMixin):
    def __init__(self, password: str, lhost: str, host: str = '127.0.0.1', port: int = 55553) -> None:
        self.logger = logging.getLogger("MsfClient")
        self.logger.info(f"Connecting to msfrpcd at {host}:{port}")
        # pymetasploit3 1.0.6 reads the host from the `server` kwarg, not `host`.
        self.client = _SharedSessionMsfRpcClient(password, server=host, port=port, ssl=True)
        self.lhost = lhost

        # create a new console, use only one
        self.console = self.client.consoles.console()

    def get_sessions(self) -> Any:
        return self.client.sessions.list

    # warning: user filtering not working as expected
    # sometimes the user field is populated incorrectly, especially after successful priviledge escalation
    # session_type: shell / meterpreter
    def get_sessions_filtered(self, ip: str | None = None, user: str | None = None, session_type: str | None = None) -> Any:
        sessions = self.get_sessions()

        if ip is not None:
            sessions = {k: v for k, v in sessions.items() if v['target_host'] == ip}

        if user is not None:
            sessions = {k: v for k, v in sessions.items() if v['username'] == user}

        if session_type is not None:
            sessions = {k: v for k, v in sessions.items() if v['type'] == session_type}

        return sessions

    def run_shell_command(self, session_id: Any, cmd: str, os: str = "linux") -> str:
        session = self.get_sessions()[str(session_id)]
        self.logger.info(f"Running `{cmd}` at #{session_id} ({session['session_host']})")

        session_cmd = self.client.sessions.session(str(session_id))

        if os == 'linux':
            command_string = f'{cmd}'
        elif os == 'windows':
            command_string = f'cmd /c "{cmd}"'
        else:
            raise RuntimeError(f"Unknown OS {os}.")

        result = self.run_module(
                module_type='post',
                module_name='multi/general/execute',
                module_params={'COMMAND': command_string, 'SESSION': int(session_id)}
        )

        def find_between(s: str, first: str, last: str) -> str:
            try:
                start = s.index(first) + len(first)
                end = s.index(last, start)
                return s[start:end]
            except ValueError:
                return ""

        result = find_between(result, '[*] Response: ', '[*] Post module execution completed')
        result = result.rstrip()

        return result

    def wait_for_job(self, job_id: Any, timeout: int = 0) -> bool:
        seconds_elapsed = 0

        while True:
            jobs = self.client.jobs.list

            if (timeout > 0) and (seconds_elapsed >= timeout):
                self.client.jobs.stop(job_id)
                return False

            if str(job_id) not in jobs.keys():
                return True

            time.sleep(1)
            seconds_elapsed += 1

    def get_session_id(self, result: str, rhost: str) -> str | None:
        lines = [line for line in result.split('\n') if '[*] Session ' in line]

        if len(lines) == 0:
            self.logger.info("No session created.")
            return None

        session_id = _group1(re.match(r".* Session (\d+) created .*", lines[0]))
        self.logger.info(f"Opened new session #{session_id} for {rhost}")
        return session_id

    def run_module(self, module_type: str, module_name: str, module_params: dict[str, Any], payload_name: str | None = None, payload_params: dict[str, Any] | None = None, forced_params: dict[str, Any] | None = None, run_with_console: bool = True, verbose: bool = False) -> Any:
        self.logger.info(f"Executing {module_type}:{module_name} with params {module_params}")

        module = self.client.modules.use(module_type, module_name)

        for pkey, pval in module_params.items():
            if pkey == 'target':
                module.target = pval
            else:
                module[pkey] = pval

        if verbose:
            module.runoptions['VERBOSE'] = 'true'
            module.runoptions['DEBUG'] = 'true'

        if payload_name is not None:
            payload = self.client.modules.use('payload', payload_name)
            assert payload_params is not None
            for pkey, pval in payload_params.items():
                if pkey == 'encoder':
                    payload.runoptions[pkey] = pval
                else:
                    payload[pkey] = pval
        else:
            payload = None

        if forced_params is not None:
            for pkey, pval in forced_params.items():
                module._runopts[pkey] = pval

        if run_with_console:
            output = self._run_module_with_output(module, payload=payload, timeout=180)

            # print(output)
            self.logger.debug(output)
            return output

        else:
            job = module.execute(payload=payload)
            job_id = job['job_id']
            self.wait_for_job(job_id)

            return None

    def _run_module_with_output(self, module: Any, payload: Any = None, timeout: int = 300) -> str:
        """Run a module on the shared console and collect its output.

        Re-implements pymetasploit3's ``MsfConsole.run_module_with_output`` so that
        NASimEmu's patches live here rather than in a vendored fork:
        - poll every 5 s instead of every 1 s,
        - emit ``unset <opt>`` for options whose value is ``None`` (upstream would
          emit ``set <opt> None``), which lets callers unset options via
          ``forced_params`` (e.g. ``{'LHOST': None}``),
        - wait until the console output stabilises (it is no longer busy and no new
          data arrived) before returning, instead of stopping as soon as it idles once.
        """
        console = self.console

        if console.is_busy():
            raise MsfError(f'Console {console.cid} is busy')
        console.read()  # clear the data buffer

        opts = module.runoptions.copy()
        if payload is None:
            opts['DisablePayloadHandler'] = True

        options_str = f'use {module.moduletype}/{module.modulename}\n'

        # Set module params; a None value means "unset" rather than "set to None".
        for key in opts.keys():
            if opts[key] is not None:
                options_str += f'set {key} {opts[key]}\n'
            else:
                options_str += f'unset {key}\n'

        # Set payload params for exploit modules.
        if module.moduletype == 'exploit':
            opts['TARGET'] = module.target
            options_str += f'set TARGET {module.target}\n'

            if opts.get('DisablePayloadHandler'):
                pass
            elif isinstance(payload, PayloadModule):
                if payload.modulename not in module.payloads:
                    raise ValueError(f'Invalid payload ({payload.modulename}) for given target ({module.target}).')
                options_str += f'set payload {payload.modulename}\n'
                for pkey, pval in payload.runoptions.items():
                    if pval is None or (isinstance(pval, str) and not pval):
                        continue
                    options_str += f'set {pkey} {pval}\n'
            else:
                raise ValueError('No valid PayloadModule provided for exploit execution.')

        options_str += 'run -z'

        # Wait for any previous activity to finish before writing.
        while console.is_busy():
            self.logger.debug("Console busy. Waiting...")
            time.sleep(5)

        console.write(options_str)

        output = ''
        old_data_len = -1
        timer = 0
        time.sleep(5)
        while output == '' or console.is_busy() or old_data_len != len(output):
            time.sleep(5)
            old_data_len = len(output)
            output += console.read()['data']
            timer += 5
            if timer > timeout:
                break

        return output

    def run_msf_command(self, cmd: str) -> str:
        self.logger.info(f"Executing msfconsole command: `{cmd}`")

        console = self.console
        console.read() # clear buffer
        console.write(cmd)

        output = ''
        timer = 0
        timeout = 300
        while output == '' or console.is_busy():
            time.sleep(5)
            output += console.read()['data']
            timer += 5
            if timer > timeout:
                break

        self.logger.debug(output)
        return output

    # create a route to the subnet via session_id (must be meterpreter)
    def add_route(self, ip: str, mask: Any, session_id: Any) -> None:
        cmd = f'route add {ip}/{mask} {session_id}'
        self.run_msf_command(cmd)


if __name__ == '__main__':
    # useful for testing
    # ------------------

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.INFO)

    import random

    msfclient = MsfClient('msfpassword', '192.168.0.100')
    # print(msfclient.get_sessions())
    msfclient.run_shell_command(3, 'DIR C:', os='windows')
    # msfclient.scan_ping_sweep('192.168.0-1.100', 1)

    # msfclient.exploit_wp_ninja_forms_unauthenticated_file_upload('192.168.1.100')
    # msfclient.exploit_wp_ninja_forms_unauthenticated_file_upload('192.168.1.100')
    # msfclient.exploit_elasticsearch_script_mvel_rce('192.168.1.100')

    # msfclient.exploit_drupal_coder_exec('192.168.1.101')
    # msfclient.exploit_proftpd_modcopy_exec('192.168.1.100')
    # msfclient.exploit_phpwiki_ploticus_exec('192.168.1.100')
    # msfclient.privesc_overlayfs_priv_esc('192.168.1.100', 3)
    # msfclient.post_shell_to_meterpreter('192.168.1.100', 8)
