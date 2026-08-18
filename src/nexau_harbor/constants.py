from pathlib import PurePosixPath

NEXAU_VERSION = "0.4.1"
NEXAU_GIT_COMMIT = "265805b86d89081e8ec9f026ee7807d2be0ee853"
NEXAU_SOURCE = f"git+https://github.com/nex-agi/NexAU.git@{NEXAU_GIT_COMMIT}"
NEXAU_DETACHED_EXEC_ENV = "_HARBOR_NEXAU_DETACHED_AGENT_EXEC"

REMOTE_WORKSPACE = PurePosixPath("/nexau-workspace")
REMOTE_VENV = PurePosixPath("/opt/nexau-venv")
REMOTE_LOG_DIR = PurePosixPath("/logs/agent")
