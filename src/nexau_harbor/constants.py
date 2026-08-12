from pathlib import PurePosixPath

NEXAU_VERSION = "0.4.0"
NEXAU_GIT_COMMIT = "114c537dce877c5b521b5c33cc20d6001403ee92"
NEXAU_SOURCE = f"git+https://github.com/nex-agi/NexAU.git@{NEXAU_GIT_COMMIT}"
NEXAU_DETACHED_EXEC_ENV = "_HARBOR_NEXAU_DETACHED_AGENT_EXEC"

REMOTE_WORKSPACE = PurePosixPath("/nexau-workspace")
REMOTE_VENV = PurePosixPath("/opt/nexau-venv")
REMOTE_LOG_DIR = PurePosixPath("/logs/agent")
