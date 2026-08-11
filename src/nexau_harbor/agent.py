from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from nexau_harbor.constants import (
    NEXAU_SOURCE,
    NEXAU_VERSION,
    REMOTE_LOG_DIR,
    REMOTE_VENV,
    REMOTE_WORKSPACE,
)
from nexau_harbor.trajectory import populate_context_from_logs


class NexAUAgent(BaseInstalledAgent):
    """Run a pinned NexAU release inside a Harbor task environment."""

    SUPPORTS_ATIF = True
    MODEL_CONNECTION = ModelConnectionSpec(
        api_key_envs=("LLM_API_KEY",),
        base_url_envs=("LLM_BASE_URL",),
    )

    _FORWARDED_PREFIXES = (
        "ANTHROPIC_",
        "BP_HTML_PARSER_",
        "BRAVE_",
        "GDPVAL_",
        "LANGFUSE_",
        "LLM_",
        "OPENAI_",
    )
    _FORWARDED_EXACT = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NEXAU_TRACE_CHECKPOINT_SEC",
        "NO_PROXY",
        "SERPER_API_KEY",
    )

    def __init__(
        self,
        config_path: str | Path | None = None,
        nexau_source: str = NEXAU_SOURCE,
        nexau_version: str = NEXAU_VERSION,
        python_version: str = "3.13",
        reasoning_effort: str | None = None,
        api_type: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._api_base = kwargs.pop("api_base", None)
        self._model_info = kwargs.pop("model_info", None)
        explicit_version = kwargs.pop("version", None)
        super().__init__(
            *args,
            version=explicit_version or nexau_version,
            **kwargs,
        )

        selected_config = (
            Path(config_path).expanduser()
            if config_path is not None
            else Path(__file__).parent / "configs" / "gdpval" / "code_agent.yaml"
        )
        selected_config = selected_config.resolve()
        if not selected_config.is_file():
            raise FileNotFoundError(f"NexAU config file not found: {selected_config}")

        self._config_path = selected_config
        self._config_dir = selected_config.parent
        self._remote_config_path = REMOTE_WORKSPACE / selected_config.name
        self._nexau_source = nexau_source
        self._python_version = str(python_version)
        self._reasoning_effort = (
            reasoning_effort
            or self._get_env("LLM_REASONING_EFFORT")
            or "medium"
        )
        self._api_type = api_type

    @staticmethod
    @override
    def name() -> str:
        return "nexau"

    @override
    def get_version_command(self) -> str | None:
        python = REMOTE_VENV / "bin" / "python"
        return (
            f"{python} -c "
            "'import importlib.metadata as m; print(m.version(\"nexau\"))'"
        )

    def _forwarded_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for prefix in self._FORWARDED_PREFIXES:
            for suffix, value in self._get_env_prefixed(prefix).items():
                env[f"{prefix}{suffix}"] = value
        for name in self._FORWARDED_EXACT:
            value = self._get_env(name)
            if value is not None:
                env[name] = value
        return env

    def _provider(self) -> str | None:
        provider = self.model_connection.provider
        if provider:
            return provider
        if self.model_name and "/" in self.model_name:
            return self.model_name.split("/", 1)[0].lower()
        return None

    def _runtime_model_name(self, provider: str | None) -> str:
        model = self.model_name or self._get_env("LLM_MODEL")
        if not model:
            raise ValueError(
                "No model configured. Pass Harbor -m/--model or set LLM_MODEL."
            )

        if "/" not in model:
            return model
        prefix, unqualified = model.split("/", 1)
        normalized_prefix = prefix.lower().replace("-", "_")
        normalized_provider = (provider or "").lower().replace("-", "_")
        if normalized_prefix == normalized_provider:
            return unqualified
        aliases = {
            "anthropic": {"anthropic"},
            "google": {"google", "gemini"},
            "openai": {"openai"},
        }
        if provider in aliases and prefix.lower() in aliases[provider]:
            return unqualified
        return model

    def _default_api_type(self, provider: str | None, model: str) -> str:
        configured = self._api_type or self._get_env("LLM_KIND")
        if configured:
            return configured
        if provider == "anthropic":
            return "anthropic_chat_completion"
        if provider in {"google", "gemini"}:
            return "gemini_rest"
        if provider == "openai" and model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
            return "openai_responses"
        return "openai_chat_completion"

    def _runtime_env(self) -> dict[str, str]:
        access = self.model_connection
        provider = self._provider()
        model = self._runtime_model_name(provider)
        env = self._forwarded_env()
        env.update(access.env)

        if access.api_key is not None:
            env["LLM_API_KEY"] = access.api_key
        if access.base_url is not None:
            env["LLM_BASE_URL"] = access.base_url
        elif self._api_base:
            env["LLM_BASE_URL"] = str(self._api_base)

        is_hosted_vllm = (self.model_name or "").lower().startswith("hosted_vllm/")
        if not env.get("LLM_API_KEY") and is_hosted_vllm:
            env["LLM_API_KEY"] = "dummy-key-for-local-vllm"
        if not env.get("LLM_API_KEY"):
            raise ValueError(
                f"No API key found for model {self.model_name!r}. Set LLM_API_KEY "
                "or the provider-specific key supported by Harbor."
            )
        if not env.get("LLM_BASE_URL"):
            raise ValueError(
                f"No API base URL found for model {self.model_name!r}. Set LLM_BASE_URL."
            )

        api_type = self._default_api_type(provider, model)
        env.update(
            {
                "LLM_MODEL": model,
                "LLM_KIND": api_type,
                "LLM_REASONING_EFFORT": self._reasoning_effort,
                "LLM_MAX_TURNS": env.get("LLM_MAX_TURNS", "300"),
                "LLM_CONTEXT_WINDOW_TOKENS": env.get(
                    "LLM_CONTEXT_WINDOW_TOKENS", "200000"
                ),
                "LLM_MAX_TOKENS": env.get("LLM_MAX_TOKENS", "32000"),
                "LLM_TIMEOUT_SEC": env.get("LLM_TIMEOUT_SEC", "600"),
                "LLM_MAX_RETRIES": env.get("LLM_MAX_RETRIES", "3"),
                "NEXAU_TRACE_CHECKPOINT_SEC": env.get(
                    "NEXAU_TRACE_CHECKPOINT_SEC", "30"
                ),
                "GDPVAL_TOOL_RESULT_MAX_CHARS": env.get(
                    "GDPVAL_TOOL_RESULT_MAX_CHARS", "10000"
                ),
                "GDPVAL_NEXAU_PRIMARY_COMPACTION_THRESHOLD": env.get(
                    "GDPVAL_NEXAU_PRIMARY_COMPACTION_THRESHOLD", "0.70"
                ),
                "GDPVAL_NEXAU_FALLBACK_COMPACTION_THRESHOLD": env.get(
                    "GDPVAL_NEXAU_FALLBACK_COMPACTION_THRESHOLD", "0.85"
                ),
                "GDPVAL_NEXAU_KEEP_ITERATIONS": env.get(
                    "GDPVAL_NEXAU_KEEP_ITERATIONS", "3"
                ),
                "GDPVAL_NEXAU_KEEP_USER_ROUNDS": env.get(
                    "GDPVAL_NEXAU_KEEP_USER_ROUNDS", "0"
                ),
                "GDPVAL_NEXAU_SUMMARY_MODEL": env.get(
                    "GDPVAL_NEXAU_SUMMARY_MODEL", model
                ),
                "GDPVAL_NEXAU_SUMMARY_BASE_URL": env.get(
                    "GDPVAL_NEXAU_SUMMARY_BASE_URL", env["LLM_BASE_URL"]
                ),
                "GDPVAL_NEXAU_SUMMARY_API_KEY": env.get(
                    "GDPVAL_NEXAU_SUMMARY_API_KEY", env["LLM_API_KEY"]
                ),
                "GDPVAL_NEXAU_SUMMARY_API_TYPE": env.get(
                    "GDPVAL_NEXAU_SUMMARY_API_TYPE", api_type
                ),
                "PROJECT_ROOT": REMOTE_WORKSPACE.as_posix(),
                "PYTHONPATH": REMOTE_WORKSPACE.as_posix(),
                "SANDBOX_WORK_DIR": ".",
                "VIRTUAL_ENV": REMOTE_VENV.as_posix(),
                "PATH": (
                    f"{REMOTE_VENV}/bin:/usr/local/sbin:/usr/local/bin:"
                    "/usr/sbin:/usr/bin:/sbin:/bin"
                ),
            }
        )

        env.setdefault("OPENAI_API_KEY", env["LLM_API_KEY"])
        env.setdefault("OPENAI_BASE_URL", env["LLM_BASE_URL"])
        return env

    def _execution_timeout(self) -> int | None:
        for name in (
            "HARBOR_AGENT_TIMEOUT_SEC",
            "E2B_AGENT_TIMEOUT_SEC",
            "GDPVAL_E2B_TIMEOUT_SEC",
        ):
            raw = self._get_env(name)
            if not raw:
                continue
            try:
                value = int(float(raw))
            except ValueError:
                continue
            if value >= 0:
                return value
        return None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(
            environment,
            ("curl", "git", "build_tools", "ripgrep", "procps", "ca_certificates"),
        )

        agent_user = shlex.quote(str(environment.default_user or "root"))
        await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {REMOTE_WORKSPACE} {REMOTE_VENV} && "
                f"chown -R {agent_user} {REMOTE_WORKSPACE} {REMOTE_VENV}"
            ),
        )

        await environment.upload_dir(
            source_dir=self._config_dir,
            target_dir=REMOTE_WORKSPACE.as_posix(),
        )
        runner_path = Path(__file__).with_name("runner.py")
        await environment.upload_file(
            source_path=runner_path,
            target_path=(REMOTE_WORKSPACE / "runner.py").as_posix(),
        )
        await self.exec_as_root(
            environment,
            command=f"chown -R {agent_user} {REMOTE_WORKSPACE}",
        )

        source = shlex.quote(self._nexau_source)
        python_version = shlex.quote(self._python_version)
        await self.exec_as_agent(
            environment,
            command=(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv python install {python_version} && "
                f"uv venv --clear {REMOTE_VENV} --python {python_version} && "
                f"uv pip install --python {REMOTE_VENV}/bin/python {source}"
            ),
        )

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        command = shlex.join(
            [
                (REMOTE_VENV / "bin" / "python").as_posix(),
                (REMOTE_WORKSPACE / "runner.py").as_posix(),
                "--config-path",
                self._remote_config_path.as_posix(),
                "--log-dir-path",
                REMOTE_LOG_DIR.as_posix(),
                "--query",
                instruction,
            ]
        )
        output_path = (REMOTE_LOG_DIR / "nexau.txt").as_posix()
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {REMOTE_LOG_DIR} && "
                f"{command} 2>&1 </dev/null | tee {shlex.quote(output_path)} >/dev/null"
            ),
            env=self._runtime_env(),
            timeout_sec=self._execution_timeout(),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        populate_context_from_logs(
            logs_dir=self.logs_dir,
            context=context,
            agent_name=self.name(),
            agent_version=self.version() or NEXAU_VERSION,
            model_name=self.model_name,
            reasoning_effort=self._reasoning_effort,
            logger=self.logger,
        )


NexAU = NexAUAgent
