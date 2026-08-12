# Harbor NexAU

这是一个独立的 NexAU-Harbor 集成仓库。它不包含 Harbor submodule，也不复制或修改
Harbor 的 `Task`、`Trial`、Registry 等核心代码：

- Harbor 是普通 Python 依赖，兼容范围固定为 `>=0.21.0,<0.22.0`。
- `uv.lock` 固定每次实际运行的完整依赖版本。
- NexAU 固定为 v0.4.0 对应的 commit
  `114c537dce877c5b521b5c33cc20d6001403ee92`。
- `nexau_harbor.agent:NexAUAgent` 是官方 `BaseInstalledAgent` 的外部实现。
- 旧 `harbor-LJH` 中的 GDPval/TB system prompt、6 工具配置、上下文压缩和
  NexAU history -> ATIF 轨迹转换被迁移到本包；旧 Harbor 核心未迁移。

官方 Harbor 支持这种导入路径形式的自定义 installed agent，见
[Agent 集成文档](https://www.harborframework.com/docs/agents)。TB3 当前数据集标识为
[`terminal-bench/terminal-bench-3`](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-3)。

## 安装

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --locked
```

如需 Daytona、E2B 或 Modal provider：

```bash
uv sync --locked --extra cloud
```

## 运行 TB3 smoke test

以 OpenAI Responses API 为例：

```bash
export OPENAI_API_KEY="..."
export LLM_KIND=openai_responses

uv run harbor run \
  --dataset terminal-bench/terminal-bench-3@sha256:88433fbcecd1a3f81f7a71bff4cc76c394d0edbefb7e028f90d4109b639fefba \
  --agent nexau_harbor.agent:NexAUAgent \
  --model openai/gpt-5 \
  --n-tasks 1 \
  --n-concurrent 1
```

也可以使用封装脚本：

```bash
scripts/smoke_tb3.sh --job-name nexau-smoke
```

脚本会通过 `python-dotenv` 自动读取仓库根目录的 `.env`，不需要先执行
`source .env`。调用脚本前已经导出的环境变量优先于 `.env`；如需读取其他文件，可设置
`HARBOR_NEXAU_ENV_FILE=/absolute/path/to/profile.env`。没有 `.env` 时也可以沿用内联配置：

```bash
HARBOR_NEXAU_MODEL=openai/gpt-5 scripts/smoke_tb3.sh
```

本机没有 Docker 时，可在 `.env` 中选择 E2B 并配置对应凭据：

```dotenv
HARBOR_NEXAU_ENV=nexau_harbor.e2b_environment:NexAUE2BEnvironment
HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC=86400
E2B_API_KEY=...
```

此后仍直接运行 `scripts/smoke_tb3.sh`；命令行显式传入的 `--env`/`-e` 会覆盖
`HARBOR_NEXAU_ENV`。仓库内的 E2B environment 使用 3600 秒租期，避免 Harbor 0.21.x
内置 E2B environment 请求 86400 秒而被一小时限额的账户拒绝；运行期间每 60 秒续租
一次。停止环境时会先停止续租再删除 sandbox；若本地进程异常退出，最后一次续租到期后
E2B 仍会自动回收。E2B 负责运行任务沙箱，远程 API 模型不要求该沙箱配备 GPU。

E2B SDK 2.x 的 Dockerfile 解析器不接受 Docker 原生支持的绝对路径
`COPY --from=ghcr.io/astral-sh/uv:<version> /uv /uvx /usr/local/bin/`。自定义 environment
在构建 Python verifier 模板时会在内存中将该语句转换为相同固定版本的 `uv` 包安装；
Terminal-Bench 任务文件和 Harbor 缓存不会被修改。转换只匹配这个已知的 Astral uv 模式，
其他多阶段复制语句保持原样。

宿主机的 `HTTP_PROXY`/`HTTPS_PROXY` 若指向 `localhost`、回环地址或未指定地址，不会传入
远程 Agent；E2B 中的 `127.0.0.1` 指向 sandbox 自身，而不是宿主机代理。过滤只发生在
Agent 执行环境，宿主机上的 E2B SDK 仍可继续使用该代理；具有远程可达主机名或 IP 的代理
仍会正常转发。

`HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC` 是包含所有续租在内的累计硬上限，默认 86400 秒
（24 小时）。临近上限时 adapter 会缩短最后几次租期，确保不会因最后一次 3600 秒续租
而额外存活近一小时。也可以用
`--environment-kwarg max_lifetime_sec=43200` 对单次运行覆盖为 12 小时。

该参数只会进一步收紧生命周期，不能突破 E2B 账户等级的连续运行限制。
[E2B 官方文档](https://e2b.dev/docs/sandbox) 当前限制 Base 为 1 小时、Pro 为 24 小时，
所以实际有效上限是本参数与账户上限中的较小值；当前会拒绝 86400 秒请求的一小时账户，
即使把本参数设为 86400，也不能连续运行 24 小时。

上述 hash 是本仓库创建时由 Harbor 0.21.0 registry 实际解析到的 75-task TB3
revision。只有在明确验证新 revision 时才设置 `HARBOR_NEXAU_DATASET` 覆盖它，避免
`latest` 在 Alpha 阶段漂移。

适配器会把 Harbor 模型名前缀转换成 SDK 需要的模型名，例如
`openai/gpt-5` -> `gpt-5`。默认 API 类型映射如下：

- Anthropic -> `anthropic_chat_completion`
- Gemini/Google -> `gemini_rest`
- GPT-5、o1、o3、o4 -> `openai_responses`
- 其他 OpenAI 兼容模型 -> `openai_chat_completion`

不符合此映射的代理网关可显式设置 `LLM_KIND`，或传入
`--agent-kwarg api_type=openai_chat_completion`。自定义配置可通过以下参数传入：

```bash
--agent-kwarg config_path=/absolute/path/to/code_agent.yaml
```

默认配置中的 `harbor_reasoning_effort` 是集成层中性字段，runner 会在构建 NexAU agent
前转换为供应商原生结构：OpenAI Responses 的 `reasoning.effort`、支持 effort 的
Anthropic 模型的 `output_config.effort`，以及 Gemini 3+ 的
`thinkingConfig.thinkingLevel`。Gemini 2.5 保持模型的动态 thinking 默认值。若自定义配置
已经显式提供原生字段，以显式字段为准。

配置文件所在目录会整体上传到任务容器的 `/nexau-workspace`，因此相对引用的 prompt、
工具 YAML 和 Python binding 会一并可用。NexAU 在任务容器内安装到独立的
`/opt/nexau-venv`，不会修改任务本身的 Python 环境。

## 轨迹和日志

每个 trial 的 `agent/` 目录会包含可用的以下文件：

- `nexau.txt`：NexAU 标准输出和错误输出。
- `nexau_history.json`：NexAU 最终执行历史，可能包含上下文压缩结果。
- `nexau_full_trace.json`：NexAU 在压缩前保存的完整对话历史。
- `nexau_in_memory_tracer.json`：完整的 LLM/工具 span、输入、输出和 usage。
- `nexau_result.json`：NexAU 最终响应。
- `nexau_trace_checkpoint.json`：最近一次 trace checkpoint 的状态和文件清单。
- `trajectory.json`：适配器生成的 ATIF-v1.7 轨迹。

运行期间默认每 30 秒原子刷新 history 和原始 tracer；可通过
`NEXAU_TRACE_CHECKPOINT_SEC` 修改间隔，设为 `0` 可关闭周期 checkpoint。正常结束后会
再次原子写入最终文件，且 ATIF 优先从未压缩的 `nexau_full_trace.json` 生成。若进程被
强制终止，最近一次 `status=running` 的 checkpoint 仍可用于诊断，但它只代表终止前的
部分轨迹。

Harbor 0.21.x 负责从本地或云环境统一下载 `/logs/agent`，随后适配器从上述日志回填
token/cost context 并生成 ATIF 文件。每道题（每个 attempt）的评分、异常、耗时和 token
统计位于 `jobs/<job-name>/<trial-name>/result.json`，verifier 原始输出位于同目录的
`verifier/`；`jobs/<job-name>/result.json` 保存整项 job 的聚合统计。

可以按题输出结果摘要：

```bash
find jobs/<job-name> -mindepth 2 -maxdepth 2 -name result.json \
  -exec jq -c '{task: .task_name, trial: .trial_name, reward: .verifier_result.rewards, error: .exception_info.exception_type}' {} \;
```

## 跟进 Harbor 更新

同一 0.21.x 兼容范围内升级：

```bash
uv lock --upgrade-package harbor
uv sync --locked
uv run ruff check .
uv run pytest
```

仓库内的 Dependabot 配置每周检查 uv 和 GitHub Actions 依赖。Harbor 发布 0.22.x
时，应先把 `pyproject.toml` 的上界提升到 `<0.23.0`，生成新锁文件，再运行单元测试和
一条真实 TB3 smoke task；不要让生产测评环境在未验证时自动漂移。

升级 NexAU 时需要同时修改：

1. `pyproject.toml` 中的 Git commit；
2. `src/nexau_harbor/constants.py` 中的版本、commit 和安装源；
3. runner/config 单元测试与 TB3 smoke test。

## 本地验证

```bash
uv sync --locked
uv run ruff check .
uv run pytest
uv build
```

单元测试验证官方 Harbor 0.21.x 的真实接口、容器安装/运行命令、默认 NexAU 配置和
ATIF 轨迹转换。完整 TB3 运行仍需要可用的 Docker 或云 provider、模型凭据，以及任务
所需的 GPU/多容器资源；单元测试通过不等同于排行榜级端到端验证。
