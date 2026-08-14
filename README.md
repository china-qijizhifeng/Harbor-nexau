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
  --agent-kwarg config_path="$PWD/src/nexau_harbor/configs/terminal_bench/code_agent.yaml" \
  --model openai/gpt-5 \
  --n-tasks 1 \
  --n-concurrent 1
```

也可以使用封装脚本：

```bash
scripts/smoke_tb3.sh --job-name nexau-smoke
```

`smoke_tb3.sh` 默认显式选择
[`configs/terminal_bench/code_agent.yaml`](src/nexau_harbor/configs/terminal_bench/code_agent.yaml)，
不会使用 installed agent 为其他评测保留的 fallback 配置。该 profile 按
`tb-agent-configs-prompts-20260606` 配置 4 个编码工具、500 次迭代、仅
`tool_result_compaction` 的上下文压缩，并删除发给模型的 `max_tokens`/
`max_output_tokens`，避免把 `.env` 中的 32000 当成 TB 输出截断上限。Harbor runner 仍会
自动加入 `InMemoryTracer`，因此 verifier、周期 checkpoint 和完整 trace 都保持开启。

如果目标是采集长轨迹而不是遵守题目自带的 Agent 限时，可启用 trace 模式：

```bash
scripts/smoke_tb3.sh \
  --trace-mode \
  --path /root/.cache/harbor/tasks/local-datasets/frontier-bench-cpu68-20260813 \
  --n-tasks 68 \
  --n-concurrent 68 \
  --job-name nexau-n2-pro-frontier-trace
```

trace 模式通过 Harbor 的 `agent.override_timeout_sec` 覆盖每道题自己的 Agent timeout，
默认统一为 72000 秒（20 小时）。这不是基础设施层面的无限运行：当前 E2B 最大生命周期仍为
86400 秒，默认预留的 4 小时用于环境构建、verifier、artifact 和 trace 下载，避免 sandbox
先被回收而丢失轨迹。可以用 `HARBOR_NEXAU_TRACE_TIMEOUT_SEC` 或
`--trace-timeout-sec SECONDS` 调整该值；切换模型仍通过 `HARBOR_NEXAU_MODEL`，trace 模式下
不要另外传 `--agent`、`--model`、`--config`、`--timeout-multiplier` 或
`--agent-timeout-multiplier`。Terminal-Bench profile 的 500 次迭代上限仍然生效。已经启动的
job 无法动态切换到 trace 模式。

如果 Harbor Hub 暂时无法解析固定的 TB3 digest，但任务已经缓存到本机，可以直接绕过
Hub 使用本地任务目录：

```bash
scripts/smoke_tb3.sh \
  --path /root/.cache/harbor/tasks/packages/terminal-bench/ico-path-patch/0115a4136189b48da79070f9b3004dc4e0dfc1a60725c5acebdd7f380d037d14 \
  --job-name nexau-smoke-local
```

显式传入 `--path`/`-p` 或 `--dataset`/`-d` 时，脚本不会再附加默认的
`HARBOR_NEXAU_DATASET`，因此本地路径不会触发 Harbor Hub 查询。

脚本会通过 `python-dotenv` 自动读取仓库根目录的 `.env`，不需要先执行
`source .env`。调用脚本前已经导出的环境变量优先于 `.env`；如需读取其他文件，可设置
`HARBOR_NEXAU_ENV_FILE=/absolute/path/to/profile.env`。没有 `.env` 时也可以沿用内联配置：

```bash
HARBOR_NEXAU_MODEL=openai/gpt-5 scripts/smoke_tb3.sh
```

切换模型只需修改 `HARBOR_NEXAU_MODEL` 以及对应的 `LLM_KIND`、API 地址和 Key。切换
harness 时设置 `HARBOR_NEXAU_CONFIG_PATH=/absolute/path/to/code_agent.yaml`；脚本会检查
文件存在后把它作为 `config_path` 传给 adapter。配置目录会整体上传，因此其中所有相对
prompt、工具 YAML 和 binding 都应位于同一目录树内。

本机没有 Docker 时，可在 `.env` 中选择 E2B 并配置对应凭据：

```dotenv
# Frontier Bench/TB3（包括多服务 Compose 题）使用 DinD adapter：
HARBOR_NEXAU_ENV=nexau_harbor.e2b_dind_environment:NexAUE2BDinDEnvironment
HARBOR_NEXAU_E2B_DIND_TEMPLATE=harbor-nexau-dind:v1
HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC=86400
HARBOR_NEXAU_NO_PROXY_HOSTS=e2b.app,.e2b.app,api.e2b.app,sandbox.e2b.app
# 自建 E2B 必须将 API URL、domain 和对应 Key 成套配置；官方 E2B 留空前两项。
E2B_API_URL=
E2B_DOMAIN=
E2B_API_KEY=...
```

只运行确定不含 Compose sidecar 的单容器题时，也可以改回
`nexau_harbor.e2b_environment:NexAUE2BEnvironment`。DinD adapter 会把普通单容器题也作为
仅含 `main` 的 Compose project 运行，因此测试整套混合数据集时无需逐题切换 environment。

此后仍直接运行 `scripts/smoke_tb3.sh`；命令行显式传入的 `--env`/`-e` 会覆盖
`HARBOR_NEXAU_ENV`。仓库内的 E2B environment 使用 3600 秒租期，避免 Harbor 0.21.x
内置 E2B environment 在创建时直接请求 86400 秒而被低额度账户拒绝；运行期间每 60 秒续租
一次。续租不能突破服务端规定的单个 sandbox 最大连续运行时间：本地将
`HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC` 设为 86400，并不会把官方 Base/Hobby 账户的一小时
上限提升到 24 小时。需要长任务时，应使用支持相应时长的官方套餐，或同时配置自建服务的
`E2B_API_URL`、`E2B_DOMAIN` 和匹配的 `E2B_API_KEY`。只替换 Key 而仍连接默认官方地址会造成
认证或额度不匹配。停止环境时会先停止续租再删除 sandbox；若本地进程异常退出，最后一次
续租到期后 E2B 仍会自动回收。E2B 负责运行任务沙箱，远程 API 模型不要求该沙箱配备 GPU。

`HARBOR_NEXAU_NO_PROXY_HOSTS` 会在创建 E2B environment 时合并进宿主进程的
`NO_PROXY` 和 `no_proxy`，保留原有的 localhost 等排除项。它适用于宿主机已经设置本地
HTTP 代理、但 E2B 控制连接需要直连的情况；无需覆盖或清空现有代理变量。

### 构建 E2B DinD 模板

需要在 E2B sandbox 内运行多容器题目时，可以构建本项目维护的
`harbor-nexau-dind:v1` 模板。它固定继承已验证的 HK E2B `swem-dind` build
`1ac1ebce-6cc3-43cf-8c30-1483468bbf38`，并固定 Docker 29.2.1、Docker Compose
5.0.2、Docker Hub 镜像源和启动就绪检查。构建前会再次查询基座的 `default` tag；若它已被
替换，脚本会停止，避免在未检查的基座上静默重建。当前已验证的模板 ID、build ID、资源和
软件版本记录在
[`src/nexau_harbor/e2b_dind/manifest.json`](src/nexau_harbor/e2b_dind/manifest.json)。

`.env` 中配置好成套的 `E2B_API_URL`、`E2B_DOMAIN`、`E2B_API_KEY` 后执行：

```bash
uv sync --locked --extra cloud
uv run scripts/build_e2b_dind_template.py build-and-smoke
```

脚本也通过 `python-dotenv` 加载 `.env`，不需要 `source .env`。首次构建默认分配 8 CPU、
32768 MB 内存；可用 `--cpu-count` 和 `--memory-mb` 调整。模板已存在时，只有显式加入
`--replace` 才会发布新 build。构建先使用一次性 `candidate-*` tag，只有构建完整成功后才会
同时移动 `default`/`v1`，避免失败 build 污染稳定 tag：

```bash
uv run scripts/build_e2b_dind_template.py build-and-smoke --replace
```

只复查当前模板而不重建：

```bash
uv run scripts/build_e2b_dind_template.py smoke
```

冒烟验证会创建一个临时 sandbox，实际执行镜像 build、Compose 网络和命名卷、
`compose exec/cp/stop/down -v` 以及 `SYS_PTRACE` 配置检查；无论成功或失败都会删除该
sandbox。模板只是可复用的 DinD 主机，不会预先包含各道题的 Dockerfile；Harbor 后续应把
当前题目的 environment 上传进 sandbox，再在其中 build/up。

`NexAUE2BDinDEnvironment` 是接入该模板的 Harbor Compose adapter。它会上传每道题的
environment，执行 pull/build/up，把 Agent 的 exec/upload/download 路由到 `main`，并支持
sidecar 的 collect hook、artifact 下载和单服务 stop；独立 verifier 也会在新的 DinD sandbox
中作为单服务 Compose project 运行。停止时执行 `down -v --remove-orphans` 后删除 sandbox。
镜像 pull/build 阶段临时使用公网，完成后由 E2B 外层网络策略限制整个 sandbox；这样既保留
Compose 内部 DNS/bridge 通信，也支持 Harbor 在不同阶段动态切换 public/no-network/allowlist。

默认 adapter 按已发布模板校验上限：8 CPU、32768 MB 内存、41062 MB 总磁盘。超出上限会在
创建 sandbox 前报错；GPU 题同样会被拒绝。若重新发布了更大模板，需要同时用
`--environment-kwarg dind_template=<name:tag>` 和对应的 `dind_cpu_capacity`、
`dind_memory_capacity_mb`、`dind_storage_capacity_mb` 告知 adapter。当前 HK E2B API 会拒绝
超过 8 CPU 的模板，所以声明 16 CPU 的题不能由当前 `v1` 完整满足；只有明确接受降配时才应
使用 Harbor 的 `--override-cpus`/`--override-memory-mb`/`--override-storage-mb`。

运行 NexAU 时，自定义 E2B environment 会将远程 Agent 作为独立后台进程启动，保存 PID、
stdout、stderr 和退出码后立即断开初始事件流，再每 30 秒用短连接检查一次状态。暂时的轮询
错误不会重新启动或重复执行 Agent；连续失败才中止 trial。这样本机仍可用 `tmux` 保护
Harbor 进程，同时远程 Agent 不依赖一条贯穿整道题的 HTTP 长连接。

单容器 `NexAUE2BEnvironment` 使用的 E2B SDK 2.x Dockerfile 解析器不接受 Docker 原生支持的绝对路径
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
