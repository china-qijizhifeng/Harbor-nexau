# Harbor NexAU

这是一个独立的 NexAU-Harbor 集成仓库。它不包含 Harbor submodule，也不直接修改已安装的
Harbor site-packages：

- Harbor 是普通 Python 依赖，兼容范围固定为 `>=0.21.0,<0.22.0`。
- `uv.lock` 固定每次实际运行的完整依赖版本。
- `harbor-nexau` CLI 在进程内安装窄范围的 Harbor 0.21 兼容补丁；当前补丁只修复 registry
  任务包目录权限受宿主 `umask` 影响的问题。
- NexAU 固定为 v0.4.1 对应的 commit
  `265805b86d89081e8ec9f026ee7807d2be0ee853`。
- runner 会为该版本缺少上限的 LLM 指数退避增加 30 秒硬上限和 25% jitter；可用
  `NEXAU_LLM_RETRY_BACKOFF_MAX_SEC`、`NEXAU_LLM_RETRY_BACKOFF_JITTER_RATIO` 调整。
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

## 运行 TB4（E2B CPU + Basestream H100）

TB4 的混合运行入口是 `scripts/run_tb4.sh`。控制器必须在 `basestream-cpu` 上运行，因为 GPU
provider需要同时访问 NorthJob、集群内 Harbor、`kubectl exec` 和用户 GPFS PVC。先把官方
数据集以 export模式放进 GPFS：

```bash
ssh basestream-cpu
cd /gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau

uv run harbor download terminal-bench/terminal-bench@4.0.0 \
  --export \
  --output-dir /gpfs/users/i-zhangguoqiang/datasets/tb4-4.0.0
```

下载后的直接任务目录是
`/gpfs/users/i-zhangguoqiang/datasets/tb4-4.0.0/terminal-bench`。全量运行示例：

```bash
scripts/run_tb4.sh \
  --path /gpfs/users/i-zhangguoqiang/datasets/tb4-4.0.0/terminal-bench \
  --trace-mode \
  --job-name nexau-tb4
```

从另一台主机提交长任务时，不要让本地monitor长期托管 `ssh basestream-cpu ...`：jumpserver可能在
远端结果已经写完后仍不关闭SSH channel。应让SSH只提交远端本机monitor并立即返回，例如：

```bash
ssh basestream-cpu \
  "python3 /root/.codex/skills/monitor-long-task/scripts/long_task_monitor.py launch \
    --project 'Harbor TB4' --summary 'TB4全量评测' \
    --cwd '/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau' \
    --state-dir '/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau/.basestream/monitors' \
    --timeout-minutes 1440 -- \
    scripts/run_tb4.sh --path \
      /gpfs/users/i-zhangguoqiang/datasets/tb4-4.0.0/terminal-bench \
      --trace-mode --job-name nexau-tb4"
```

入口会解析每道题的 `task.toml` 并生成审计 plan，不按题名硬编码，也不修改数据集：

- CPU题继续交给现有 auto-partition，真实预检后进入普通 E2B或 E2B DinD；
- 请求 H100的单容器题进入
  `nexau_harbor.basestream_environment:BasestreamEnvironment`；
- GPU Agent和 `environment_mode = "separate"` 的 verifier分别创建独立 Volcano Job；
- 官方 `task.toml` 已声明 Docker Hub镜像时，先逐字节校验 manifest/config digest，再按
  NexMLTask同一合同用固定 Skopeo执行 `copy --all --preserve-digests`，通过 CPU egress从官方
  Docker Hub复制到内部 Harbor；源和目标保持相同digest，GPU节点只拉内部 `@sha256`。未声明
  镜像时才由 CPU Kaniko Job原样构建对应 Dockerfile。若经实测可达，也可用
  `HARBOR_BASESTREAM_DOCKER_HUB_MIRROR`指定镜像源；
- Pod不覆盖镜像 `ENTRYPOINT`，并在启动后校验 H100型号、可用磁盘和 service-account token。

GPU并发默认是 1，可用 `HARBOR_NEXAU_BASESTREAM_CONCURRENCY` 调整。`jax-speedrun-gpu` 会原样
申请 `16 CPU / 32 GiB / 1024000 MiB ephemeral-storage / 1 H100`；如果集群没有满足条件的空闲
节点，它会保持排队而不是静默降配。数据集、jobs目录以及 GPU任务实际解析到的路径都必须位于
`HARBOR_BASESTREAM_PVC_HOST_ROOT` 下。完整设计与安全边界见
[`docs/rfcs/0001-basestream-gpu-environment.md`](docs/rfcs/0001-basestream-gpu-environment.md)。

## 运行 TB3 smoke test

以 OpenAI Responses API 为例：

```bash
export OPENAI_API_KEY="..."
export LLM_KIND=openai_responses

uv run harbor-nexau run \
  --dataset terminal-bench/terminal-bench-3@sha256:88433fbcecd1a3f81f7a71bff4cc76c394d0edbefb7e028f90d4109b639fefba \
  --agent nexau_harbor.agent:NexAUAgent \
  --agent-kwarg config_path="$PWD/src/nexau_harbor/configs/terminal_bench/code_agent.yaml" \
  --model gpt-5.5 \
  --n-tasks 1 \
  --n-concurrent 1
```

也可以使用封装脚本：

```bash
scripts/smoke_tb3.sh --job-name nexau-smoke
```

`harbor-nexau` 和 `smoke_tb3.sh` 会在 Harbor 完成任务解析后，只检查解析到
`~/.cache/harbor/tasks/packages` 内的任务树，并把包内目录规范为 `0755`。Harbor 0.21 的 registry
归档只记录文件条目；在宿主 `umask=0077` 时，解包器会把隐式父目录创建为 `0700`，导致独立
verifier 切换到非 root 用户后无法读取 agent 产物。兼容补丁覆盖新下载、缓存命中以及指向 package
cache 的本地 dataset symlink，不跟随包内目录 symlink、不改文件权限，也不修改 package cache
之外的用户本地任务。DinD artifact 的 tar 解包同时保留经安全过滤后的目录 mode；仍使用 Python
`data_filter` 拒绝路径穿越和越界 link，但不再让 agent→verifier 目录权限受宿主 `umask` 影响。

`smoke_tb3.sh` 默认显式选择
[`configs/terminal_bench/code_agent.yaml`](src/nexau_harbor/configs/terminal_bench/code_agent.yaml)，
不会使用 installed agent 为其他评测保留的 fallback 配置。该 profile 按
`tb-agent-configs-prompts-20260606` 配置 4 个编码工具、500 次迭代和两级上下文压缩：
达到 70% 时先压缩旧工具结果并保留最近 8 轮，达到 85% 时再生成历史 summary 并保留
最近 4 轮。同时删除发给模型的 `max_tokens`/
`max_output_tokens`，避免把 `.env` 中的 32000 当成 TB 输出截断上限。Harbor runner 仍会
自动加入追加写入的 JSONL tracer，因此 verifier、周期 checkpoint 和完整 trace 都保持开启，
同时不会在任务容器内累计第二份完整 span 树。

如果目标是采集长轨迹而不是遵守题目自带的 Agent 限时，并运行本地 CPU-only 任务集，
推荐让脚本自动预检并选择 E2B provider：

```bash
scripts/smoke_tb3.sh \
  --auto-partition \
  --trace-mode \
  --path /root/.cache/harbor/tasks/local-datasets/frontier-bench-cpu68-20260813 \
  --e2b-start-ramp-delay-sec 5 \
  --job-name nexau-gpt-5-5-frontier
```

要保持模型和任务不变、仅将 NexAU harness 替换为 Codex，可以增加
`--harness codex`：

```bash
scripts/smoke_tb3.sh \
  --auto-partition \
  --trace-mode \
  --harness codex \
  --path /root/.cache/harbor/tasks/local-datasets/frontier-bench-cpu68-20260813 \
  --e2b-start-ramp-delay-sec 5 \
  --job-name codex-harness-gpt-5-5-frontier
```

该模式使用仓库内的 `TraceSafeCodexAgent`，不会修改 Harbor site-packages，也不会生成
Harbor `trajectory.json`。当前 Distill 对齐配置使用裸模型标识 `gpt-5.5`，因此脚本也会将
`gpt-5.5` 原样传给 Codex CLI。带 provider 前缀的模型仍只会在前缀与 Harbor provider 一致时
去掉第一段。Codex CLI 版本和 reasoning 设置可分别通过 `HARBOR_CODEX_VERSION`、
`HARBOR_CODEX_REASONING_EFFORT`、`HARBOR_CODEX_REASONING_SUMMARY` 固定。

Codex harness 默认使用 CLI `0.148.0` 的标准 function-tool 模式：启用稳定的 `unified_exec`，
显式关闭实验性的 `code_mode`/`code_mode_only` 和 multi-agent namespace tools。这样
`exec_command`、`write_stdin`、`update_plan`、`view_image` 会携带正式 JSON schema 发送给模型，
符合 `gpt-5.5` 的 Distill 配置。标准模式仍会把 `apply_patch` 暴露为 Responses custom tool，
因此每个 sandbox 保留 loopback bridge，仅转换 custom/function 协议，不再引入 code-mode custom
`exec`。bridge 同时将 Codex 侧 advertised model `gpt-5.5` 重写为 NorthGate 实际提供的
`nex-agi/Nex-N2-Pro`，并在响应侧恢复 advertised model；可通过
`HARBOR_CODEX_UPSTREAM_MODEL` 覆盖。provider 同时关闭 WebSocket 和 web search，并显式设置
N2 实际提供的 262144-token context。

这些 Codex 工具模式不是模型别名。脚本默认继续使用上述 N2 兼容配置，但可通过
`HARBOR_CODEX_UNIFIED_EXEC`、`HARBOR_CODEX_CODE_MODE`、
`HARBOR_CODEX_CODE_MODE_ONLY`、`HARBOR_CODEX_MULTI_AGENT`、
`HARBOR_CODEX_RESPONSES_BRIDGE` 和 `HARBOR_CODEX_CONTEXT_WINDOW` 覆盖。比如原生
`gpt-5.6-terra` 使用真实模型名、`code_mode=true`、`code_mode_only=true`、
`unified_exec=false`、`responses_bridge=false` 和 272000-token context；此时不会设置或改写
`upstream_model_name`。

Codex 0.148.0 不会重试直接 HTTP 429，尽管错误文案是 `exceeded retry limit`。Trace-safe
harness 会在最终错误属于 408/429、HTTP 5xx、连接/stream/transport 临时失败时，从首次事件中
读取精确 thread ID，并在同一 sandbox 内用 `codex exec resume <SESSION_ID>` 恢复；不会重新执行
完整 Harbor trial。默认最多追加 8 次恢复，退避从 5 秒指数增长并带随机抖动，单次等待上限
120 秒。可通过 `HARBOR_CODEX_TRANSIENT_RETRY_ATTEMPTS`、
`HARBOR_CODEX_TRANSIENT_RETRY_BASE_DELAY_SEC` 和
`HARBOR_CODEX_TRANSIENT_RETRY_MAX_DELAY_SEC` 调整；设置 attempts=0 可关闭。400/401/403、确定性
404、context 超限、策略错误和普通工具退出不会自动重试。

`--auto-partition` 不会只凭“是否有 Compose 文件”猜测兼容性。Compose 题直接进入 DinD；其余题在
Agent 启动前使用 `NexAUE2BEnvironment` 做一次真实的 Template + sandbox 预检。预检成功的题进入
普通 E2B，确定性的 Dockerfile/E2B 构建不兼容进入 DinD；网络、限流或 provisioning 临时错误会
退避重试，仍无法确认时本次运行中止，不会把它误记成 DinD。预检不安装或运行 NexAU、不请求模型，
且成功构建的 E2B Template 会被正式 job 复用。

分流完成后两组 Harbor 子 job 并发启动，并自动使用全部组内并发：

- `<job-name>-plain-e2b` 使用 `NexAUE2BEnvironment`；
- `<job-name>-dind` 使用 `NexAUE2BDinDEnvironment`。

DinD 子 job 在启动 Harbor 前，会从本轮选中任务的所有 `Dockerfile*` 和 Compose services 自动
收集 Docker Hub 基础/服务镜像，再在一个专用的临时 E2B DinD sandbox 中以默认 2 并发执行
`docker pull`。这会沿正式任务相同的 E2B→registry 路径预热共享 pull-through cache；完成后
会移除该 sandbox 的本地镜像以控制磁盘占用，最后立即删除预热 sandbox，不运行 Agent。
非 Docker Hub registry、本地 `build:` 输出镜像和无法解析的变量引用会跳过，仍由正式 Compose
build 处理。默认每个镜像只尝试 1 次、最多 60 秒；失败时会保留诊断并继续，真正的可用性由
每个正式 DinD sandbox 的官方源 fallback 决定。

无需手写 `--n-tasks`/`--n-concurrent`，也不要同时传 `--env` 或题名过滤器。自动模式目前只接受
已经解析到本机的 `--path` 数据集；它不修改原题。可信分流结果按 environment hash、E2B SDK 版本
和 E2B backend 缓存在 `~/.cache/harbor-nexau/e2b-routes.json`，后续运行直接复用；环境内容或 backend
改变会自动重新预检。需要强制重检时添加 `--refresh-e2b-routing`。预检单题默认最多 7200 秒、3 次
尝试，可分别通过 `HARBOR_NEXAU_E2B_PREFLIGHT_TIMEOUT_SEC` 和
`HARBOR_NEXAU_E2B_PREFLIGHT_ATTEMPTS` 调整；缓存路径可通过
`HARBOR_NEXAU_E2B_ROUTE_CACHE` 覆盖。

预热默认开启但属于 best-effort，可用 `HARBOR_NEXAU_DIND_PREWARM=false` 关闭；并发、单次
pull 上限和尝试数
分别由 `HARBOR_NEXAU_DIND_PREWARM_CONCURRENCY`、
`HARBOR_NEXAU_DIND_PREWARM_TIMEOUT_SEC`、`HARBOR_NEXAU_DIND_PREWARM_ATTEMPTS`
设置，默认 2 并发、60 秒、1 次。只有显式设置
`HARBOR_NEXAU_DIND_PREWARM_REQUIRED=true` 才会因预热失败阻止 Harbor。自动发现要求本次调用
有已解析的本地 `--path`；只给 Harbor Hub `--dataset` 时会明确跳过，不会猜测或修改远端数据集。

`--e2b-start-ramp-delay-sec 5` 不降低组内并发：trial 会立即进入 Harbor 调度，但每个子 job
内部的 E2B sandbox 创建会按 5 秒错峰。预检使用一条创建队列；正式运行的普通 E2B 和 DinD
子 job 各有独立队列。也可通过 `HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC` 设置；`0` 表示关闭
错峰。自动模式会把两个子 job 放进独立进程组，`Ctrl-C` 会同时通知完整的 Harbor 进程树，并给
清理流程最多 90 秒后再强制终止。

trace 模式通过 Harbor 的 `agent.override_timeout_sec` 覆盖每道题自己的 Agent timeout，
默认统一为 36000 秒（10 小时）。这不是基础设施层面的无限运行：当前 E2B 最大生命周期仍为
86400 秒，默认预留的 14 小时用于环境构建、verifier、artifact 和 trace 下载，避免 sandbox
先被回收而丢失轨迹。可以用 `HARBOR_NEXAU_TRACE_TIMEOUT_SEC` 或
`--trace-timeout-sec SECONDS` 调整该值；切换模型仍通过 `HARBOR_NEXAU_MODEL`，trace 模式下
不要另外传 `--agent`、`--model`、`--config`、`--timeout-multiplier` 或
`--agent-timeout-multiplier`。Terminal-Bench profile 的 500 次迭代上限仍然生效。已经启动的
job 无法动态切换到 trace 模式。

Agent 安装阶段使用独立的 `agent.override_setup_timeout_sec`，默认从
`HARBOR_AGENT_SETUP_TIMEOUT_SEC=3600` 读取并写入 trace Job 配置；这与模型解题的10小时上限
无关。仅设置宿主环境变量但不写入 Agent config 不会改变 Harbor 0.21 的默认360秒 setup 限制。
在执行 apt 前，两个 installed agent 还会仅对已安装 `ca-certificates-java` 的 Debian 镜像创建
缺失的 `/etc/ssl/certs/java`，必要时重新配置该包，避免二次 apt 触发官方任务镜像的 keystore
父目录缺失问题；非 Debian 或未安装该包的镜像保持 no-op。

环境构建使用 Harbor 原生 `--environment-build-timeout-multiplier`，脚本默认从
`HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER=4` 读取；例如官方600/900秒的 Compose build
预算会变为2400/3600秒。命令行显式传入同名参数时优先使用显式值。该设置只延长 provisioning，
不改变 Agent 或 verifier timeout，也不修改任务 Dockerfile。

如果只想单独重跑或诊断某一组，仍可手工选择固定 CPU68 分组：

```bash
# 57 道非 Compose 题：强制普通 E2B（仅用于诊断，不做兼容性预检）
scripts/smoke_tb3.sh \
  --trace-mode \
  --non-compose-only \
  --path /root/.cache/harbor/tasks/local-datasets/frontier-bench-cpu68-20260813 \
  --n-tasks 57 \
  --n-concurrent 57 \
  --e2b-start-ramp-delay-sec 5 \
  --job-name nexau-n2-pro-frontier-non-compose

# 11 道 Compose 题：强制 harbor-nexau-dind
scripts/smoke_tb3.sh \
  --trace-mode \
  --compose-only \
  --path /root/.cache/harbor/tasks/local-datasets/frontier-bench-cpu68-20260813 \
  --n-tasks 11 \
  --n-concurrent 11 \
  --e2b-start-ramp-delay-sec 5 \
  --job-name nexau-n2-pro-frontier-compose
```

两个手工模式会自动加入对应的 Harbor 题名过滤器并选择 environment：`--non-compose-only`
使用 `NexAUE2BEnvironment`，`--compose-only` 使用 `NexAUE2BDinDEnvironment`。它们只针对
固定的 CPU68 任务集合，不修改缓存中的原题；为了避免筛选或路由互相覆盖，启用分组模式时
不要再显式传 `--env`、`--include-task-name` 或 `--exclude-task-name`。

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
HARBOR_NEXAU_E2B_DIND_TEMPLATE=e2b-nex/harbor-nexau-dind:v1
HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC=86400
HARBOR_NEXAU_NO_PROXY_HOSTS=e2b.app,.e2b.app,api.e2b.app,sandbox.e2b.app
# 自建 E2B 必须将 API URL、domain 和对应 Key 成套配置；官方 E2B 留空前两项。
E2B_API_URL=
E2B_DOMAIN=
E2B_API_KEY=...
```

只运行确定不含 Compose sidecar 的单容器题时，也可以改回
`nexau_harbor.e2b_environment:NexAUE2BEnvironment`。DinD adapter 仍能把普通单容器题包装为
仅含 `main` 的 Compose project，但整套 CPU68 默认推荐使用上面的自动分组模式，避免普通题
承担不必要的 DinD 镜像拉取、磁盘和启动开销。

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
软件版本记录在 `src/nexau_harbor/e2b_dind/` 下的发布 manifests。每份 manifest 都包含完整
E2B reference；adapter 会据此加载 smoke 实测容量，未知自定义模板仍必须显式提供容量。

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

16 CPU 档不从旧 DinD snapshot 派生：E2B 的 `FROM TEMPLATE` 会继承旧 snapshot 的根盘大小，
即使新项目额度更大或加入 `--skip-cache` 也不会扩盘。以下命令改从固定 Ubuntu 25.04 镜像
安装精确版本的 Docker/containerd/buildx/Compose，让项目当前磁盘额度参与创建新的 ext4
rootfs；它不会覆盖原有 8 CPU/32 GB 档：

```bash
uv run scripts/build_e2b_dind_template.py build-and-smoke \
  --name harbor-nexau-dind-16c \
  --version v1 \
  --cpu-count 16 \
  --memory-mb 16384 \
  --replace \
  --standalone-ubuntu 25.04
```

独立构建固定 Docker 29.2.1、containerd 2.2.1、buildx 0.31.1 和 Compose 5.0.2，并将
Ubuntu/source package 版本写入 provenance。Smoke 会先从 Docker Hub 官方 registry 显式预拉
Alpine，再以 `pull_policy: never` 验证本地 build 与完整 Compose 生命周期，避免 mirror 暂停被
误判为模板失败。

`--trust-pinned-base` 仍只用于旧的 `FROM TEMPLATE` 构建：它跳过无权限完成的公共基座 tag
查询；预期 build ID 会写入 provenance，但不能证明公共 tag 未漂移，因此只应在基座所有者另行
确认版本时使用。

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
耗时的 pull/build 以远端后台进程运行，每 30 秒用短连接轮询；其内部命令不重复应用题目原始
build timeout，而由 Harbor 外层经 multiplier 计算后的 environment-build deadline 统一取消。
遇到 `unexpected EOF`、`short read` 或连接重置时，会在同一 sandbox 内指数退避重试最多
3 次；未完成镜像层的字节计数连续 180 秒不变时，也会保存输出、结束完整 Compose 进程树并
重试。只有 elapsed time 变化不算下载进度，普通编译、包安装和确定性的 Dockerfile/build 错误
不会因此被终止或重试。
正式 DinD sandbox 会先按题目原引用尝试已配置的 Docker Hub mirror；默认 60 秒失败后，改用
`registry-1.docker.io` 官方地址下载并重新 tag 成题目原名称。随后 `docker compose pull` 只拉
本地缺失的 service image，避免再次命中故障 mirror。Mirror、官方源和最终 Compose 输出都会
进入 trial.log。Mirror/官方源上限和同 sandbox 并发分别由
`HARBOR_NEXAU_DIND_MIRROR_PULL_TIMEOUT_SEC`、`HARBOR_NEXAU_DIND_DIRECT_PULL_TIMEOUT_SEC`、
`HARBOR_NEXAU_DIND_IMAGE_PULL_CONCURRENCY` 配置。单个 sandbox 默认串行拉取，避免 `swem-dind`
Docker wrapper 并发初始化 daemon 配置时发生竞争；不同 Harbor trial 仍按 job 并发运行。
镜像 pull/build 阶段临时使用公网，完成后由 E2B 外层网络策略限制整个 sandbox；这样既保留
Compose 内部 DNS/bridge 通信，也支持 Harbor 在不同阶段动态切换 public/no-network/allowlist。

项目当前发布两档 DinD 模板：

| E2B reference | CPU | 内存 | smoke 根盘 |
| --- | ---: | ---: | ---: |
| `e2b-nex/harbor-nexau-dind:v1` | 8 | 32768 MB | 41062 MB |
| `s-project-e89a/harbor-nexau-dind-16c:v1` | 16 | 16384 MB | 55342 MB |

16 CPU 档完整通过 Docker 29.2.1、Compose 5.0.2、网络、命名卷、`exec/cp/stop/down -v` 和
`SYS_PTRACE` smoke。其 E2B build 记录为 59324 MB，运行态根盘为 55342 MB（可用 54694 MB）。
两档是互补而非替代关系：16 CPU 档适合 `16 CPU / 16 GB` Compose 题，原档仍适合需要超过
16 GB、但不超过 8 CPU 的题。

选择已发布的 16 CPU 档时只需传 template reference，adapter 会从 packaged manifest 自动读取
容量：

```bash
--environment-kwarg \
  dind_template=s-project-e89a/harbor-nexau-dind-16c:v1
```

超出所选档位容量会在创建 sandbox 前报错；GPU 题同样会被拒绝。对于不在 packaged manifests
中的自定义模板，仍需同时提供 `dind_cpu_capacity`、`dind_memory_capacity_mb` 和
`dind_storage_capacity_mb`。只有明确接受非标准资源时，才应使用 Harbor 的
`--override-cpus`/`--override-memory-mb`/`--override-storage-mb`。

运行 NexAU 时，自定义 E2B environment 会将远程 Agent 作为独立后台进程启动，保存 PID、
stdout、stderr 和退出码后立即断开初始事件流，再每 30 秒用短连接检查一次状态。暂时的轮询
错误不会重新启动或重复执行 Agent；连续失败才中止 trial。这样本机仍可用 `tmux` 保护
Harbor 进程，同时远程 Agent 不依赖一条贯穿整道题的 HTTP 长连接。

Agent 结束或超时后，DinD adapter 会先从 self-bind 日志目录单独下载
NexAU JSONL，或 Codex events、stderr、result 和完整 native sessions，再打包其余日志。
普通 E2B 同样会先流式恢复这些原始 trace source。大文件通过 E2B 流式接口原子写入宿主机，不再把
整个归档读入内存，也不再对完整传输应用 300 秒总期限；单文件最多重试 5 次，同时最多下载
4 份大 trace，避免大量 trial 同时超时时形成下载尖峰。若其余日志下载失败但 JSONL 已保存，
原始事件和 rollout 仍可独立分析。若两条下载路径都失败，adapter 会在清理前
再试一次；仍失败时不会立即删除含 trace 的 sandbox，而是在日志中记录 sandbox ID，并保留
最长 15 分钟的人工恢复窗口。

单容器 `NexAUE2BEnvironment` 使用的 E2B SDK 2.x Dockerfile 解析器不接受 Docker 原生支持的绝对路径
`COPY --from=ghcr.io/astral-sh/uv:<version> /uv /uvx /usr/local/bin/`。自定义 environment
在构建 Python verifier 模板时会在内存中将该语句转换为相同固定版本的 `uv` 包安装；
Terminal-Bench 任务文件和 Harbor 缓存不会被修改。转换只匹配这个已知的 Astral uv 模式，
其他多阶段复制语句保持原样。

普通 `NexAUE2BEnvironment` 会把题目 `environment.storage_mb`（包括 Harbor 显式传入的
`--override-storage-mb`）写入自托管 E2B v3 template build 的 `diskSizeMB` 字段。模板别名
包含实际请求值，因此不同磁盘规格不会误复用同一个缓存；sandbox 启动后还会用 `df` 核验根盘
总量和可用空间，不满足请求时立即清理并报错。这里不会设置统一的大磁盘默认值：没有声明磁盘
要求的题目继续沿用 E2B 默认容量。

TB3 `layout-config-recreation2` 的官方 Dockerfile 固定到一个已失效的 Hugging Face repository
revision。普通 E2B 只在同时匹配该 revision、repository 和解包命令时，将内存中的 Dockerfile
切换到仍可访问的官方 repository commit，并在解包前校验字体归档的固定 SHA-256；兼容版本也会
进入模板别名，避免误用旧缓存。原始任务目录和 Dockerfile 不会被修改，其他下载地址不会被改写。

E2B 2.x 的 template runtime 可能不恢复 Dockerfile 或基础镜像声明的环境变量；依赖仍安装在
镜像 virtualenv/Conda 环境中，但 Harbor 命令会落到系统 `python`。普通
`NexAUE2BEnvironment` 会按 Dockerfile 声明顺序恢复全部显式 `ENV` 并展开变量；若其中声明
`VIRTUAL_ENV`，还会验证对应 `bin/python` 并前置其 `bin`。对未在 Dockerfile 重申环境的已知
vLLM 基础镜像，会在 sandbox 启动后验证其镜像级 `/opt/venv`。其他隐式环境可通过
`runtime_virtualenv=/absolute/path` 明确指定，或用 `runtime_virtualenv=discover` 要求仅在恰好
发现一个 `pyvenv.cfg` 候选时激活；多候选会 fail closed，不猜测路径。调用方显式传入的
`PATH`/`VIRTUAL_ENV` 始终具有更高优先级。E2B command service 会忽略 SDK `envs` 中的 PATH，
因此 adapter 只把 PATH 作为 shell `export` 前缀，其余变量仍走 SDK env channel，避免把 secret
写入命令文本；原始任务 Dockerfile 不会被修改。

E2B 2.x 的 Dockerfile parser 还会把反斜线续行后的缩进空格保留进多行 `ENV` 值，例如把
`PIP_NO_CACHE_DIR=1` 解析成无效的 `"1    "`，使 pip 在下载前 exit 2。adapter 会在内存中将
可完整解析的多行 `ENV` 规范化为等价单行，并把兼容版本加入 template alias；无法完整解析时
保持原文而不是部分改写。该处理同时作用于 build 和上述 runtime ENV恢复，不修改任务缓存。

Docker 会为多源 `COPY file1 file2 /absolute/dir/` 自动创建目标目录；E2B 2.x 将其拆成多个
单文件 move 时不会创建目录，首个文件会失败。adapter 对可安全解析、无 `--from` 且目标是绝对
目录的多源 COPY，在内存中前置幂等 `mkdir -p`，并以独立 alias 隔离旧模板。JSON COPY、相对
目标、单源及跨 stage COPY 保持原样。

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
- `nexau_full_trace.json`：仅在兼容旧版或自定义 NexAU 仍暴露 `Agent.full_trace` 时生成的旧格式
  对话快照；v0.4.1 不再提供该内存属性。
- `nexau_trace_events.jsonl`：主轨迹来源；运行时逐 span 追加完整输入、输出、错误和 usage，每个事件写入后
  立即 flush，异常退出时保留到最后一次成功写入。
- `nexau_in_memory_tracer.json`：Harbor 下载日志后在宿主机从 JSONL 重建的嵌套 span 树；
  文件名保持兼容，但不再由任务容器中的 InMemoryTracer 维护。
- `nexau_result.json`：NexAU 最终响应。
- `nexau_trace_checkpoint.json`：最近一次 trace checkpoint 的状态和文件清单。
- `trajectory.json`：适配器生成的 ATIF-v1.7 轨迹。

span 事件不依赖周期 checkpoint，每次 start/end 都会追加。运行期间默认每 30 秒只原子刷新
轻量的 checkpoint 状态，不再整体序列化 history 和 span 树；可通过
`NEXAU_TRACE_CHECKPOINT_SEC` 修改状态刷新间隔，设为 `0` 可关闭。正常结束时才写最终 history
和 result；若存在旧格式 full trace，ATIF 优先使用它，否则使用最终 history。无论正常结束、
超时还是进程被强制终止，`nexau_trace_events.jsonl` 都保留终止前已经落盘的完整 span 事件，
Harbor 下载后据此重建嵌套 span 树。

Harbor 0.21.x 负责从本地或云环境统一下载 `/logs/agent`，随后适配器从上述日志回填
token/cost context 并生成 ATIF 文件。每道题（每个 attempt）的评分、异常、耗时和 token
统计位于 `jobs/<job-name>/<trial-name>/result.json`，verifier 原始输出位于同目录的
`verifier/`；`jobs/<job-name>/result.json` 保存整项 job 的聚合统计。

使用 `--harness codex` 时，每个 trial 的 `agent/` 目录使用另一套原始证据：

- `codex_events.jsonl`：`codex exec --json` 的追加式公开事件流；stdout 不与 stderr 混合。
- `codex_stderr.log`：Codex CLI 自身诊断输出。
- `northgate-responses-bridge.log`：标准 `unified_exec` 模式下用于转换 `apply_patch` custom tool
  的 loopback bridge 启动/错误日志。
- `codex_history.jsonl`：Codex 的持久化历史；运行时直接链接到日志目录。
- `sessions/**/rollout-*.jsonl`：全部 Codex native rollout。`CODEX_HOME/sessions` 在运行开始前
  直接链接到 `/logs/agent/sessions`，不是结束后才复制，所以 timeout 时也保留已落盘内容。
- `codex_result.json`：进程状态、trace 完整性、最终公开消息和 usage；硬中断显示为
  `incomplete`，不会伪装成正常完成；`model` 与 `upstream_model` 分别记录 Codex advertised
  model 和 NorthGate 实际请求模型。
- `trace_manifest.json`：上述文件的大小、SHA-256、JSONL 完整性和事件计数。

Codex 模式的可信分析源是 `codex_events.jsonl` 与所有 native rollout；按当前约定不生成
`trajectory.json`。reasoning 仅包含 Codex 公开发出的 summary/event，不包含模型内部隐藏思维链。

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
