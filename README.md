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
HARBOR_NEXAU_MODEL=openai/gpt-5 scripts/smoke_tb3.sh
```

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
- `nexau_history.json`：NexAU 对话历史。
- `nexau_in_memory_tracer.json`：LLM tracer 数据。
- `nexau_result.json`：NexAU 最终响应。
- `trajectory.json`：适配器生成的 ATIF-v1.7 轨迹。

Harbor 0.21.x 负责从本地或云环境统一下载 `/logs/agent`，随后适配器从上述日志回填
token/cost context 并生成 ATIF 文件。

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
