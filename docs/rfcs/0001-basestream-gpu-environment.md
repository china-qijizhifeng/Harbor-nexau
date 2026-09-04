# RFC 0001: Harbor Basestream GPU Environment

## 状态

已实现，真实集群验证中。

## 背景

Terminal-Bench 4.0包含 66 道任务，其中 3 道单容器任务要求 H100：
`fp8-rmsnorm-gemm`、`jax-speedrun-gpu` 和 `math-eval-grader`。当前
`NexAUE2BEnvironment` 与 `NexAUE2BDinDEnvironment` 都明确不提供 GPU；已经用于
PostTrainBench 的 Basestream控制面能够通过 Infrawaves提交 Volcano Job，但不是 Harbor
`BaseEnvironment`，不能承接 Harbor的 Agent、artifact和 separate verifier生命周期。

Basestream的 CPU开发机可访问 NorthJob凭据、Kubernetes只读/exec API、内部 Harbor和用户
GPFS PVC。GPU节点无直接公网，现有 CPU egress proxy提供受控 HTTP(S)出口。集群同时存在
H100与 B300，任务不能只申请通用 `nvidia.com/gpu` 后假设型号正确。

## 目标

- 新增可通过导入路径选择的 `BasestreamEnvironment`；
- Harbor控制器直接运行在 `basestream-cpu`；
- 优先校验并使用任务声明的官方不可变镜像；未声明镜像时原样构建
  `environment/Dockerfile` 或 `tests/Dockerfile`，不修改 TB4任务；
- Agent与 separate verifier各使用独立的 Volcano GPU Job；
- 严格申请 task.toml中的 CPU、内存、GPU与 ephemeral storage；
- H100调度同时使用节点排除规则和运行时型号校验；
- 保留镜像 ENTRYPOINT，只把长驻 shell作为镜像参数；
- Agent不获得 NorthJob、Kubernetes、Harbor或其他 trial凭据；
- 实现 Harbor要求的命令执行与文件上传/下载接口；
- Job、镜像、资源与结果均产生可审计 receipt。

## 非目标

- 不让 E2B直接提供 GPU；
- 不把 PostTrainBench runtime作为 TB4基础镜像；
- 不修改 TB4官方 Dockerfile、task.toml或 verifier；
- 不支持 GPU Compose任务（TB4 4.0的 3 道 GPU任务均为单容器）；
- 不在本机通过长连接 SSH远程托管完整 trial；
- 不把 GPU Manager任意命令队列重新引入默认路径；
- 首版不支持 TPU、Windows或 GPU共享。

## 生命周期

```text
Harbor Trial on basestream-cpu
  ├─ Agent BasestreamEnvironment
  │   ├─ verify official prebuilt digest, then Skopeo preserve-digest copy
  │   ├─ or: CPU Kaniko Volcano Job -> immutable internal task image
  │   ├─ GPU Volcano Job -> long-lived main container
  │   ├─ kubectl exec/upload/download
  │   └─ collect declared artifacts, then delete Job
  └─ Verifier BasestreamEnvironment
      ├─ stage tests/ and build immutable verifier image
      ├─ independent GPU Volcano Job
      ├─ upload only Harbor-collected artifacts
      ├─ run trusted verifier
      └─ delete Job
```

Harbor自身已经在 separate verifier模式下创建第二个 environment、下载 Agent artifacts并只把
声明产物上传到 verifier。因此 provider不复刻评分流程，只实现一个环境实例的完整接口。

## 镜像构建

当 task.toml提供 `docker_image@sha256` 时，provider必须识别无显式 registry前缀的 Docker Hub
引用，通过官方匿名 Bearer token端点读取 manifest/config，逐字节重算两层 digest，并限制 token
realm为 `auth.docker.io`；内部 Harbor引用仍使用既有 Basic auth。Basestream GPU节点的 containerd
不使用Pod内 proxy环境，不能直接拉 Docker Hub。验证完成后，provider复用 NexMLTask已经验证的
固定 Skopeo镜像，通过 CPU egress从 `docker.io`按源digest执行
`copy --all --preserve-digests` 到内部 Harbor。可选镜像源只有在当前egress路径实测可达时才启用；
目标raw manifest必须仍能以同一个digest读取，且
config digest、ENTRYPOINT、USER与WORKDIR必须与官方源完全一致；GPU Job只使用该内部
`@sha256`。这个步骤不是用空 Dockerfile重建镜像。当前egress由单路SSH stdio承载，直接做
registry-to-registry流式复制时，源端Range重连会关闭仍在进行的Harbor PATCH。provider因此先将
源manifest与layer串行写到当前environment hash专属的GPFS `dir:` transport，逐字节重算manifest
digest后，再从本地staging推送Harbor；两个网络阶段完全解耦。成功后删除staging，失败时保留完整
blob供下一attempt复用；默认每阶段最多3次外层尝试、每次Skopeo内部10次重试。

任务未声明镜像时，环境内容 hash决定内部 Harbor tag，同样只运行 Kaniko写回的 `@sha256` 引用。
构建上下文与输出目录位于 provider控制的 GPFS根目录，Kaniko只读挂载当前 context、可写挂载当前
output和只读挂载 Harbor auth。并发构建同一 hash时使用 host file lock；已有 receipt只有在 task
hash、源镜像引用和 registry manifest全部一致时才复用。

镜像 config用于记录原始 `User`。运行 Pod以 root启动以支持 Harbor installed-agent的可信安装
步骤；普通 Agent命令在未显式指定 user时通过 `su`降到镜像原始用户。若镜像用户非 root但
容器不提供可用的 `su/getent`，启动必须 fail closed。

## GPU Job

- `nvidia.com/gpu` requests与 limits均使用任务声明值；
- CPU和内存 requests/limits使用任务声明值；
- `ephemeral-storage` requests/limits使用 `storage_mb`，JAX任务即 `1024000Mi`；
- `/dev/shm` 使用受限内存 emptyDir；
- 主容器不设置 Kubernetes `command`，以保留镜像 ENTRYPOINT；
- `args` 为 `/bin/bash --norc -lc <keepalive>`；
- 禁止 service-account token与 privilege escalation；
- 只挂载当前 trial日志目录和当前 provider session目录；
- 不挂载 NorthJob、Harbor auth、其他 task context或兄弟 trial；
- 公网流量只通过已有 CPU egress proxy；no-network模式不注入 proxy；首版拒绝 allowlist模式。

当前集群没有 GPU型号 node label。首版通过配置排除已知 B300节点，并在容器启动后执行
`nvidia-smi --query-gpu=name`，要求每张可见卡精确包含 `H100`；型号不符立即删除 Job。后续应由
平台为 GPU product提供稳定 label并替换 hostname规则。

## 命令与传输

`exec` 使用 `kubectl exec`，在远端 shell中应用 cwd、环境变量、用户和 `timeout`。上传/下载使用
流式 tar，不把大型 artifact整体读入内存；host侧解包使用安全 data filter，拒绝路径穿越和
特殊文件。所有目标路径均在容器内显式引用，host staging路径限制在当前 session GPFS目录。

## 配置

首版环境参数及对应环境变量：

- namespace、queue、priority class与 PVC；
- NorthJob API地址和凭据文件；
- provider控制根目录及其 PVC subPath；
- Kaniko镜像、cache repository和 Harbor auth subPath；
- 任务镜像 repository前缀；
- CPU egress proxy URL；
- 已知非 H100节点列表；
- Job发现、启动和构建 timeout。

配置不得包含明文 NorthJob或 Harbor密码；只接受权限受限的既有文件路径。

## TB4路由

现有两路 auto-partition扩展为：

1. GPU任务 -> `BasestreamEnvironment`；
2. CPU且 plain E2B预检成功 -> `NexAUE2BEnvironment`；
3. 其余 CPU任务 -> `NexAUE2BDinDEnvironment`。

GPU选择必须依据 task.toml资源字段，而不是题名硬编码。TB4本地数据集解析后再生成三个子 Job；
不在同一个 Harbor Job里混用 environment provider。

## 验证

1. 单元测试覆盖配置、manifest、H100 affinity、1 TB storage、旧/恶意路径、用户切换、传输、
   timeout、Job清理与 separate verifier session命名；
2. registry/build smoke确认官方 task镜像 digest；
3. `math-eval-grader` install-only验证 Agent image；
4. oracle或 nop Agent验证 Agent artifact -> separate verifier闭环；
5. `fp8-rmsnorm-gemm`验证 CUDA 13.2 compat与 H100；
6. `jax-speedrun-gpu`先做环境/磁盘 admission，再决定是否运行完整两小时 verifier。

只有真实 Agent与 verifier Job均完成、reward文件被 Harbor读取且 Job无残留，才能宣称对应任务
已接通；单元测试或镜像构建成功不等同于 TB4全量可运行。
