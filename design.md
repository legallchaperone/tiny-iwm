# Video World Model Research Repository Design

> 状态：第一版系统设计，尚未实现。
>
> 更新日期：2026-09-21。
>
> 用途：从 research-template 建立研究仓库，作为后续实现、代码复用和实验审查的共同依据。

## 1. 研究目标与设计边界

本项目构建一个规模适中、变量可控的视频 world model，用于研究：

**输入 representation → DiT 内部表示与动态 → 长期 autoregressive rollout → revisit consistency。**

研究变量可以是 encoder/latent、内部 representation regularization，也可以是经过明确对照的后续机制。目标是获得可解释、可重复的实验，而非追求最大生成质量或实时部署速度。

仓库必须支持多 encoder、多 seed、完整一分钟 rollout、layer-wise probing 和大量 ablation。模型、训练、生成、评估与 probing 应相互解耦，但共享同一份模型实现和同一条生成路径。

### 1.1 已确定的基线

| 项目 | 决定 |
|---|---|
| 仓库基础 | 沿用 research-template 的 Hydra、Experiment 和 Lightning 骨架 |
| DiT | 约 250M–400M，目标约 300M；joint spatiotemporal softmax attention |
| DiT 初始化 | 随机初始化，不使用 pretrained DiT weights；允许复用已有模型代码 |
| 视觉 representation | 第一版使用冻结的视频 autoencoder；后续通过统一接口比较 VAE/RAE/V-RAE 类方案 |
| 相机条件 | 视频版 PRoPE，以 Matrix-Game 3.5 的 tiled PRoPE 实现为参考 |
| Stage A | 从头训练带相机条件的 bidirectional native Flow Matching 模型 |
| Stage B | 从本项目 Stage A 权重初始化，进行 teacher-forced chunk-causal FM 训练 |
| 时间设置 | 正式训练对齐 SANA-WM 公开配置的 961 RGB 帧、16 FPS；正式 benchmark 生成完整约 60 秒序列 |
| 历史可见性 | 在基线的一分钟 episode 内保留已有历史；不默认采用短滑动窗口 |
| 评估 | 使用 SANA-WM 官方协议；支持明确、可复现的子集 |
| 主实验排除项 | Diffusion Forcing 的逐帧噪声配方、few-step distillation、self-rollout post-training、额外 retrieval/memory/anchor 机制 |

冻结的 VAE 或文本 encoder 可以使用预训练权重。“从头训练”指 DiT，而不是要求重新训练所有基础组件。论文中应明确这一范围。

普通的历史帧条件和等价的 KV cache 属于 autoregressive 模型本身。第一帧是生成任务的初始条件；不额外引入周期性重注入、检索旧帧或特殊 anchor 选择策略。

### 1.2 与上游工作的关系

本项目参考 SANA-WM 的公开数据、时间设置、阶段划分和评估协议，**不声称逐项复现其完整训练 recipe**。当前锁定版本的 SANA-WM chunk-causal 配置包含 `task: df` 和 chunk timestep mixture；其公开训练数据也不等于内部训练使用的全部数据混合。我们明确采用简化的 native FM + teacher forcing 方案。[SANA 训练配置][sana-causal-config]、[官方说明][sana-docs]

同样，使用 Matrix-Game 的 PRoPE 不意味着使用其 Patch Memory；使用 Causal Forcing 的 mask/cache 代码不意味着采用其完整训练或蒸馏流程。

### 1.3 尚需测量或明确的参数

以下内容保留为显式配置，在正式实验前固定并记录，不在实现时隐式决定：

- DiT 的 depth、width、head 数和 patch size，以及实际可训练参数量。
- 训练空间分辨率、batch size、梯度累积与分布式策略。
- 具体 VAE 权重、编码因果模式、归一化统计。
- chunk 对应的物理时长与 latent 帧数；比较 encoder 时优先固定物理时间含义。
- 文本条件是否启用；若启用，固定文本 encoder、caption 处理与 dropout/CFG 策略。
- FM 时间采样、solver、采样步数、学习率、训练预算、EMA 规则。
- 正式数据规模、固定 validation split、评估子集与 seeds。

预算不足时，应先测量 token 数和 attention/activation/KV 开销，再调整空间分辨率、patch 或执行策略。短序列可用于单元测试，但不能代替一分钟正式协议。

## 2. 系统组成与依赖方向

整个系统分成两条主路径。

**训练路径：**

`main → Experiment → Dataset/Representation → TrainingBatch → DiT → FM/Regularizer → Checkpoint`

**生成与分析路径：**

`Checkpoint + 条件清单 → Rollout → 视频/latent/metadata → 官方评估`

`DiT 中间特征 → Probe capture → 特征摘要或受控 replay → 表示分析`

依赖约束如下：

1. `experiments/` 负责调度任务，不实现 attention 或 loss。
2. DiT 是普通 `nn.Module`，不依赖 Lightning、Dataset、评估器或日志平台。
3. `algorithm.py` 组织训练 step；`runtime/` 管理执行环境与生命周期。
4. `inference/` 使用同一个 DiT，不维护推理专用的第二份模型。
5. Evaluation 与 probing 调用同一条 rollout 路径。
6. `core/` 只定义共享数据协议、时间布局与几何约定，不反向依赖训练或评估模块。
7. Scripts 是轻量入口，不承载只有该脚本才有的核心业务逻辑。

## 3. 目标仓库结构

以下是逐步实现的目标结构；未来扩展不要求立即创建空文件。

```text
research-repo/
├── main.py                              # 模板入口：Hydra 配置与任务选择
├── pyproject.toml                       # 依赖、工具与测试配置
├── README.md
├── LICENSE
├── design.md
│
├── configurations/
│   ├── config.yaml                     # 组合各配置组
│   ├── experiment/world_model.yaml     # 任务选择与执行顺序
│   ├── algorithm/world_model.yaml      # optimizer、EMA、训练设置
│   ├── model/joint_dit_300m.yaml        # 网络规模与 patch/head 设置
│   ├── stage/
│   │   ├── bidirectional.yaml
│   │   └── causal_tf.yaml
│   ├── representation/vae.yaml         # codec、权重、编码方式、归一化
│   ├── conditioning/camera_prope.yaml  # PRoPE 布局、坐标与尺度策略
│   ├── dataset/
│   │   ├── sana_train.yaml
│   │   └── sana_benchmark.yaml
│   ├── rollout/minute.yaml             # 长度、solver、steps、CFG、历史策略
│   ├── evaluation/
│   │   ├── full.yaml
│   │   └── subset.yaml
│   ├── probing/
│   │   ├── off.yaml
│   │   └── standard.yaml
│   ├── runtime/
│   │   ├── single_gpu.yaml
│   │   └── distributed.yaml
│   └── sweep/                          # 多 seed 与受控 ablation
│
├── experiments/
│   ├── exp_base.py                     # 模板基类
│   └── world_model.py                  # train/sample/evaluate/probe 任务调度
│
├── core/
│   ├── types.py                        # 模块间的数据对象
│   ├── video_layout.py                 # RGB/latent/token/chunk 时间与空间映射
│   └── camera.py                       # 坐标变换、内参更新与相机几何
│
├── algorithms/
│   ├── common/                         # 模板公共训练基类
│   └── world_model/
│       ├── algorithm.py                # LightningModule
│       ├── training_batch.py           # Stage A/B 输入、噪声与可见性
│       ├── flow.py                     # 统一的 FM 数学约定
│       ├── regularizers.py             # 表示约束，默认关闭
│       └── models/
│           ├── dit.py                  # 唯一 DiT 主体
│           ├── blocks.py               # attention/MLP/norm/modulation
│           ├── attention.py            # joint attention、mask、KV 接口
│           ├── position.py             # 时空 RoPE 与全局时间位置
│           ├── prope.py                # 相机投影对 Q/K/V/output 的变换
│           ├── conditioning.py         # timestep、可选文本等条件
│           └── latent_io.py            # patchify、输入/输出投影、unpatchify
│
├── representations/
│   ├── base.py                         # encode/decode/spec 接口
│   ├── video_vae.py                    # 第一种视频 VAE 适配器
│   ├── normalization.py               # 固定 latent 统计与可逆变换
│   └── diagnostics.py                 # 重建与 latent 统计检查
│
├── datasets/sana_wm/
│   ├── manifest.py                     # 样本索引、split、来源与版本
│   ├── reader.py                       # 视频/ZIP/latent/camera 读取
│   ├── transforms.py                   # 图像处理与同步相机变换
│   ├── dataset.py                      # 训练/验证 Dataset
│   ├── collate.py                      # batch、padding、valid mask
│   ├── benchmark.py                    # 官方生成条件与 revisit metadata
│   └── prepare.py                      # 建索引、验证、离线缓存
│
├── controls/
│   └── camera_controller.py            # 按键/鼠标 → camera trajectory
│
├── inference/
│   ├── rollout.py                      # 唯一 autoregressive 生成循环
│   ├── history.py                      # 历史 latent/KV 生命周期
│   ├── solver.py                       # FM 数值积分
│   └── writer.py                       # 视频、latent、生成元数据
│
├── evaluation/
│   ├── selection.py                    # 固定场景/split/seed 清单
│   ├── runner.py                       # 完整性检查与指标调度
│   ├── sana_adapter.py                 # 官方文件格式与脚本适配
│   └── aggregate.py                    # 逐场景、split、seed 汇总
│
├── probing/
│   ├── capture.py                      # 特征采集位置与采样策略
│   ├── replay.py                       # 固定条件下的受控前向对比
│   ├── metrics.py                      # norm/rank/similarity/drift
│   └── store.py                        # 特征摘要、原始样本与索引
│
├── runtime/
│   ├── trainer.py                      # Lightning Trainer 与并行策略
│   ├── checkpoint.py                   # init_from / resume_from
│   ├── callbacks.py                    # EMA、保存、日志、定期验证
│   └── preflight.py                    # 配置校验与完整长度 profiling
│
├── third_party/
│   ├── sana_wm_metrics/                # 固定版本的官方评估代码
│   ├── UPSTREAMS.yaml                  # 来源、commit、许可、本地修改
│   └── licenses/
├── scripts/                            # 轻量启动与准备入口
├── tests/
├── docs/                               # 数据协议、实验协议、实现决策
├── utils/                              # 保留模板通用工具
├── data/                               # 原始数据与缓存，通常不进 Git
└── outputs/                            # 实验产物，通常不进 Git
```

新增 representation 时再增加相应适配器。第一版不实现蒸馏 trainer、memory manager 或通用插件框架。

## 4. 共享数据协议

接口应显式携带时间、几何与来源信息，避免依靠 tensor shape 猜测含义。

| 对象 | 最低职责 |
|---|---|
| `VideoBatch` | 样本 ID、视频或 latent、camera、可选文本、有效区域、时间戳与数据来源 |
| `LatentSpec` | 通道数、时空压缩规则、normalization、codec 标识、时间因果性与编码策略 |
| `VideoLayout` | RGB、latent、patch token、chunk 之间的索引关系、真实时间与 padding |
| `CameraCondition` | 原始 c2w、对应内参、时间戳、单位尺度、预处理和参考坐标信息 |
| `TrainingBatch` | 带噪输入、干净条件、FM 时间、target、loss mask、attention 可见性描述 |
| `RolloutResult` | 生成文件与 latent 索引、checkpoint、seed、条件清单和完整推理设置 |
| `ProbeEvent` | 层、采集位置、rollout 时间、FM 时间、token 类别、分支、前向用途 |

视觉 tensor 的存储顺序必须统一并在接口中声明，例如 codec 接口采用 `[B,C,T,H,W]`，DiT 内部采用 `[B,N,D]`。上游不同排列由适配层一次性转换。

### 4.1 唯一的时间布局来源

`video_layout.py` 负责以下映射，其他模块只查询结果：

- 首帧如何编码、每个 latent 覆盖哪些 RGB 帧。
- 时间压缩和 patchify 后，各 token 属于哪个物理时间区间。
- chunk 起止、首 chunk 的特殊长度、末尾补齐。
- 每个 token 应使用哪些 camera 样本及 RoPE 时间坐标。
- 解码后如何恢复官方帧数与初始条件帧约定。

训练窗口按 RGB 帧/物理时间定义；latent 帧数由 codec 推导。不得在 reader、mask、camera 下采样和 writer 中各写一份固定 stride 逻辑。

## 5. 数据与 representation 管线

### 5.1 数据流

`原始视频 + camera/caption → manifest → 验证与预处理 → 编码/缓存 → Dataset → collate → VideoBatch`

- `manifest.py` 固定样本清单、数据来源、split 与数据版本。
- `reader.py` 只负责读文件及基础格式解析；参考 SANA reader 的 ZIP/latent/camera 操作。
- `transforms.py` 进行 resize/crop，并同步更新 camera intrinsics。
- `dataset.py` 根据 manifest 选择样本和时间窗口，不自行决定模型 chunk 规则。
- `collate.py` 输出显式的 valid mask，避免 padding 被当作有效训练或评估内容。
- `prepare.py` 构建离线缓存并记录失败样本；训练期间不隐式修改缓存。

缺失或错误 camera 不得静默替换成 identity pose 后继续当作正常监督。SANA reader 中与其工程环境相关的 fallback、文件大小启发式和 registry 依赖不应原样继承。[上游 reader][sana-reader]

训练/验证划分应考虑源视频或场景身份，避免重叠片段跨 split。扩展数据规模时使用固定、可追溯的嵌套清单，而不是每次重新抽样。

### 5.2 Representation 接口与缓存

`representations/base.py` 规定 encode、decode 和 spec 接口。第一版由 `video_vae.py` 适配所选官方 VAE；不自行重新实现编码器。Encoder、decoder 和 normalization 作为一个带版本的 representation system 管理。

可训练的输入/输出投影属于 `models/latent_io.py`，计入 DiT 的可训练参数量。冻结的 codec 及其权重单独计数和记录。

缓存身份至少包括：

`数据版本 + 样本/帧选择 + resize/crop + codec/权重 + 因果编码模式 + latent normalization`

不同 encoder、空间分辨率或编码策略的 latent 缓存不可混用。归一化统计只从训练 split 估计并冻结；推理时不按当前 rollout 动态重新估计。

### 5.3 编码因果性与公平比较

必须检验历史 latent 是否依赖未来 RGB。即使 DiT mask 正确，非因果编码器仍可能造成训练信息泄漏。应明确采用因果编码、前缀编码或其他经过验证的边界策略，并将策略写入缓存身份。SANA 的 causal VAE 是可参考实现，不代表任意 LTX 权重和调用方式天然满足同一因果协议。[SANA causal VAE][sana-vae]

统一接口保证组件可替换，不自动保证实验公平。每个 representation variant 需要记录：

- token 数、latent 通道数、时空压缩与实际时间覆盖。
- DiT 和 adapter 参数量、训练数据量、训练步数/计算量。
- reconstruction fidelity、latent norm/variance。
- encoder/decoder 权重、编码因果策略和解码方式。

System-level comparison 与 capacity-controlled comparison 分开报告。不同 latent 通常需要配套 decoder；第一版不把整套 autoencoder 的差异解释为纯 encoder 效应。`diagnostics.py` 提供 held-out 重建与统计结果作为控制信息。

## 6. 相机与动作控制

### 6.1 相机条件：视频版 PRoPE

相机条件采用 Matrix-Game 3.5 的 tiled PRoPE 思路：将相机投影变换与时空 RoPE 组合，在每个 DiT block 的视频 self-attention 中使用。论文中的 tiled PRoPE 与 README 中的 Warped PRoPE 命名需在来源记录中注明。[论文][matrix-paper] [实现][matrix-prope]

给定 camera-to-world 外参 `T_c2w` 和内参 `K`，构造 lifted 投影矩阵：

`P = lift(K) · inverse(T_c2w)`。

`prope.py` 负责对 Q/K/V 和 attention 输出进行配套变换。它不是普通 camera MLP，也不能简化成只对 Q/K 任意乘一个矩阵。`attention.py` 组合这些变换、时空 RoPE 和可见性 mask。

第一版替换原先设想的 camera MLP/Plücker 注入路径，不同时叠加两种相机机制。Stage A/B 使用相同的相机表示与 PRoPE 规则。

PRoPE 描述可见 token 之间的相机关系；causal mask 控制可见范围。目标 chunk 使用该 chunk 的 camera 指令，不提前读取未来 chunk 控制来改变历史状态。

### 6.2 几何和 cache 约定

- `core/camera.py` 统一保存 c2w 语义，并在明确位置转换成 w2c。
- 内参随图像 resize/crop 更新；若 PRoPE 需要进一步归一化，只在适配层执行一次。不能将 SANA reader 的 latent-grid 内参误当成 RGB 像素内参。
- 第一版采用固定 episode 参考原点与固定单位尺度，记录到配置。
- 不直接移植上游按窗口重新中心化和非线性平移压缩后继续复用旧 KV。改变相机变换规则时，必须重新验证 cache 等价性。
- 相机矩阵运算的数值精度、有效尺度和异常值处理需要明确；不能通过未经记录的逐样本缩放改变控制含义。
- 当前 Matrix-Game 实现有四个 sub-frame camera 和特定 head layout 的假设。我们的 codec/patch/head 不一定相同，必须通过 `VideoLayout` 适配，不能硬编码四帧布局。

### 6.3 动作与数据的关系

第一版动作接口是相机导航：

`键盘/鼠标 → camera_controller → 相机轨迹 → PRoPE → DiT`。

训练使用数据提供的相机轨迹；benchmark 直接使用官方轨迹；交互 demo 可以由 controller 生成轨迹。这与 SANA 的 action-string/camera 接口相符。[SANA 文档][sana-docs] [controller][sana-controller]

模型不直接接收字符 W/A/S/D，因此不要求训练数据具有原始按键日志。但该接口也不意味着模型学会了按键动力学或碰撞规则：按键如何转换成目标位姿由 controller 决定。

## 7. DiT 与训练设计

### 7.1 单一模型实现

以 Wan 的 joint spatiotemporal DiT 为代码基础，抽取 block、MLP、norm、timestep modulation、patchify 和时空位置编码。按配置缩放参数量，删除自动加载 pretrained DiT 的路径。[Wan 模型][wan-model]

| 文件 | 职责与交互 |
|---|---|
| `dit.py` | 调用 latent I/O、条件处理与 block，输出 velocity；可按请求返回中间特征 |
| `blocks.py` | 组成标准 block，定义稳定、可命名的特征观测位置 |
| `attention.py` | joint attention、mask 与 KV 接口；不实现实验 loss |
| `position.py` | 全局时空位置；后续 chunk 不从时间零重新编号 |
| `prope.py` | 相机几何变换；与 attention kernel 解耦 |
| `conditioning.py` | timestep 与可选文本条件；不重新引入隐式 camera 分支 |
| `latent_io.py` | 将不同 latent shape 转成相同模型接口，并还原预测 shape |

Bidirectional 与 causal 使用同一份权重结构，避免维护两个逐渐分叉的模型类。小尺寸 dense attention 可作为正确性参考；正式规模使用兼容的高效 kernel。普通 token 级 `is_causal=True` 不等价于 chunk 内双向的 mask。

### 7.2 Native Flow Matching 约定

`flow.py` 是训练与采样共用的数学定义。建议固定：

- `t=0` 为干净 latent，`t=1` 为噪声。
- `z_t = (1-t) z_clean + t epsilon`。
- velocity target 为 `epsilon - z_clean`。
- 推理从 `t=1` 积分至 `t=0`。

时间采样和 weighting 是显式配置。第一版不采用逐帧独立噪声 schedule 或上游 chunk timestep mixture。为简单、可解释，单个训练样本的带噪 target 部分使用共同 FM 时间；干净条件保持干净并标记相应状态。

借用 scheduler 或 loss 代码时必须转换到以上约定，而不是同时保留两套符号、时间端点或缩放定义。

### 7.3 Stage A：bidirectional 条件视频 FM

`training_batch.py` 构造干净初始条件、带噪视频目标、相机/可选文本条件和 loss mask。视频目标在该阶段允许 bidirectional attention。

已知条件帧不作为普通预测目标计算 loss；在对应生成路径中也不能被 solver 任意更新。条件帧的输入方式、噪声状态与 loss 区域由同一个 batch builder 明确定义。

Stage A 从随机 DiT 权重开始，正式样本时间范围保持一分钟。checkpoint 保存模型结构、codec、相机表示和训练配置，使 Stage B 可以可靠继承。

### 7.4 Stage B：teacher-forced chunk-causal FM

训练条件为干净 GT 历史，推理条件为已生成历史。该差异是本项目要研究的 compounding error 来源之一，第一版不通过 self-rollout training 预先消除。

逻辑输入由 clean history 和 noisy target 组成。若并行训练多个 chunk，可以参考 Causal Forcing 的 clean/noisy 双路 mask，但必须满足：

1. Noisy target chunk 只看到此前的 clean chunks 和本 chunk 的 noisy tokens。
2. Noisy target 不能看到本 chunk 的 clean GT 或未来 clean GT。
3. Clean 历史分支只在自己的合法历史范围内计算，不能通过中间层携带未来信息。
4. Chunk 内允许双向；chunk 间因果。
5. Loss 只作用于规定的 noisy target 有效区域。

正确性参考实现可以逐 target chunk 重算前缀；生产实现可用经过对照验证的并行 mask。是否使用双路并行是执行选择，不能改变条件分布或有效训练范围。双路实现的额外 token/activation 开销必须计入 profiling。

参考文件是 Causal Forcing 的 `model/diffusion.py` 和 `wan/modules/causal_model.py`。不能将其名称含有 teacher forcing 的少步模拟 pipeline 整套当作普通 FM trainer。[loss 参考][cf-loss] [mask 参考][cf-model]

### 7.5 Lightning、regularizer 与 checkpoint

`algorithm.py` 负责一个训练 step：构造 batch、调用 DiT、计算 FM 与可选 regularization loss、记录指标。`runtime/trainer.py` 管理 precision、分布式、累积和 Trainer 配置。

Regularizer 接收模型显式返回、保留梯度的特征；默认关闭。不要在 attention 内写实验 loss，也不要复用已经 detach 的观察性 probe 数据。

Checkpoint 必须区分：

| 选项 | 语义 |
|---|---|
| `init_from` | 只初始化模型权重，开始新阶段；重置 optimizer、scheduler、step，并按明确规则重新初始化 EMA |
| `resume_from` | 恢复同一次训练，包括 optimizer、scheduler、step、EMA 及可恢复的随机数/数据进度状态 |

两者互斥。Stage A→B 使用 `init_from`，中断续训使用 `resume_from`。记录 Stage B 的父 checkpoint，明确加载原始还是 EMA 权重。不能将模板现有的 checkpoint 恢复入口直接等同于跨阶段权重初始化。[模板 Experiment][template-experiment]

## 8. 一分钟生成与历史状态

### 8.1 唯一 rollout 路径

`rollout.py` 执行：

`加载初始条件 → 为当前 chunk 初始化噪声 → solver 去噪 → 提交干净历史 → 下一个 chunk → 解码/保存`。

- `solver.py` 使用 `flow.py` 的相同约定，采样步数、CFG 等显式记录。
- `history.py` 保存历史 latent、位置、相机和 KV，并区分临时与持久状态。
- `writer.py` 输出视频和元数据，处理初始帧、末尾 padding 与精确帧数。
- 生成时可以选择性输出 latent，以便 replay；不要求所有正式评估都保存全部内部状态。

首版即支持完整一分钟。生成质量差仍可评分和定位问题；不能用“短片能跑”替代完整协议，也不能把“完成一分钟输出”解释为“一分钟质量已经合格”。

### 8.2 KV cache 的正确性

必须保留 reference 与 cached 两种执行模式：前者重算合法历史用于验证，后者用于正式生成。

每个去噪步骤的 target KV 是临时状态。当前 chunk 完成后，应按干净历史语义执行必要的前向计算，再提交可供后续 chunk 使用的 KV。不能把最后一次带噪前向的缓存直接当作最终干净历史。[Causal Forcing 多步推理参考][cf-inference]

缓存身份与生命周期需要覆盖位置、相机变换、条件分支和 checkpoint。若使用 CFG，条件/无条件分支的状态必须正确隔离；更换 episode、权重或条件时明确清空。

本项目基线在一分钟范围内不执行历史 eviction。后续若研究有限窗口或 memory，应作为明确实验变量，不以隐蔽的性能优化方式加入。

## 9. 官方 evaluation 与子集协议

### 9.1 三个可独立执行的任务

1. **Select**：`selection.py` 产生明确的 scene/split/seed 清单。
2. **Generate**：对该清单运行统一 rollout，输出完整轨迹视频。
3. **Score**：`runner.py` 检查产物后，通过 `sana_adapter.py` 调用固定版本官方指标。

修改指标或评估环境时，不重新生成视频。只有 checkpoint、条件、seed、codec 或采样设置等生成身份改变，才创建新的生成产物。

### 9.2 子集选择与完整性

支持指定 scene IDs、simple/hard split、固定 seed 抽样和按类别分层抽样。实际 ID 清单必须保存，可被不同 representation 和训练 seed 复用。

子集减少场景数，不截短官方轨迹或重新定义 revisit pairs。全量与子集报告分开标记。

评分前必须检查：

- 选定视频是否全部存在且可读取。
- 帧数、FPS、预处理和 revisit pair 索引是否有效。
- 是否混入其他 checkpoint、seed 或场景的旧文件。
- 指标成功覆盖了多少场景/配对，失败项是什么。

正式完整报告不得通过静默跳过失败视频产生。调试时可输出部分结果，但必须明确标记 incomplete 和覆盖率。

### 9.3 官方脚本与本地汇总的边界

`third_party/sana_wm_metrics/` 固定官方 `eval_unified.py`、camera evaluation 和必要辅助文件。`sana_adapter.py` 负责目录、文件名、metadata 与调用参数，不重新定义官方 metric。[官方指标目录][sana-metrics]

第一版支持官方 revisit、camera、VBench、temporal 指标入口，可选择只运行某一类。Revisit 使用官方配对规则比较相应生成帧；不能擅自替换为对完整 GT 视频的逐帧预测误差。

`aggregate.py` 在官方逐场景结果之上汇总 split、seed 和不确定性。与原始官方聚合结果保持可核对；新增统计不能覆盖原始指标或改变其量纲。

官方 benchmark 条件不应被假设为有完整配对 GT 视频。Clean-vs-rollout probing 另用 held-out 真实视频。

低分辨率输出可用于内部消融，但必须报告生成分辨率和官方评估预处理；不能将其与其他模型的 720p 结果宣称为完全等设置比较。

## 10. Representation probing 与受控分析

### 10.1 采集协议

`blocks.py` 提供命名稳定的观测点，`capture.py` 根据配置选择层、时间、token 和条件分支。采集不得改变模型可见性、随机数消费或输出。

每条记录至少标注：

- sample ID、训练/生成 seed、checkpoint 与配置身份。
- Layer 和观测点，例如 attention 后、MLP 后、pre/post norm。
- Rollout 时间与 FM 时间；不能只记录其中一个。
- Context/target token、物理位置、条件/无条件分支。
- 前向用途：去噪、提交干净历史、训练或 controlled replay。

避免将 activation checkpointing 的重算当作新的独立样本。默认观察 residual hidden states；若分析 Q/K/V，应额外区分 PRoPE 变换前后，防止将坐标变换导致的 norm 差异误认成表示退化。

### 10.2 第一版指标与存储

优先实现：

1. Feature norm 与基本分布统计。
2. Singular-value spectrum / effective rank。
3. Cosine similarity 或 CKA。
4. GT-history 与 generated-history 的 feature alignment / drift。

每个指标明确是否中心化、按什么轴组成样本矩阵、采样多少 token。Rank 等指标受矩阵尺寸和采样数影响，不同设置必须可比较。Intrinsic dimension 留作后续扩展。

`metrics.py` 只处理特征；`store.py` 默认保存统计摘要，仅对选定样本保留原始特征。避免默认保存一分钟全部层、全部去噪步的激活。

### 10.3 Controlled replay 与 regularization

`replay.py` 在 held-out 视频上构造对照：尽量固定 target、噪声、FM 时间、camera、文本和 token 选择，只改变 GT history 与 generated history。

原始 rollout 的观察性比较与受控 replay 分开标记。两者回答的问题不同，相关性也不自动等于因果机制。

后续 regularization 使用保留梯度的明确特征接口；观察性 capture 可以 detach。新增 loss 必须与关闭该 loss 的同预算基线比较，保持训练、生成和评分路径一致。

## 11. 配置、运行记录与环境

### 11.1 配置原则

Hydra 负责组合配置，Experiment 负责执行任务。新增 encoder、regularizer 或 probe 主要通过配置表达，不复制一份 trainer/rollout 修改。

`preflight.py` 在运行前检查相互约束：codec stride 与 chunk、head dimension 与 PRoPE layout、camera 与帧索引、checkpoint 与模型结构、正式长度与输出设置等。不兼容时直接报错，不静默使用默认值。

### 11.2 每次运行的产物

```text
outputs/<run_id>/
├── resolved_config.yaml
├── provenance.json              # Git、环境、数据/codec、父 checkpoint、seeds
├── checkpoints/
├── selections/                  # 实际评估 ID 清单
├── rollouts/                    # 按 checkpoint、split、seed 等隔离
├── metrics/                     # 原始逐样本结果 + 聚合
└── probes/                      # 采集索引、摘要与选定原始特征
```

记录原始/EMA 权重选择、初始化与恢复方式、数据清单 hash、缓存版本、生成参数和官方评估版本。保存实际解析后的配置，不能只依赖可变化的默认 YAML。

W&B 是可选展示层，本地记录独立可用。

### 11.3 执行环境与预算

- 训练、数据准备、完整官方评估可以使用不同依赖组或环境，通过标准文件交接。
- Pi3X/VBench 等重型评估依赖不必进入最小训练环境。
- Runtime 配置管理 DDP/FSDP、precision、activation checkpointing 和 attention backend。
- 完整长度 profiling 至少报告：实际参数量、latent/token 数、训练峰值显存、KV 大小、单步训练时间与 rollout 时间。
- 300M 参数量不能单独决定一分钟 joint attention 的计算成本；高效 kernel 也不会消除其所有计算增长。
- 数据清单、训练预算、训练 seed 和生成 seed 分别管理。配对实验尽量使用相同评估场景和生成 seeds。

## 12. 上游来源与防漂移规则

### 12.1 锁定的参考版本

这些是本次设计核对的版本，不是要求永远不升级。升级必须成为明确的实现变更，并重新运行受影响的正确性检查。

| 来源 | Commit |
|---|---|
| `legallchaperone/research-template` | `e4a3f528d4aad7872adf6ad9cd94f80c06d91e48` |
| `Wan-Video/Wan2.1` | `9737cba9c1c3c4d04b33fcad41c111989865d315` |
| `thu-ml/Causal-Forcing` | `31a43313af7c805af10e4e2bcbad1fa036e0ced9` |
| `NVlabs/Sana` | `f9178744c096dcf2a2ea773da183e341bcbeb044` |
| `Riemann-Dynamics/Matrix-Game-3.5` | `fbf7def0693ae14f745ba35bf2a26215d4ef991d` |
| `liruilong940607/prope` | `4c11297761225d25258e5ec21c61e9b19ab61e38` |

### 12.2 目标文件到上游实现的映射

| 本仓库目标 | 参考来源 | 应复用的内容 | 必须适配或排除的内容 |
|---|---|---|---|
| `main.py`、`experiments/`、训练基类 | [research-template][template] | Hydra 入口、Experiment 分发、Lightning 骨架 | 扩展独立任务、local logging、collate/strategy 配置及 init/resume 语义 |
| `models/dit.py`、`blocks.py`、`position.py`、`latent_io.py` | [Wan model.py][wan-model] | 标准 block、位置编码、modulation、patch 操作 | 配置化规模，随机初始化；不自动加载 Wan DiT 权重 |
| `models/prope.py` | [Matrix-Game prope_attention.py][matrix-prope]；[原始 PRoPE][prope-original] 用于核对几何 | Q/K/V/output 的配套变换、视频 PRoPE 思路 | 适配 sub-frame/head 布局、mask 与 cache；不混入 memory 或额外位置实验分支 |
| `core/camera.py` | [SANA cam_utils.py][sana-cam-utils] 与 PRoPE 几何代码 | 外参/内参等基础几何运算 | 统一 c2w/w2c、RGB/latent 内参和单位；不继续叠加 Plücker 条件分支 |
| `controls/camera_controller.py` | [SANA camera_control.py][sana-controller] | 按键到速度、位姿积分等 | 固定按键语义、FPS 和速度单位；不当成学习到的动作动力学 |
| `training_batch.py`、`attention.py` 的 causal 部分 | [Causal Forcing causal_model.py][cf-model] | clean/noisy 双路 mask、chunk 因果逻辑 | 删除固定 token/层数/长度；接入本项目 PRoPE；检查无信息泄漏 |
| `flow.py` 与 `algorithm.py` 的监督训练组织 | [Causal Forcing diffusion.py][cf-loss] | supervised flow target 与 clean history 组织 | 统一时间/velocity 约定；不复制自动 pretrained wrapper、context noise 或独立 trainer |
| `inference/rollout.py`、`history.py` | [Causal Forcing causal_diffusion_inference.py][cf-inference] | 多步 chunk AR、干净历史提交、分支 cache | 适配完整一分钟、动态 shape、PRoPE 与统一 solver |
| `datasets/sana_wm/reader.py` | [SANA sana_wm_zip_latent_data.py][sana-reader] | ZIP、latent、camera sidecar 读取 | 严格失败记录，独立 manifest；去掉隐式 fallback 和 codec 专属大小启发式 |
| `representations/video_vae.py` | 所选 codec 官方实现；[SANA causal_vae.py][sana-vae] 为已核对参考 | encode/decode 与必要 streaming state | 明确权重、因果模式、边界、归一化；不假设不同 checkpoint 等价 |
| `third_party/sana_wm_metrics/` | [SANA 官方 metrics][sana-metrics] | 官方评分与必要辅助代码 | 尽量原样保留；格式适配放外层，协议变更另起版本 |
| `evaluation/aggregate.py` | [官方 aggregate_results.py][sana-aggregate] | 官方量纲与聚合口径核对 | 新增 seed/覆盖率统计不覆盖原始结果 |
| `core/types.py`、`video_layout.py`、selection、probe/replay、provenance | 本项目 | 研究接口与控制变量 | 自己实现，保持小而明确，不引入通用框架依赖 |

### 12.3 复用规则

1. 每个移植文件在 `UPSTREAMS.yaml` 中记录目标路径、上游路径、commit、许可、修改目的和对应验证。
2. 已复制的上游文件保留版权头和许可证。模板要求的作者署名保留在 README 与 LICENSE 中。[模板许可][template-license]
3. “原样 vendoring”与“改写适配”明确区分；官方 metrics 优先原样，模型机制允许小范围适配。
4. 不整体复制 SANA/Causal Forcing 的 trainer 与分布式生命周期。仓库只有一套主训练执行框架。
5. 不根据文件名推断语义：尤其不能把 Causal Forcing 的 `pipeline/teacher_forcing_training.py` 少步模拟流程当作本项目普通 TF FM。
6. 不将上游固定的层数、token 数、latent 长度、VAE stride 或训练默认值带入本项目。
7. 新增 normalization、history eviction、context corruption、loss weighting 或 sampler trick 都是实验变更，不能隐藏在“代码复用”中。

## 13. 最低正确性验证

只为有实际风险的协议和机制写测试；不为简单配置搬运堆积镜像测试。

| 检查 | 需要证明的性质 |
|---|---|
| 时间布局 | 首帧、压缩、patch、chunk、padding 与解码帧数一致 |
| 相机预处理 | resize/crop 后内参正确，c2w/w2c 约定和 PRoPE 布局一致 |
| Codec 因果性 | 改变未来 RGB 不影响声明为历史可用的 latent |
| Causal mask | 改变未来/本 chunk 的 clean GT 不影响不应可见的预测 |
| FM 约定 | 插值端点、velocity 符号和 solver 方向一致 |
| Cache 等价性 | 多个 chunk 下 cached 与 reference 输出在约定容差内一致 |
| Probe 不干扰 | 开关观察性 probe 不改变随机数路径和生成输出 |
| Checkpoint | 跨阶段初始化不恢复旧 optimizer；同 run 恢复保留预期状态 |
| Evaluation | 精确评分选定清单，缺失或混入产物能被发现 |
| 数据隔离 | split 与缓存身份检查能发现重叠或不兼容输入 |

容差应按 precision/backend 在测试配置中声明。Resume 不默认承诺跨硬件、跨并行规模的逐位一致性，但必须记录实际恢复范围。

## 14. 实现 Milestones

每个 milestone 交付可检查的产物，并在通过验收后推进。Debug 小模型用于验证机制；正式一分钟生成能力不推迟到扩展研究阶段。

### M0：建立模板骨架与实验契约

**工作：**从锁定 research-template 建仓，保留署名，加入设计文档、配置组、来源记录与 local/offline 运行支持。定义 `core/types.py` 和配置检查边界。

**交付：**可解析的基线配置、清晰的任务入口、`UPSTREAMS.yaml` 初版。

**验收：**能解析完整配置并保存 resolved config/provenance；没有隐式 pretrained DiT 加载；无需 W&B 凭据即可执行不训练的检查任务。

### M1：完整数据样本、时间布局与 codec

**依赖：**M0。

**工作：**接通 SANA 视频/camera/metadata、固定小数据 manifest 和 split；实现 `VideoLayout`、VAE 适配、缓存身份和重建诊断。

**交付：**完整 961 帧样本的读取/编码/解码结果、相机对齐报告、数据失败清单。

**验收：**首帧与末尾映射正确；codec 因果策略通过验证；缺失 camera 会明确失败；不同预处理/codec 不会误命中同一缓存。

### M2：Joint DiT + PRoPE + FM 数学路径

**依赖：**M1。

**工作：**移植 Wan 基本 block、视频 PRoPE 和 FM 定义；用小尺寸模型验证 shape、数值、条件输入和梯度。

**交付：**唯一 DiT 实现、PRoPE 适配记录、bidirectional FM 可运行路径。

**验收：**没有固定上游 token/层数假设；相机与位置变换测试通过；FM 端点和采样方向一致；probe 观测位置具有稳定名称。

### M3：正式配置 profiling 与 Stage A 基线

**依赖：**M2。

**工作：**在完整时间长度上测量训练成本，确定约 300M 的网络与空间设置；接通 optimizer、EMA、checkpoint 和 held-out validation，训练 Stage A。

**交付：**冻结的首版正式配置、profiling 报告、Stage A checkpoint 和条件生成样例。

**验收：**正式长度可执行；checkpoint 可恢复；实际参数/显存/时间可追溯；相机条件和短期生成质量经过检查。若质量不足，明确定位和修正，不将其掩盖为长期一致性结论。

### M4：Stage B teacher forcing 与历史状态正确性

**依赖：**M2；正式训练权重依赖 M3。

**工作：**实现 clean/noisy mask、Stage A→B 权重初始化、reference/cached rollout 和干净历史提交。

**交付：**causal TF 训练路径、Stage B checkpoint、cache 对照报告。

**验收：**目标看不到自身 clean GT，历史无未来泄漏；跨多个 chunk 的 reference/cache 对照通过；PRoPE 坐标和 cache 一致；未引入 DF timestep mixture 或 self-rollout loss。

### M5：完整一分钟生成与官方子集评分闭环

**依赖：**M4；评估适配与清单逻辑可以在此前独立准备。

**工作：**固定少量官方场景和 seeds，生成完整一分钟视频；接入官方 revisit/camera 等指标、完整性检查和独立评分任务。

**交付：**完整视频、官方格式产物、逐场景指标、覆盖率和聚合报告。

**验收：**所有选定场景输出满足帧数/FPS/轨迹要求；评分可在不重新生成的情况下重跑；缺失结果不能静默变成完整报告。得分低是待分析的结果，不是无法执行 benchmark 的理由。

### M6：Probing 与受控 replay

**依赖：**M4–M5；基础 capture 可在 M2 后准备。

**工作：**实现层/时间采样、摘要存储、首批指标，在 held-out 视频上实现 GT-history 与 generated-history replay。

**交付：**一组可复现的表示轨迹、相应 rollout/revisit 结果，以及清楚的采集 metadata。

**验收：**probe 不改变输出；FM 时间与 rollout 时间分开记录；对照 replay 固定噪声和目标；特征存储量在预算内。

### M7：受控 representation / regularization 实验

**依赖：**M5–M6。

**工作：**加入第二种 representation 或一个明确的 regularizer；先一次改变一类因素，再进行多 seed 实验。扩展固定数据规模与官方评估覆盖。

**交付：**基线与变体的配置差异、重建/容量控制、长期指标及 probing 对照。

**验收：**System-level 与 capacity-controlled 结论分开；实际训练/采样预算可比较；无不同 rollout 或评估实现造成的隐式差异；完整官方评估结果与子集结论能够对应。

### M8：可选 robustness 与外部架构验证

**依赖：**已经获得稳定、可重复的主实验结论。

**工作：**按研究问题选择更大/不同 backbone、rollout-aware post-training、distillation 或显式 memory 干预；每项建立独立配置与对照。

**交付：**方法是否跨架构、跨规模或在后训练之后保持效果的证据。

**验收：**不改变或覆盖主实验基线；清楚区分 representation 效应、训练分布变化和额外历史信息带来的改善。

[template]: https://github.com/legallchaperone/research-template/tree/e4a3f528d4aad7872adf6ad9cd94f80c06d91e48
[template-experiment]: https://github.com/legallchaperone/research-template/blob/e4a3f528d4aad7872adf6ad9cd94f80c06d91e48/experiments/exp_base.py
[template-license]: https://github.com/legallchaperone/research-template/blob/e4a3f528d4aad7872adf6ad9cd94f80c06d91e48/LICENSE
[wan-model]: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py
[cf-model]: https://github.com/thu-ml/Causal-Forcing/blob/31a43313af7c805af10e4e2bcbad1fa036e0ced9/wan/modules/causal_model.py
[cf-loss]: https://github.com/thu-ml/Causal-Forcing/blob/31a43313af7c805af10e4e2bcbad1fa036e0ced9/model/diffusion.py
[cf-inference]: https://github.com/thu-ml/Causal-Forcing/blob/31a43313af7c805af10e4e2bcbad1fa036e0ced9/pipeline/causal_diffusion_inference.py
[sana-docs]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/docs/sana_wm.md
[sana-causal-config]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/configs/sana_wm/stage1/sana_wm_stage1_sekai_chunk_causal_cp2_fsdp2.yaml
[sana-reader]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/diffusion/data/datasets/video/sana_wm_zip_latent_data.py
[sana-cam-utils]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/diffusion/utils/cam_utils.py
[sana-controller]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/inference_video_scripts/wm/camera_control.py
[sana-vae]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/diffusion/model/ltx2/causal_vae.py
[sana-metrics]: https://github.com/NVlabs/Sana/tree/f9178744c096dcf2a2ea773da183e341bcbeb044/tools/metrics/sana_wm
[sana-aggregate]: https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/tools/metrics/sana_wm/aggregate_results.py
[matrix-paper]: https://arxiv.org/html/2608.29910v1#S2.SS1.SSS1
[matrix-prope]: https://github.com/Riemann-Dynamics/Matrix-Game-3.5/blob/fbf7def0693ae14f745ba35bf2a26215d4ef991d/diffsynth/models/prope_attention.py
[prope-original]: https://github.com/liruilong940607/prope/blob/4c11297761225d25258e5ec21c61e9b19ab61e38/prope/torch.py
