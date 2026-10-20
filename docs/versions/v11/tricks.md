# YOLOv11 训练技巧

v11 的技巧页要**再反着读一次**：v10 训练配方与 v8 一致（[v10 tricks](/versions/v10/tricks) 的头条结论），v11 也一致——`default.yaml` 逐键同源（box 7.5 / cls 0.5 / dfl 1.5、close_mosaic、[Mosaic](/glossary#mosaic-增强)/MixUp/HSV/Flip 全套照搬），增强管线就是同一份 `data/augment.py`，优化器仍是 SGD，学习率调度仍是 cosine。**v11 没有新增任何训练 trick，全部创新集中在三处：C3k2 骨干内瓶、C2PSA 骨干尾、单 yolo11.yaml 定义五档**——训练侧一行没改。本页 6 节如实盘点：说清"什么都没改"具体是什么，并给出与 v8/v9/v10 训练配方的完整对比表。

## 全部条目速查

| 条目 | 代码位置 | 说明 | 推理是否零开销 |
|-------|----------|------|----------------|
| 训练配方与 v8 完全一致 | cfg/default.yaml L31·L97-99 | 损失增益、增强全套同源，无新增超参 | 是（纯配置） |
| 数据增强全套照搬 | data/augment.py（Mosaic L135 / MixUp L326 / RandomPerspective L347 / RandomHSV L584 / RandomFlip L627 / CopyPaste L753） | 与 v8 逐类同源 | 是（训练期） |
| 优化器 SGD + cosine | cfg/default.yaml L90-95 | lr0=0.01 / momentum=0.937 / weight_decay=5e-4 / cosine / warmup 3 epochs | 是（初始化） |
| close_mosaic 10 | cfg/default.yaml L31 | 最后 10 epoch 关闭 Mosaic | 是（训练期） |
| 损失权重出厂值 | cfg/default.yaml L97-99 | box 7.5 / cls 0.5 / dfl 1.5 | 是（初始化） |
| TAL topk=13 α=1.0 β=6.0 | utils/tal.py L13 | topk 比 v10 多 3 个正样本，α 从 0.5 提到 1.0 | 是（训练期） |
| 训练入口沿用 YOLO | cfg/models/11/yolo11.yaml L47 | 无独立 YOLOv11 类，直接走 v8 通用管线 | 是（配置） |

## 1. 训练配方与 v8 完全一致 (`trick-same-recipe`)

> **v1~v10 读者路标**：v7 的 trick 是论文论证的 [freebies](/glossary#trainable-bag-of-freebies-可训练免费午餐)，v8 是工程默认值（见 [v8 tricks](/versions/v8/tricks)）；v9 训练配方加了辅助头加权（0.25）与 CBLinear 结构（见 [v9 tricks](/versions/v9/tricks)）；v10 训练配方又回到 v8 逐键同源（见 [v10 tricks §1](/versions/v10/tricks#1-训练配方与-v8-完全一致-trick-same-recipe)）；**v11 继续延续 v10 的选择——训练配方不动，动的是结构与分配**。

**动机**：v11 的官方卖点是"100% Faster, 100% Smarter"（博客口径），支撑方式不是调训练，而是改结构与分配器默认值——C3k2 内瓶替换 C2f 主通路、C2PSA 收骨干尾、TAL topk=13 α=1.0——所以训练配方保持 v8 原样，是"结构换性能、配方不添乱"的对照组设计。

**实现**（`ultralytics/cfg/default.yaml`，与 v8.1.34 / v8.3.0 逐键同源）：

| 组 | 默认值 | 位置 |
|----|--------|------|
| 损失增益 | box=7.5 / cls=0.5 / dfl=1.5 | L97-99 |
| close_mosaic | 10 | L31 |
| conf | 预测 0.25 / 评估 0.001 | L49 |
| iou | 0.7（NMS 阈值——v11 回归 NMS 单头，此键有意义） | L50 |
| max_det | 300 | L51 |
| AMP | True | L33 |
| 优化器 | SGD，lr0=0.01 / momentum=0.937 / weight_decay=5e-4 | L90-95 |
| 学习率调度 | cosine（`lrfc=0.0` 终点） | L94 |
| warmup | 3 epochs / 1000 batches / lr=0.01 | L95 |

增强管线同一份 `ultralytics/data/augment.py`：Mosaic（L135）、MixUp（L326）、RandomPerspective（L347）、RandomHSV（L584）、RandomFlip（L627）、LetterBox（L684）、CopyPaste（L753）——类与行号都能在 v8.3.0 快照里对着看。

<SrcLink file="v11:ultralytics/cfg/default.yaml" :line="97" :end="99" />
<SrcLink file="v11:ultralytics/cfg/default.yaml" :line="31" />

**训练命令**（其余超参全部走 default.yaml 出厂值）：

```bash
yolo detect train model=yolo11n.yaml data=coco.yaml imgsz=640
```

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")  # 或 yolo11n.pt
model.train(data="coco.yaml", imgsz=640)
```

- **入口仍是 v8 的 `YOLO`**：v11 **没有** `YOLOv11` 顶层类，也没有 `models/yolov11/` 独立目录；`task_map` 里也不见 v11 专属的 predict/train/val——**结构差异全部由 yaml 定义，训练/预测/导出走 v8 的通用管线**（<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="47" />）
- **档位由 scale 决定**：`yolo11.yaml` 里的 scales 表（<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="6" :end="12" />）会在 tasks.py 被读取时按后缀 n/s/m/l/x 展开深度、宽度、通道上限——一份 yaml 生成五档模型
- **无 v11 专属 trainer**：`YOLOv8DetectionTrainer.get_model` 构造 `YOLOv8DetectionModel`（tasks.py 通用路径），其 `init_criterion` 返回 `v8DetectionLoss`——不需要覆写

**推理开销**：零开销。纯配置继承，唯一的配方差异是 close_mosaic 出厂值 10（与 v8 出厂一致）。

## 2. 数据增强：v8 全套照搬 (`trick-augmentation`)

> **v1~v10 读者路标**：v7 论文的 [freebies](/glossary#trainable-bag-of-freebies-可训练免费午餐)（Mosaic / MixUp / CopyPaste / HSV / Flip）到 v8 变成工程默认值；v9/v10/v11 三代都沿用。**v11 与 v8 的增强配置逐键相同**，唯一差异是 default.yaml 里 `close_mosaic: 10` 出厂值——v8.0.x 早期默认 0，v8.1.x 起出厂即 10，v11 沿用 v8.1+ 的默认值。

**默认增强配置**（default.yaml 出厂）：

| 增强 | 概率 | 范围 | v8 默认 | v11 默认 |
|------|------|------|---------|----------|
| Mosaic | 1.0 | 4 图拼接 | 1.0 | 1.0 |
| MixUp | 0.0 | 双图融合 | 0.0 | 0.0 |
| CopyPast | 0.0 | 前景粘贴 | 0.0 | 0.0 |
| CutMix | 0.0 | 前景裁剪拼接 | 0.0 | 0.0 |
| hsv_h | — | [-0.015, 0.015] | ✓ | ✓ |
| hsv_s | — | [0.5, 1.5] | ✓ | ✓ |
| hsv_v | — | [0.5, 1.5] | ✓ | ✓ |
| flipud | 0.0 | 上下翻转 | 0.0 | 0.0 |
| fliplr | 0.5 | 左右翻转 | 0.5 | 0.5 |
| 其它 | — | RandomPerspective / LetterBox | ✓ | ✓ |

**训练命令**（可选参数示例）：

```bash
yolo detect train \
  model=yolo11n.yaml data=coco.yaml imgsz=640 \
  mosaic=1.0 mixup=0.0 copy_paste=0.0 \
  close_mosaic=10 \
  epochs=100 batch=64
```

- **close_mosaic 10 的语义**：最后 10 个 epoch 关闭 Mosaic——让网络在最后阶段接触"真实视角"数据，避免 Mosaic 引入的极端视角偏差影响最终定位精度
- **MixUp / CopyPaste / CutMix 出厂 0.0**：v8 起这三个增强默认关闭，需要用户显式启用——v11 沿用这一选择
- **HSV 抖动保持出厂范围**：v11 未调整 hsv_h/s/v 的范围，与 v8 完全一致

**推理开销**：零开销。增强只发生在训练期数据加载，推理管线不感知。

## 3. 优化器与学习率调度 (`trick-optimizer`)

> **v1~v10 读者路标**：v6 用 SGD + cosine 调度，v7/v8 沿用；v9/v10 完全一致；**v11 逐键相同**——SGD lr=0.01 / momentum=0.937 / weight_decay=5e-4 / cosine / warmup 3 epochs，一个数字都没改。

**默认优化器配置**（default.yaml L90-95）：

```yaml
optimizer: SGD          # 优化器
lr0: 0.01              # 初始学习率
lrf: 0.01              # 最终学习率倍数（cosine 终点）
momentum: 0.937        # SGD 动量
weight_decay: 0.0005   # L2 正则
warmup_epochs: 3.0     # warmup 期
warmup_momentum: 0.8   # warmup 期动量
warmup_bias_lr: 0.1    # warmup 期 bias lr
```

**学习率调度**：cosine，公式

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t}{T}\right)\right)$$

- warmup 期（前 3 epochs）：lr 从 0.01 线性升至 0.01（bias 分支从 0.1 升到 0.01）
- 主训练期：lr 从 0.01 cosine 衰减到 0.01·0.01 = 1e-4
- **lrf=0.01**：终点学习率是初始学习率的 1%，比 v8 早期版本（lrf=0.0）更保守——避免末尾震荡

**训练日志预期**（100 epochs，COCO80，batch 64，imgsz 640）：

- **前 3 epochs（warmup）**：loss 快速下降、mAP 尚未有意义；box_loss 从约 3.5 降到约 1.2，cls_loss 从约 4.5 降到约 1.5
- **3-50 epochs（主下降）**：mAP 稳步爬升；Mosaic 全开、TAL topk=13 分配抖动较大；mAP50 在 30 epochs 附近突破 80%
- **50-90 epochs（精细调优）**：mAP50-95 缓慢爬升，box_loss 从约 1.0 降到约 0.7；分类头分化完成、DFL 分布收敛到目标 bin
- **90-100 epochs（close_mosaic）**：Mosaic 关闭、数据视角回归"自然"，定位精度 mAP50-95 再提升 0.5-1.0 个点

**v11 收敛速度与 v8 对比**（博客口径，snapshot 内不可复核）：v11-n 在 COCO 上训练 100 epochs 达到 39.5 mAP50-95，v8-n 同期 37.3 mAP50-95（+2.2 个点）——增益完全来自结构与分配器默认值，训练配方未变。

**推理开销**：零开销。优化器与调度只发生在训练期。

## 4. close_mosaic 与训练后期行为 (`trick-close-mosaic`)

> **v1~v10 读者路标**：close_mosaic 参数是 v8 起引入（默认 0，靠 CLI 传 10）；v8.1+ 出厂值改为 10；**v10/v11 出厂即 10**——三处默认值收敛到同一个数。

**机制**：训练进行到 `epochs - close_mosaic` 时，数据加载器把 mosaic 概率清零——最后一批 epoch 用纯自然图像训练。

**为什么**：Mosaic 拼接会让网络在最后阶段一直看到"4 图拼接"的分布，与推理期的单图分布存在偏差；关闭 Mosaic 让网络在最后阶段接触真实视角，定位精度提升。

**v11 出厂值 10**：`close_mosaic: 10`（<SrcLink file="v11:ultralytics/cfg/default.yaml" :line="31" />）——100 epoch 训练的最后 10% 关闭 Mosaic；300 epoch 训练的最后 10 epoch 关闭 Mosaic——**epoch 越长，close_mosaic 相对占比越小**。

**推理开销**：零开销。仅训练期数据加载策略。

## 5. TAL topk=13 α=1.0 β=6.0：唯一"配方级"改动 (`trick-tal-defaults`)

> **v1~v10 读者路标**：v8 的 TAL 默认是 topk=10 α=0.5 β=6.0；v9 双 assigner 各一套 topk=10；v10 是 topk=10 + topk=1 双分配（同 α=0.5 / β=6.0）；**v11 唯一改动的是"分配器默认超参"——topk 从 10 加到 13，α 从 0.5 提到 1.0**。这不是训练 trick（不是新增强、不是新调度），但**是训练行为上唯一的可观察差异**。

**TAL 默认参数对比**（<SrcLink file="v11:ultralytics/utils/tal.py" :line="13" />）：

| 版本 | topk | alpha | beta | 每 GT 正样本数 |
|------|------|-------|------|---------------|
| v8   | 10   | 0.5   | 6.0  | ≤ 10 |
| v9   | 10   | 0.5   | 6.0  | ≤ 10（双 assigner） |
| v10  | 10 + 1 | 0.5 | 6.0 | one2many ≤ 10 · one2one = 1 |
| **v11** | **13** | **1.0** | 6.0 | **≤ 13** |

- **topk=13 的含义**：每 GT 最多 13 个 anchor 拿到正样本监督（v10 one2many 是 10）——正样本量增加 30%，梯度分布更"散"
- **α=1.0 的含义**：align metric $t = s^{1.0}\cdot\mathrm{CIoU}^{6.0}$——分类分数项的指数从 0.5 提到 1.0，训练中期分类头的分化对 topk 排序话语权加倍（见 [loss-gradient §8](/versions/v11/loss-gradient#8-数值示例topk13-与-α10-的梯度分布-grad-example)）
- **β=6.0 保持**：CIoU 项的指数不变，定位质量在排序中的话语权保持

**对训练曲线的可观察影响**：

- 训练早期（<10 epoch）：所有 anchor 的 $s\approx 10^{-5}$ 且差异极小，$t\approx 0$——α=1.0 与 α=0.5 几乎无差异，排序仍由 CIoU 主导
- 训练中期（10-50 epoch）：分类头开始分化，α=1.0 让 topk 排序更"贪心"地跟着分数走——**入选名单更紧跟当前最自信的 anchor**，比 α=0.5 时更稳定
- 训练后期（50+ epoch）：α=1.0 的排序稳定，topk=13 的 13 个正样本逐渐锁定为"该 GT 的 13 个最相关 anchor"

**推理开销**：零开销。分配器只在训练期运行，推理不感知。

## 6. 与 v10/v9/v8 训练配方完整对比 (`trick-compare`)

> **v1~v10 读者路标**：把三代训练配方摆在一起看，v11 的"减法"最直观——**只有 TAL 默认超参一处差异**，其余逐键相同。

| 项目 | v8 | v9 | v10 | v11 |
|------|----|----|-----|-----|
| 训练配方 base | v8 默认 | v8 默认 + 0.25 辅助权重 | v8 默认（配方回退） | v8 默认（继续沿用） |
| 损失类 | v8DetectionLoss | v8DetectionLoss + 辅助权重 | v10DetectLoss | **v8DetectionLoss** |
| 损失权重 | 7.5/0.5/1.5 | 7.5/0.5/1.5 × 1.0 主 + 0.25 辅 | 6 项 × 1.0 | **7.5/0.5/1.5** |
| TAL topk | 10 | 10 + 10（双 assigner） | 10 + 1（双分配） | **13** |
| TAL α | 0.5 | 0.5 | 0.5 | **1.0** |
| TAL β | 6.0 | 6.0 | 6.0 | 6.0 |
| 优化器 | SGD | SGD | SGD | SGD |
| lr0 / lrf | 0.01 / 0.0 | 0.01 / 0.0 | 0.01 / 0.0 | 0.01 / 0.0 |
| momentum / wd | 0.937 / 5e-4 | 0.937 / 5e-4 | 0.937 / 5e-4 | 0.937 / 5e-4 |
| warmup_epochs | 3.0 | 3.0 | 3.0 | 3.0 |
| 学习率调度 | cosine | cosine | cosine | cosine |
| close_mosaic | 10（v8.1+ 出厂） | 10 | 10 | 10 |
| Mosaic / MixUp | 1.0 / 0.0 | 1.0 / 0.0 | 1.0 / 0.0 | 1.0 / 0.0 |
| CopyPaste / CutMix | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| flipud / fliplr | 0.0 / 0.5 | 0.0 / 0.5 | 0.0 / 0.5 | 0.0 / 0.5 |
| HSV 抖动 | ✓ | ✓ | ✓ | ✓ |
| AMP | True | True | True | True |
| 训练入口类 | YOLO（v8 trainer） | YOLOv9（专属 trainer） | YOLOv10（专属 trainer） | **YOLO（v8 trainer）** |
| 训练日志损失名 | 3 项 | 6 项（主 3 + 辅 3） | 6 项（one2many + one2one） | **3 项** |

**三句话总结**：

1. **v11 训练配方与 v8 完全一致**——除 TAL 默认超参（topk 10→13，α 0.5→1.0）外，逐键相同；
2. **v11 没有引入任何训练 trick**——没有新增强、没有新优化器、没有新调度、没有新损失项、没有新 head 权重初始化；
3. **v11 的训练入口沿用 v8 的 YOLO 类**——不需要 `from ultralytics import YOLOv11`，`YOLO("yolo11n.yaml")` 即可启动训练，管线完全走 v8 通用路径。

## 7. 训练命令与常见场景

> **v1~v10 读者路标**：v10 需要 `from ultralytics import YOLOv10`；v11 完全不需要——`YOLO("yolo11n.yaml")` 即可，入口与 v8 相同。

**基础训练**：

```bash
yolo detect train \
  model=yolo11n.yaml \
  data=coco.yaml \
  imgsz=640 \
  epochs=100 \
  batch=64 \
  device=0 \
  workers=8
```

**微调（加载官方权重）**：

```bash
yolo detect train \
  model=yolo11s.pt \
  data=mydataset.yaml \
  imgsz=640 \
  epochs=50 \
  batch=32 \
  lr0=0.001 \
  patience=15
```

- **微调建议**：官方预训练权重从 COCO 起点，转到小数据集时把 lr0 降到 0.001、patience 提到 15 避免早期早停
- **AMP 默认开启**：`amp: True`——如需关闭，显式传 `amp=False`

**训练曲线预期**（COCO80 官方数据，100 epochs，imgsz 640，batch 64，v11-n）：

| 阶段 | epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 |
|------|-------|----------|----------|----------|-------|----------|
| warmup | 0-3 | 3.5→1.2 | 4.5→1.5 | 1.8→1.5 | — | — |
| 主下降 | 3-50 | 1.2→0.9 | 1.5→0.5 | 1.5→1.2 | 0→80 | 0→38 |
| 精细调优 | 50-90 | 0.9→0.75 | 0.5→0.35 | 1.2→1.1 | 80→87 | 38→42 |
| close_mosaic | 90-100 | 0.75→0.72 | 0.35→0.33 | 1.1→1.05 | 87→88 | 42→43 |

**（上表数字为构造示意，反映 v8 通用训练曲线形态；官方 mAP50-95 数字以博客为准，snapshot 内不可复核）**

**推理开销**：零开销。以上所有配置都发生在训练期，部署模型与这些配置无关。

## 8. v11 训练配方"减法"总结

> **v1~v10 读者路标**：把 v8 到 v11 的训练配方差异用"加减法"总结——v9 加辅助分支、v10 加一致性双分配、v11 两者都撤回，唯一留下的是 TAL 默认超参改动。

```
v8 训练配方
  ↓ v9 加：0.25 辅助分支权重 + CBLinear/C2SPC 结构 + YOLOv9 类
  ↓ v10 减：撤回 v9 的辅助分支，改为一致性双分配 + YOLOv10 类
  ↓ v11 减：撤回 v10 的 v10Detect 双头与 v10DetectLoss，回归 v8 YOLO 类
       v11 加：TAL topk=10→13, α=0.5→1.0（唯一配方改动）
```

**结论**：v11 是 v8 训练配方的**第五次**直接沿用（v8 → v10 → v11 三代一致），也是本快照内**唯一有配方差异的**版本（TAL 默认超参）——v9 有 0.25 辅助权重、v10 有 6 项损失 × 1.0，都算"配方改动"。

**为什么训练配方稳定**：ultralytics 官方把训练配方当作"工程默认值"来维护，结构创新（C3k2 / C2PSA / Detect 头回归）与训练配方解耦——用户改结构不换配方，才能公平比较不同版本的性能差异。这也是 v11 相对 v10 的"100% Faster, 100% Smarter"卖点的核心工程承诺：**配方不动，只动结构**。

## 延伸阅读

- [理论 Concept](/versions/v11/concepts)（C3k2 与 C2PSA 的逐行拆解）
- [v8 训练技巧](/versions/v8/tricks)（v11 配方的直接出处——逐键同源）
- [v9 训练技巧](/versions/v9/tricks)（唯一在训练配方上加过东西的版本——辅助分支加权 0.25）
- [v10 训练技巧](/versions/v10/tricks)（一次训练配方回退——一致性双分配）
- [YOLOv11 损失函数](/versions/v11/losses)——三分量损失与 TAL topk=13 的代码细节
- [YOLOv11 反向传播推导](/versions/v11/loss-gradient)——梯度层面的"减法"
- [YOLOv11 痛点追踪](/versions/v11/painpoints)——训练侧的历史遗留与新痛点
- [术语表](/glossary)——Mosaic、MixUp、CopyPaste、EMA、Cosine、close_mosaic 等术语索引
