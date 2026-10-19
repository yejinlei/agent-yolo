# YOLOv8 训练技巧

v8 的技巧清单与 v7 性质不同：v7 的 trick 是论文论证过的 **freebies**（训练加料、推理免费），v8 的 trick 多是**工程化默认值**——把 v6/v7 验证过的好东西（TAL、DFL、C2f）做成"不改参数就享受"的出厂设置，再配上一套写死在 `default.yaml` 的训练配方。本页 7 条按此如实盘点。

## 全部 Trick 速查

| Trick | 代码位置 | 说明 | 推理是否零开销 |
|-------|----------|------|----------------|
| C2f 取代 C3 | block.py L169-189 | split → n×Bottleneck → 全拼接，n+2 路 concat | 是（纯结构） |
| anchor-free 与 TAL 全面落地 | tal.py L57-81 / detect/train.py L156 | topk=10、α=0.5、β=6.0 出厂即用 | 是（分配纯训练期） |
| DFL 内建为默认 | head.py L41 / loss.py L50-59 | reg_max=16，回归出 64 通道分布 | 是（固定权重算子） |
| close_mosaic 末期关闭 | default.yaml L30 / trainer.py L293-299 | 默认 0，CLI 常用 10 | 是（纯训练期） |
| 训练默认超参 | default.yaml L80-106 | SGD lr0=0.01，box/cls/dfl=7.5/0.5/1.5 | 是（纯配置） |
| 单一代码库与 YAML 配置回归 | yolov8.yaml L6-12 | scales 五档复合缩放，yaml 层表回归 | 是（纯配置） |
| 多任务统一损失继承 | segment/train.py L62 / pose/train.py L76 | SegLoss/PoseLoss 继承检测 Loss | 是（纯训练期） |

## 1. C2f 取代 C3 (`trick-c2f`)

**动机**：C3（v5 起用，见 [C3 词条](/glossary#c3)）只有两条路进 concat——bottleneck 堆栈的输出与 cv1 直通的那一半；梯度想从深层抄近路，只有 bottleneck 内部那条残差。v7 用 ELAN（见 [E-ELAN 词条](/glossary#elan-e-elan-高效层聚合)）证明了"多路梯度高速"的价值，v8 把这思想装回 CSP 壳里且更省：**C2f 让 bottleneck 的每一级输出都进 concat**。

**实现**（<SrcLink file="v8:ultralytics/nn/modules/block.py" :line="169" :end="189" />）：cv1 出 `2c` 通道 chunk 成两半，第一半逐级过 n 个 Bottleneck，每级输出 `extend` 进列表，最后全部拼接过 cv2：

```python
def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))     # 劈两半
    y.extend(m(y[-1]) for m in self.m)    # n 个 Bottleneck 逐级输出全部收集
    return self.cv2(torch.cat(y, 1))      # n+2 路 concat
```

- cv2 的入通道因此是 `(2 + n) * self.c`（<SrcLink file="v8:ultralytics/nn/modules/block.py" :line="176" :end="176" />），对比 C3 的 `2 * c_`——多出的 n 路就是新梯度旁路
- yaml 里骨干 C2f 带 `shortcut=True`（如 `[-1, 6, C2f, [256, True]]`），head 里的 C2f 不传该参数、走默认 `shortcut=False`；重复数 n 由 parse_model 统一注入 `args.insert(2, n)`（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="500" :end="502" />）
- 骨干重复 3/6/6/3、head 重复 3/3/3/3，depth 缩放（n 档 ×0.33）后取整

**效果**：更密的梯度分流 + 更少参数——n 档全模型 3,157,200 参数、8.9 GFLOPs，比同档 v7（36.9M）小一个量级仍在 COCO 拿 37.3 mAP。逐级 concat 的收益可对比 v6 的 CSPStackRep：同一思想在不同 CSP 壳里的两次工程落地。

**推理开销**：零开销。纯结构设计，训练推理同形态。

## 2. anchor-free 与 TAL 全面落地 (`trick-tal`)

**动机**：anchor 用不用，v6（anchor-free + TAL）与 v7（anchor-based + 双头分配）在 2022 年给出相反答案（见 [v7 痛点：路线分叉](/versions/v7/painpoints)）。v8 一锤定音：**全线 anchor-free + TAL，无开关、无备选**——TAL 不再是 v6 式的可选配置，而是唯一出厂分配器。

**实现**：

- **头侧**：Detect 无 anchor 张量，`anchors = torch.empty(0)` 只是占位（<SrcLink file="v8:ultralytics/nn/modules/head.py" :line="24" :end="28" />）；推理时 `make_anchors` 生成网格中心点，`grid_cell_offset=0.5` 取格子中心（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="246" :end="258" />），解码用 `dist2bbox` 把 ltrb 距离加回中心（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="261" :end="270" />）
- **分配侧**：`TaskAlignedAssigner` 类默认 topk=13（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="57" :end="81" />），检测训练实例化时改为 **topk=10、alpha=0.5、beta=6.0**（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="156" :end="156" />）；对齐度量 `align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)`（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="157" :end="160" />）
- 正样本 = **in_gt × topk × mask_gt 三重掩码**（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="129" :end="139" />）——中心落在 GT 内、度量进 top-k、GT 掩码有效三者同时成立；一个点被多个 GT 抢时由 `select_highest_overlaps` 按 IoU 最高裁决（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="29" :end="54" />）
- 归一化的 align_metric 乘回 one-hot 得 **target_scores 软标签**（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="120" :end="125" />）

**效果**：640 输入出 8400 个预测（6400+1600+400），无 objectness 分支省掉一整路参数；分类分数即框置信度，部署侧不用再算 obj×cls 联合打分。

**推理开销**：零开销。分配只发生在训练期；anchor-free 头本身还比 anchor-based 头少 3 组 anchor 的输出通道。

## 3. DFL 内建为默认 (`trick-dfl-default`)

**动机**：DFL（GFL 论文提出，见 [DFL 词条](/glossary#dfl-distribution-focal-loss)）在 v6 是服务于**框回归自蒸馏**的可选项——标准训练默认不开。v8 把它扶正：**距离分布表示 + DFL 损失是检测头的出厂配置**，`reg_max=16` 写死在 Detect 构造函数里。

**实现**：

- 头侧：`self.reg_max = 16`、`self.no = nc + self.reg_max * 4`（<SrcLink file="v8:ultralytics/nn/modules/head.py" :line="34" :end="35" />），box 分支出口是 `4 * reg_max = 64` 通道（<SrcLink file="v8:ultralytics/nn/modules/head.py" :line="37" :end="40" />）；`self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()`（<SrcLink file="v8:ultralytics/nn/modules/head.py" :line="41" :end="41" />）
- DFL 模块本体是个**固定权重**的 1×1 conv：权重预填 `arange(16)`，forward 里 softmax 后做期望值积分（<SrcLink file="v8:ultralytics/nn/modules/block.py" :line="18" :end="36" />）——16 个 bin 的分布一步积分成标量距离
- 损失侧：`use_dfl = m.reg_max > 1`（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="154" :end="154" />）；`_df_loss` 只打真值两侧最近的两个 bin，线性加权交叉熵（<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="50" :end="59" />），与 CIoU 一起在 BboxLoss 里按 target_scores 加权

**效果**：回归从"直接出 4 个数"变成"出 4 组 16-bin 分布取期望"——定位边界的不确定性有了表达空间，小目标的模糊边界不再被迫押注单点。

**推理开销**：零开销。DFL 的 1×1 conv 权重是固定 arange、`requires_grad_(False)`，部署时就是一个普通算子（或预先融合）。

## 4. close_mosaic 末期关闭 (`trick-close-mosaic`)

**动机**：[Mosaic](/glossary#mosaic-增强) 四图拼接是强增强，训练末期目标分布（单图少物体、位置均匀）与真实分布（多物体、密集遮挡）偏离拉大——收尾阶段关掉增广让模型回到真实分布上微调，是 v4 以来社区的经验做法，v8 第一次把它做成**引擎级一键开关**。

**实现**：

- 超参默认 `close_mosaic: 0`，注释 "(int) disable mosaic augmentation for final epochs"（<SrcLink file="v8:ultralytics/yolo/cfg/default.yaml" :line="30" :end="30" />）——默认不启用，CLI 常用 `--close_mosaic 10`
- 触发逻辑在 BaseTrainer 主循环：`epoch == epochs - close_mosaic` 时打日志 "Closing dataloader mosaic" 并调 `dataset.close_mosaic(hyp=self.args)`（<SrcLink file="v8:ultralytics/yolo/engine/trainer.py" :line="293" :end="299" />）
- `close_mosaic` 方法把 `mosaic`、`copy_paste`、`mixup` 三个概率**全部归零**并重建 transforms（<SrcLink file="v8:ultralytics/yolo/data/dataset.py" :line="163" :end="168" />）——注释里写明 "keep the same behavior as previous v8 close-mosaic"，即一次关闭三件套是 v8 起的既定行为

**效果**：末期若干 epoch 以真实分布收尾，mAP50-95 通常更高；与 [closes 系超参](/versions/v8/concepts#_10-close-mosaic-末期关闭增广-concept-close-mosaic)一起构成"先猛增广、后真分布"的两段式训练。

**推理开销**：零开销。纯训练期配置，且默认关闭。

## 5. 训练默认超参 (`trick-hparams`)

**动机**：v5/v7 靠多份 hyp 文件分档手动挑选（v7 有 p5/p6/tiny/custom 四份），v8 收敛为**一份 `default.yaml` 写死"中强度 COCO 默认"**——n 到 x 五档共用一套，命令行只覆盖想改的键。

**实现**（<SrcLink file="v8:ultralytics/yolo/cfg/default.yaml" :line="79" :end="106" />）：

| 组 | 默认值 | 位置 |
|----|--------|------|
| 优化器 | SGD（可选 Adam/AdamW/RMSProp） | L23 |
| 学习率 | lr0=0.01、lrf=0.01（终率 = lr0×lrf）、momentum=0.937 | L80-82 |
| 正则 | weight_decay=5e-4 | L83 |
| warmup | 3 epoch（momentum 0.8 → 0.937、bias_lr 0.1，见 [warmup 词条](/glossary#warmup-热身)） | L84-86 |
| 损失增益 | **box=7.5 / cls=0.5 / dfl=1.5** | L87-89 |
| 几何增广 | translate=0.1、scale=0.5、degrees=0、fliplr=0.5、flipud=0 | L97-103 |
| 色彩增广 | hsv_h=0.015、hsv_s=0.7、hsv_v=0.4 | L94-96 |
| 拼图类 | mosaic=1.0、mixup=0.0、copy_paste=0.0 | L104-106 |
| 训练长度 | epochs=100、patience=50（早停） | L10-11 |

- Loss 类构造时整包接住超参 `self.hyp = h`，`__call__` 末尾 `loss[0] *= self.hyp.box; loss[1] *= self.hyp.cls; loss[2] *= self.hyp.dfl`（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="226" :end="228" />）——三分量增益在代码里的落点就这三行
- 配置体系带类型校验（CFG_INT_KEYS 等按类型合并 CLI 覆盖），覆盖即校验，改错当场报错

**效果**：`yolo detect train model=yolov8n.pt data=coco.yaml` 一行即可复现官方基线；新手不用再理解 anchor_t、loss_ota 这类 v5/v7 时代的开关——因为它们连同 anchor 一起消失了。

**推理开销**：零开销。纯训练配置。

## 6. 单一代码库与 YAML 配置回归 (`trick-unified-repo`)

**动机**：v6 把 v5 的 yaml 换成 Python dict（见 [Python dict 配置词条](/glossary#python-dict-配置)），v7 又拆成 training/deploy 双份 yaml 加两套训练脚本（单头/双头分叉）——配置形态三年三变，工程上各自为政。v8 回归**yaml 层表 + 单库统一入口**，并且在一个仓库里管四个任务。

**实现**：

- **YAML 回归**：网络描述回到 `[from, repeats, module, args]` 四元组层表（<SrcLink file="v8:ultralytics/models/v8/yolov8.yaml" :line="15" :end="46" />），由 parse_model 统一翻译成 nn.Module（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="461" :end="534" />）——见 [yaml 驱动架构词条](/glossary#yaml-驱动架构)
- **scales 五档**：n/s/m/l/x 写成 `[depth, width, max_channels]` 三元组（<SrcLink file="v8:ultralytics/models/v8/yolov8.yaml" :line="6" :end="12" />），parse_model 读出后 `depth, width, max_channels = scales[scale]`（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="467" :end="474" />）；`max_channels` 是 v8 新增的第三参数，把 l/x 档的大通道压到 512 防止参数爆炸——对比 [compound scaling 词条](/glossary#compound-scaling-复合缩放)的 v7 两参数版
- **单库多任务**：顶层只导出 `YOLO` / `RTDETR` / `SAM`（<SrcLink file="v8:ultralytics/__init__.py" :line="3" :end="11" />）；detect/segment/pose/classify 共用 engine（BaseTrainer/BaseValidator/predictor/exporter），同库还带 ByteTrack/BOT-Sort 跟踪器——一个 `YOLO('yolov8n.pt')` 对象即全部入口

**效果**：一份 yaml + 一个 `scale` 后缀 = 五个模型；新任务 = 新 trainer 继承 + 新 head 模块，配置/训练/验证/导出全链路复用。

**推理开销**：零开销。纯工程治理。

## 7. 多任务统一损失继承 (`trick-multitask-loss`)

**动机**：检测/分割/姿态的损失框架同构——都要"分配目标 → 回归 + 分类（+ 各自增量项）"，没必要各写一套。v7 的双头损失已经是两份几乎复制的实现；v8 直接用**类继承**表达这种复用。

**实现**：

- 基类是检测 `Loss` 类（v8.0.100 里就叫这个名字，后世文档称 v8DetectionLoss，见 [命名漂移痛点](/versions/v8/painpoints)）：TAL 分配 + BboxLoss + BCE 软标签的总装（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="138" :end="230" />）
- `SegLoss(Loss)`（<SrcLink file="v8:ultralytics/yolo/v8/segment/train.py" :line="62" :end="62" />）：构造函数 `super().__init__(model)` 全盘复用 TAL/BboxLoss，增量只有 mask 相关系数与 overlap 损失两项
- `PoseLoss(Loss)`（<SrcLink file="v8:ultralytics/yolo/v8/pose/train.py" :line="76" :end="76" />）：同样先 `super().__init__(model)`，增量是 `KeypointLoss`（OKS sigma 加权的关键点回归），返回的 loss 向量从 3 分量扩到 5 分量（box/cls/dfl/kpt_location/kpt_visibility）

**效果**：加一个任务 ≈ 继承 + 一个增量损失项 + 一个 head 模块；检测侧的分配与回归逻辑改动一次，三个任务同时受益。详细的分量拆解见[损失函数页](/versions/v8/losses#_7-多任务损失继承-segloss-poseloss-loss-multitask)。

**推理开销**：零开销。继承只存在于训练期代码组织，推理图不变。

## 延伸阅读

- [理论 Concept](/versions/v8/concepts)
- [损失函数](/versions/v8/losses)
- [源码走读](/versions/v8/code-walkthrough)
- [痛点追踪](/versions/v8/painpoints)
- [YOLOv7 训练技巧](/versions/v7/tricks)（对照看配方传承——v7 是论文 freebies，v8 是工程默认值）
- [v6 训练技巧](/versions/v6/tricks)（TAL 的第一次登场）
