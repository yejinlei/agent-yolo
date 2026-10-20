# YOLOv9 训练技巧

v8 的 trick 是「工程化默认值」；v9 的性质又不一样——**引擎整体继承 v7 代码库**（dataloaders、损失文件结构、多入口脚本都是 v7 底子），训练侧的功夫花在「高强度超参 + 双头配方 + 结构侧的 PGI」上：损失与增强的出厂值大量沿用旧版，真正的新 trick 是结构性的（GELAN / PGI，见[架构解析](/versions/v9/architecture)）。本页 5 条按此如实盘点。

## 全部 Trick 速查

| Trick | 代码位置 | 说明 | 推理是否零开销 |
|-------|----------|------|----------------|
| 三入口对应单/双/三头 | train.py L39 / train_dual.py L40 / train_triple.py L40 | gelan 单头、yolov9 双头、Triple 变体 | 是（辅助分支推理拆除） |
| 双头损失权重 1.0 / 0.25 | loss_tal_dual.py L221-245 | 主头主导梯度（主 ×1.0 / 辅 ×0.25），PGI 的训练面 | 是（纯训练期） |
| README 训练配方 | README.md L91/L101 | 500 epochs + close-mosaic 15 + hyp.scratch-high | 是（纯配置） |
| 高强度增强包 | hyp.scratch-high.yaml L18-30 | mosaic 1.0 + mixup 0.15 + copy_paste 0.3 | 是（纯训练期） |
| close_mosaic 末期关闭 | train_dual.py L272-274 | 末 15 epoch 停 mosaic，mixup/copy_paste 随分支一起停 | 是（纯训练期） |
| EMA + warmup + 线性/OneCycleLR | train_dual.py L168 / L294-302 / L146-164 | 与 v7 引擎同款调度 | 是（纯训练期） |
| 损失增益 7.5/0.5/1.5 | loss_tal_dual.py L247-249 | 与 v8 相同口径，写死在损失文件 | 是（纯配置） |
| 辅助分支只训练时存在 | yolov9-c.yaml L81-116 + DualDDetect | 结构由 yaml 决定，hyp 无 PGI 开关 | 是（推理整体拆除） |

## 1. 训练配方 三入口与 README 命令行 (`trick-recipe`)

**动机**：v7 用两套训练脚本区分单头/双头（[v7 tricks](/versions/v7/tricks)），v8 收敛成单一入口；v9 介于两者之间——**同一引擎、三个入口**，因为头数不同需要不同的 forward 返回结构与损失类，索性一人一个脚本。

**实现**：

- **yolov9 系列（双头）用 train_dual.py**，README 单卡命令（<SrcLink file="v9:README.md" :line="91" />）：

```bash
python train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 \
    --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c \
    --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
```

- **8 卡 DDP** 把 batch 提到 128 并加 `--sync-bn`（<SrcLink file="v9:README.md" :line="101" />）：

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_dual.py \
    --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 \
    --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c \
    --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
```

- **gelan 系列（单头基线）用 train.py**，默认 batch 32（<SrcLink file="v9:README.md" :line="94" />）——同一份 hyp.scratch-high.yaml
- **入口与损失类的对应**：train.py 导入 `utils.loss_tal` 的单头 `ComputeLoss`（<SrcLink file="v9:train.py" :line="39" />）、train_dual.py 导入 `utils.loss_tal_dual` 的双头 `ComputeLoss`（<SrcLink file="v9:train_dual.py" :line="40" />，L41-42 注释掉 lead-head 备选，见[损失页](/versions/v9/losses#_6-computelosslh-lead-head-引导版-loss-lh)）、train_triple.py 导入 `utils.loss_tal_triple`（<SrcLink file="v9:train_triple.py" :line="40" />）；验证同理分 val.py / val_dual.py（README 示例 <SrcLink file="v9:README.md" :line="53" />）/ val_triple.py
- **脚本内的关键默认**：`--hyp` 默认就是 hyp.scratch-high.yaml（<SrcLink file="v9:train_dual.py" :line="445" />）、`--epochs` 默认 100、`--batch-size` 默认 16（<SrcLink file="v9:train_dual.py" :line="446" :end="447" />）、`--optimizer` 默认 SGD（<SrcLink file="v9:train_dual.py" :line="462" />）、`--close-mosaic` 默认 0（<SrcLink file="v9:train_dual.py" :line="479" />）——README 命令里显式写出的 500/15 都是 CLI 覆盖
- **梯度累积**：名义 batch nbs=64，`accumulate = max(round(nbs / batch_size), 1)`，weight_decay 按 `batch_size × accumulate / nbs` 等比缩放（<SrcLink file="v9:train_dual.py" :line="140" :end="142" />）——v7 引擎的既定行为

**效果**：官方口径下 yolov9-c 在 COCO val 640 拿 53.0% AP（25.5M 参数 / 102.8G FLOPs）、yolov9-e 55.6%（58.1M / 192.5G），均出自 README（COCO val 640 口径）；与 gelan-c 同配方对照正是论文消融的结构变量控制。

**推理开销**：零开销。纯训练配置。

## 2. 数据增强 高强度包与 close_mosaic (`trick-aug`)

**动机**：v8 default.yaml 的拼图类增强是 mosaic 1.0、mixup 0.0、copy_paste 0.0（中强度）；v9 的 hyp.scratch-high.yaml 把后两个也拉起来——**v7→v8→v9 三代里最强的出厂增强包**，配合 500 epochs 长训练。

**实现**（<SrcLink file="v9:data/hyps/hyp.scratch-high.yaml" :line="18" :end="30" />）：

| 组 | 值 | 行 |
|----|----|----|
| 色彩 | hsv_h=0.015、hsv_s=0.7、hsv_v=0.4 | L18-20 |
| 几何 | degrees=0、translate=0.1、scale=0.9、shear=0、perspective=0 | L21-25 |
| 翻转 | flipud=0.0、fliplr=0.5 | L26-27 |
| 拼图类 | **mosaic=1.0、mixup=0.15、copy_paste=0.3** | L28-30 |

- **v9 继承 v7 的 dataloaders.py**：增强管线在 `LoadImagesAndLabels.__getitem__`——`self.mosaic = self.augment and not self.rect`（<SrcLink file="v9:utils/dataloaders.py" :line="454" />），掷骰 `random.random() < hyp['mosaic']`（<SrcLink file="v9:utils/dataloaders.py" :line="652" />）进 4 图拼接
- **mixup 嵌在 mosaic 分支里**：mosaic 命中后再以 `hyp['mixup']` 概率叠一次 MixUp（<SrcLink file="v9:utils/dataloaders.py" :line="659" :end="660" />）；**copy_paste 同样只活在 mosaic 分支**——在 `load_mosaic` 内部对 segments 执行 `copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])`（<SrcLink file="v9:utils/dataloaders.py" :line="791" />、<SrcLink file="v9:utils/dataloaders.py" :line="868" />）；HSV 与随机透视变换在两种路径共用的收尾段（<SrcLink file="v9:utils/dataloaders.py" :line="676" :end="682" />、<SrcLink file="v9:utils/dataloaders.py" :line="694" />）
- **close_mosaic 末期关闭**：主循环里 `epoch == epochs - close_mosaic` 时打日志并 `dataset.mosaic = False`（<SrcLink file="v9:train_dual.py" :line="272" :end="274" />），README 配方给 15——只翻转 mosaic 标志，**但 mixup / copy_paste 都嵌在 mosaic 分支内，实际三者一起停**；另外 `create_dataloader(close_mosaic=...)` 会改用普通 DataLoader 而非 InfiniteDataLoader（<SrcLink file="v9:train_dual.py" :line="200" />、<SrcLink file="v9:utils/dataloaders.py" :line="140" />）
- 与 v8 的差异：v8 是引擎级 `dataset.close_mosaic()` 显式把 mosaic/copy_paste/mixup 三个概率归零（[v8 tricks](/versions/v8/tricks#_4-close-mosaic-末期关闭-trick-close-mosaic)）；v9 靠「嵌套结构 + 单一标志位」达到同样效果，机制不同、结果相同
- scale=0.9 也比 v8 默认（0.5）激进——随机透视变换的缩放幅度上限拉到 ±90%

**效果**：强增强 + 长训练（500 epochs）是 scratch-high 档的定位（v7 时代该档就是给大模型从头训的配方）；v9 的 PGI 辅助分支提供更有效的梯度，恰好支撑得起这种高强度训练而不崩。

**推理开销**：零开销。纯训练期管线。

## 3. 优化器 EMA 与学习率调度 (`trick-optimizer`)

**动机**：v9 引擎复用 v7 的 `smart_optimizer / ModelEMA` 套件——优化策略与 v7/v8 一脉相承，hyp 里 lr0=0.01、momentum=0.937、weight_decay=0.0005 与 v8 default.yaml 同值（<SrcLink file="v9:data/hyps/hyp.scratch-high.yaml" :line="1" :end="4" />）。

**实现**：

- **优化器**：`smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])`（<SrcLink file="v9:train_dual.py" :line="143" />；实现 <SrcLink file="v9:utils/torch_utils.py" :line="314" />）——三组参数（bias / weight / BN），bias 不做 weight decay
- **[warmup](/glossary#warmup-热身)**：`warmup_epochs: 3.0`，前 3 epoch bias lr 从 0.1 线性降到 lr0、其余参数 lr 从 0 升到 lr0、momentum 从 0.8 升到 0.937（<SrcLink file="v9:train_dual.py" :line="294" :end="302" />；hyp <SrcLink file="v9:data/hyps/hyp.scratch-high.yaml" :line="5" :end="7" />）
- **调度四选一**：默认线性衰减 `lf = (1 - x/epochs)·(1 - lrf) + lrf`（<SrcLink file="v9:train_dual.py" :line="153" />）；`--cos-lr` 切 [OneCycleLR](/glossary#学习率-lr-decay) 式余弦 `one_cycle(1, hyp['lrf'], epochs)`（<SrcLink file="v9:train_dual.py" :line="146" :end="147" />；one_cycle 实现 <SrcLink file="v9:utils/general.py" :line="676" />）、`--flat-cos-lr` 平台余弦（<SrcLink file="v9:train_dual.py" :line="148" :end="149" />）、`--fixed-lr` 恒定（<SrcLink file="v9:train_dual.py" :line="150" :end="151" />）；lrf=0.01 即终率 = lr0 × 0.01（<SrcLink file="v9:data/hyps/hyp.scratch-high.yaml" :line="2" />）——README 配方未指定，走默认线性
- **[EMA](/glossary#ema-指数移动平均)**：`ModelEMA` 只在主进程创建（<SrcLink file="v9:train_dual.py" :line="168" />；类实现 <SrcLink file="v9:utils/torch_utils.py" :line="501" />），每次 optimizer step 后 `ema.update(model)`（<SrcLink file="v9:train_dual.py" :line="331" :end="332" />），验证与 best.pt 都用 EMA 权重（<SrcLink file="v9:train_dual.py" :line="360" />、<SrcLink file="v9:train_dual.py" :line="382" />）
- **梯度裁剪 + AMP**：`clip_grad_norm_(model.parameters(), max_norm=10.0)`（<SrcLink file="v9:train_dual.py" :line="327" />）；`torch.cuda.amp.GradScaler` 与 autocast 包住前向（<SrcLink file="v9:train_dual.py" :line="255" />、<SrcLink file="v9:train_dual.py" :line="313" :end="322" />）
- **早停**：`--patience` 默认 100 epochs（<SrcLink file="v9:train_dual.py" :line="473" />），`EarlyStopping` 按 fitness 检查（<SrcLink file="v9:train_dual.py" :line="256" />、<SrcLink file="v9:train_dual.py" :line="370" />）——500 epochs 配方下通常不会触发

**效果**：与 v8 相同的「lr0 0.01 + SGD 0.937 + wd 5e-4」手感；500 epochs 长跑下线性衰减的尾部学习率足够小，社区复现时常切 `--cos-lr`。

**推理开销**：零开销。EMA/调度都是训练期机制，部署拿的是收敛后的权重。

## 4. PGI 训练面 辅助分支只在训练期存在 (`trick-pgi`)

**动机**：v9 的核心贡献 PGI（可编程梯度信息）不是一个「可开关的 trick」，而是**结构决策**——辅助分支织进模型 yaml，训练技巧只剩两条：给主分支输出全量监督权重 ×1.0、给辅助分支保留一条 ×0.25 的辅助监督梯度通路，以及训练完把辅助分支拆掉。

**实现**：

- **结构由 yaml 决定，hyp 无 PGI 开关**：yolov9-c.yaml 的 multi-level reversible auxiliary branch 段（<SrcLink file="v9:models/detect/yolov9-c.yaml" :line="81" :end="116" />）写死 CBLinear 抽头（5/7/9 层）、辅助主干（从 Silence 重新下采样）与 3 处 CBFuse；检测头一层声明主辅 6 路输入（<SrcLink file="v9:models/detect/yolov9-c.yaml" :line="123" />）。hyp.scratch-high.yaml 里找不到任何 pgi/auxiliary 字样——想开或关 PGI，换 yaml（gelan-c.yaml 无辅助分支即基线，<SrcLink file="v9:models/detect/gelan-c.yaml" :line="78" />）
- **辅助梯度的定量配比在损失文件**：主分支输出 ×1.0、辅助分支输出 ×0.25（<SrcLink file="v9:utils/loss_tal_dual.py" :line="222" :end="245" />）——想调辅助话语权只能改代码，hyp 不暴露
- **三入口对应三种头数**：train.py（单头）/ train_dual.py（双头）/ train_triple.py（三头，导入 `utils.loss_tal_triple`，<SrcLink file="v9:train_triple.py" :line="40" />）——入口、头、损失类三者绑定，选错入口会因 forward 返回结构不匹配而报错；train_dual.py 的 lead-head 备选 `ComputeLossLH` 是 PGI 论文里引导式分配的代码化，留档未启用（<SrcLink file="v9:train_dual.py" :line="41" :end="42" />）
- **训练期 forward 返回两路**：DualDDetect 训练态返回 `[d1, d2]`（<SrcLink file="v9:models/yolo.py" :line="227" :end="228" />），验证态返回 `(y, [d1, d2])`（<SrcLink file="v9:models/yolo.py" :line="237" :end="238" />）——辅助分支的一切计算只发生在训练/验证期
- 梯度怎么经 CBFuse/CBLinear 回流骨干：见[反向传播推导第 5 节](/versions/v9/loss-gradient#_5-辅助分支梯度的回流路径-grad-cb)

**效果**：辅助分支在训练期为骨干提供「从更深、信息更全的路径」回流的梯度（缓解深层信息瓶颈），推理期零负担——这是 v9 能把 yolov9-e 推到 55.6% AP（README，COCO val 640）而推理图不带辅助计算的原因。

**推理开销**：零开销（相对去掉辅助分支的纯主干推理）；若直接部署训练权重且不裁辅助输出， DualDDetect 的 export 态仍会算出辅助路输出（<SrcLink file="v9:models/yolo.py" :line="237" :end="238" />），见下节。

## 5. 部署注意 export.py 与辅助分支拆除 (`trick-deploy`)

**动机**：v9 README 没有 v8 那种系统化文档，也没有任何 ONNX 量化条目——部署相关只给了验证命令与继承自 v7 引擎的 export.py，量化/精简需要自己动手。

**实现**：

- **导出走 export.py**：快照自带标准导出脚本（<SrcLink file="v9:export.py" :line="1" />），支持 torchscript / onnx / coreml / tflite 等格式，随 v7 引擎行为：导出前 `fuse()` 融合 Conv+BN、`model.eval()` 切换 export 模式；README **未提供** ONNX 量化（INT8/FP16 校准）条目——量化需求要自备工具链（onnxruntime / tensorrt 校准流程不在仓库内）
- **辅助分支的拆除**：export 模式下 DualDDetect 只返回 `y`（主辅两路输出的列表，<SrcLink file="v9:models/yolo.py" :line="238" />）——推理图里辅助分支的计算**并不会自动消失**；要拿到「纯主分支」的部署图，用无辅助分支的 yaml 重建模型再加载主分支权重（gelan 式结构），或在导出前把辅助输出从图里裁掉。参数口径也要对齐：README 的 25.5M（yolov9-c）是**训练权重**口径，去掉辅助分支后的推理参数更少
- **DFL 是固定权重算子**：头里的 DFL 模块权重是固定 arange 的 1×1 conv（不训练），导出后就是普通算子，或按 v8 的做法预先融合进 dist2bbox
- **重参数化**：GELAN 系的 RepNCSPELAN4 训练推理同形态，无需 RepConv 式融合；fuse() 只处理常规 Conv+BN
- **验证对齐**：部署前先用 val_dual.py 跑一遍官方口径确认（<SrcLink file="v9:README.md" :line="53" />：`--conf 0.001 --iou 0.7`），再对比导出模型——双头的验证脚本与训练脚本一一对应，别拿 val.py 去测双头权重

**效果**：导出即得标准 ONNX；主分支精简后的推理延迟与 gelan 同档结构相当——PGI 的「训练加料、推理免费」承诺在这一步兑现。

**推理开销**：本节就是关于推理开销的：正确拆除辅助分支后，PGI 零推理开销；不拆除则白付一份辅助计算。

## 延伸阅读

- [架构解析](/versions/v9/architecture)
- [损失函数](/versions/v9/losses)
- [反向传播推导](/versions/v9/loss-gradient)
- [YOLOv7 训练技巧](/versions/v7/tricks)（v9 引擎与增强管线的直系来源）
- [v8 训练技巧](/versions/v8/tricks)（对照看 close_mosaic 的两种实现与 default.yaml 式配置的取舍）
- [术语表](/glossary)
