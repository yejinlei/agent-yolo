# ultralytics/models/yolo/detect/train.py

YOLO 目标检测训练器：**~271 行、单一 DetectionTrainer 类**，Master 直接继承，未做任何修改。它是 Master 训练管线的入口——所有超参、loss 集成、MoE 辅助损失都在这里串联。

> 快照路径：`source/yolomaster/snapshot/ultralytics/models/yolo/detect/train.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/models/yolo/detect/train.py" />

**阅读路线**：

- [L1-L21 文件头](#L1-L21)：import BaseTrainer / DetectionModel / DEFAULT_CFG
- **DetectionTrainer**（核心）[L24-L271](#L24)：继承 `BaseTrainer`
  - [L55 __init__](#L55)：接收 cfg / overrides / _callbacks
  - [L65-L77 build_dataset](#L65-L77)：构建 YOLO 数据集，`gs = max(stride.max(), 32)` 是骨干最大 stride
  - [L79-L105 get_dataloader](#L79-L105)：DDP 环境用 `torch_distributed_zero_first`，val 模式 workers 翻倍
  - [L107-L137 preprocess_batch](#L107-L137)：
    - [L116-L118 到 device](#L116-L118)：`non_blocking` 加速
    - [L119 归一化](#L119)：`img.float() / 255`
    - [L120-L136 multi_scale 增强](#L120-L136)：随机 resize 到 imgsz 的 `multi_scale` 比例内
  - [L139-L150 set_model_attributes](#L139-L150)：
    - 把 nc / names / args 挂到 model
    - [L148-L149 end2end 检测](#L148-L149)：`set_head_attr(max_det=args.max_det)`
  - [L157-L186 class weights](#L157-L186)：
    - [L162 compute_class_weights](#L162)：`weights = (1/count)^cls_pw`
    - [L167-L186 set_class_weights](#L167-L186)：归一化到 mean=1
  - [L188-L204 get_model](#L188-L204)：`DetectionModel(cfg, nc=nc, ch=channels)` 建网
  - [L206-L211 get_validator](#L206-L211)：
    - [L208 loss_names](#L208)：**关键**——`("box_loss", "cls_loss", "dfl_loss")` 三分量
    - **注意**：loss_names 里没有 MoE 相关损失——**MoE 的辅助损失走 `MoELoss.aux_loss` 独立通道**，不进入 label_loss_items（[痛点 8](../painpoints#8-moeloss-是-无主-辅助损失-pain-loss-integration)）
  - [L213-L228 label_loss_items](#L213-L228)：把 loss 值拼成 dict
  - [L230-L238 progress_string](#L230-L238)：训练日志格式化
  - [L240-L258 plot_training_samples / plot_training_labels](#L240-L258)：训练可视化
  - [L260-L271 auto_batch](#L260-L271)：自动 batch size 计算

## 与 v8/v12 train.py 的对比

| 维度 | v8 DetectionTrainer | v12 DetectionTrainer | Master DetectionTrainer |
|------|---------------------|----------------------|-------------------------|
| 行数 | ~200 | ~260 | **~271（完全一致于 v12）** |
| loss_names | box/cls/dfl | box/cls/dfl | **完全一致** |
| MoE 集成 | 无 | 无 | **通过 MoELoss 独立挂接** |
| Multi-scale | 无 | 有 | 完全一致 |
| Class weights | 无 | 有 | 完全一致 |

**Master 训练的关键点**：

1. **`get_validator.loss_names` 只有三个**：box/cls/dfl——MoE 的 balance/z-loss **不出现在训练日志里**，只能通过 `MoELoss.aux_loss.item()` 手动观察
2. **MoE 集成的位置**：`DetectionModel.init_criterion`（在 `nn/tasks.py`）里 `return v8DetectionLoss(self) + MoELoss(...)`——**MoELoss 是挂在 DetectionModel 上的属性，不是 DetectionTrainer 的一部分**
3. **end2end 检测**：[L148-L149](#L148-L149) 判断 `self.model.end2end`——Master yaml 默认 `end2end=False`，所以走 v8DetectionLoss 而非 E2EDetectLoss
4. **DDP 支持**：`torch_distributed_zero_first` 保证 dataset cache 只初始化一次；`get_dataloader` 里 rank=0 负责建 cache

**为什么 train.py 没变但 Master 能训练**：因为 Master 的所有增量都在 `DetectionModel` / `ES_MOE` / `MoELoss`——`DetectionTrainer` 只需要按标准流程跑 forward + backward，MoE 的辅助损失通过 `MoELoss.forward` 返回并被 BaseTrainer 求和。**训练入口没变，损失组件变了**（[痛点 8](../painpoints#8-moeloss-是-无主-辅助损失-pain-loss-integration)）。

**与 yaml 的联动**：`yolo-master-n.yaml` 里的 scales 决定 model 大小，`nc=80` 传给 `set_model_attributes`，`args.max_det` 传给 `set_head_attr`——**train.py 是所有配置串起来的最后一个环节**。

Detect 头见 [head.py 源码页](./head-py)，MoE 损失见 [moe-loss-py 源码页](./moe-loss-py)，架构总览见 [architecture 页](../architecture)，走读总览见 [code-walkthrough 页](../code-walkthrough)。

## 常见问题

**Q：MoE 损失在哪里显示？**
A：MoE 的辅助损失**不进入 label_loss_items**——train.py L208 `loss_names = ("box_loss", "cls_loss", "dfl_loss")` 只有三分量。用户需要手动从 `model.loss` 里取 MoELoss.aux_loss 才能看到。

**Q：如何训练 Master 模型？**
A：
```python
from ultralytics import YOLO
model = YOLO("yolo-master-n.yaml")
model.train(data="coco8.yaml", epochs=3, batch=8, imgsz=640)
```
`DetectionTrainer` 会按标准流程跑，MoE 的辅助损失自动挂接。

**Q：multi_scale 增强是什么？**
A：[L120-L136 preprocess_batch](#L120-L136) 里的 multi_scale 参数——每个 batch 随机 resize 到 `imgsz * (1 ± multi_scale)` 的范围，增强尺度鲁棒性。

**Q：DDP 环境下怎么建数据集？**
A：[L92-L94 get_dataloader](#L92-L94) 用 `torch_distributed_zero_first`——**只有 rank=0 建 dataset cache，其他 rank 等待**，避免重复 IO。

**Q：class_weights 是什么？**
A：[L162-L186 set_class_weights](#L162-L186)——处理类别不平衡，`weights = (1/count)^cls_pw`，归一化到 mean=1。COCO 上默认 `cls_pw=0.0` 不启用。
