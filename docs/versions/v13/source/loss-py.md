# ultralytics/utils/loss.py

YOLOv13 的损失仓库：**743 行文件里，v13 检测主路径只用 L157 的 `v8DetectionLoss`**——v8 以来的三分量总装（[CIoU](/glossary#ciou-loss) + [DFL](/glossary#dfl-distribution-focal-loss) + BCE）；文件末尾只剩 `E2EDetectLoss`（L728，端到端双分支）这一个遗产类，**`v10DetectLoss` 在 v11/v12/v13 里都不存在**（旧类名由 tasks.py L825 映射到 `E2EDetectLoss`）。**`yolov13.yaml` 不引用它们**。

> 快照路径：`source/yolov13/snapshot/ultralytics/utils/loss.py`（743 行，commit `73289949`）。点击左侧行号可复制/分享行锚点。

**这个文件在 v13 里的角色**：v13 对 loss.py 的**训练行为净增量为零**——「超图消息传递 + 门控连接」是纯结构改动，损失侧完全沿用 v8 三分量总装。v13 与 v11/v12 的 loss.py 差异仅限**代码风格**（license 头、`if n := matches.sum()` 写法、docstring 列表格式），`v8DetectionLoss` 的签名、L176 assigner 显式参数、L177 `BboxLoss(m.reg_max)` 全部逐行相同。**`tal_topk=10` 参数位在 v11 就已存在**，不是 v13 新增。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="16" :end="37" /> | VarifocalLoss | VFL 原式实现，存而未用 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="40" :end="62" /> | FocalLoss | 存而未用 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="65" :end="88" /> | DFLoss | GFL 双 bin 插值 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="91" :end="113" /> | BboxLoss | CIoU + DFL，target_scores 兼当回归权重 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="116" :end="137" /> | RotatedBboxLoss | OBB 专用（probiou） |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="140" :end="154" /> | KeypointLoss | pose 专用 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="157" :end="259" /> | **v8DetectionLoss** | **v13 检测主用**：三分量总装 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="160" /> | __init__ 签名 | **`tal_topk=10`（v11 起就有，v12/v13 沿用）** |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="176" /> | assigner 实例化 | **显式传 topk=tal_topk, α=0.5, β=6.0** |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="205" :end="259" /> | __call__ | 三分量总装 |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="263" :end="444" /> | v8SegmentationLoss | 实例分割总装（v13 未动） |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="446" :end="599" /> | v8PoseLoss | 关键点总装（v13 未动） |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="601" :end="610" /> | v8ClassificationLoss | 分类交叉熵（v13 未动） |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="612" :end="725" /> | v8OBBLoss | OBB 总装（v13 未动，用 RotatedTaskAlignedAssigner） |
| <SrcLink file="v13:ultralytics/utils/loss.py" :line="728" :end="743" /> | E2EDetectLoss | 端到端双分支总装，**v13 不使用** |

<SourceViewer file="v13:ultralytics/utils/loss.py" />

## 与 v12/v11 的差异（代码风格，无行为差异）

| 位置 | v11 | v12 | **v13** |
|------|-----|-----|---------|
| 文件行数 | 745 | 743 | **743** |
| license 头 | `__copy__` 保留 | 简化 | 简化（与 v12 一致） |
| `if n := matches.sum()` 写法 | 保留旧式 | 新式 | 新式（与 v12 一致） |
| docstring 列表格式 | 旧式 | 新式 | 新式（与 v12 一致） |
| `v8DetectionLoss` 签名 | 相同 | 相同 | **相同** |
| `v8DetectionLoss.__init__` tal_topk=10 | 相同 | 相同 | **相同** |
| `v8DetectionLoss.__init__` assigner 显式覆盖 | 相同 | 相同 | **相同** |
| `E2EDetectLoss` | 相同 | 相同 | **相同** |

**结论**：**v13 loss.py 与 v11/v12 行为完全一致，无训练行为差异**。

## 三分量总装（L205-L259）

**v8DetectionLoss.__call__** 逐行拆解：

```python
def __call__(self, preds, batch, verbose=True):
    batch_idx = batch['batch_idx'].view(-1)
    ...
    # 1. TAL 分配（detach 预测后送入分配器）
    targets = self.assigner(batch["bboxes_aux"], batch["cls_aux"], batch_idx,
                           batch["batch_idx_raw"].view(-1), pred_distri, pred_scores)
    fg_mask, best_gt_idx, target_bboxes, target_scores, target_gt_scores = ...

    # 2. box loss（CIoU × q）
    loss[0] = self.bbox_loss(pred_distri, target_bboxes, target_scores, fg_mask.sum())

    # 3. cls loss（BCE × q）
    loss[1] = self.bce_loss(pred_scores, target_scores)

    # 4. dfl loss（双 bin 插值 CE × q）
    loss[2] = self.dfl_loss(pred_distri, target_bboxes, fg_mask) if self.reg_max > 1 else torch.tensor(0.0)

    loss *= self.batch_size
    return sum(loss), torch.cat((loss.sum(), loss.mean(1, keepdim=True)))
```

**三分量增益**（default.yaml L99-L101）：

| 分量 | 增益 | 计算 |
|------|-----|------|
| box | 7.5 | CIoU + DFL，用 target_scores 作回归权重 |
| cls | 0.5 | 软标签 BCE |
| dfl | 1.5 | 双 bin 插值 CE（`reg_max=16` 时） |

## TAL 分配器（v8DetectionLoss 内部）

**L176 显式覆盖**：

```python
self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
```

- `tal_topk=10` 是 loss.py L160 `__init__` 的默认值，yaml 不覆盖；
- **α=0.5、β=6.0 是硬编码常量**，与 tal.py L29 的类默认 `topk=13, α=1.0` **不一致**——v11/v12/v13 三代都有这个隐性坑；
- **训练实际值**：topk=10, α=0.5, β=6.0（v8 老参数）。

## 端到端双分支（未使用）

`yolov13.yaml` 不引用 `E2EDetectLoss`——`DetectionModel.init_criterion`（<SrcLink file="v13:ultralytics/nn/tasks.py" :line="812" :end="828" />）在 `end2end=False`（Detect 的默认）时**直接返回 `v8DetectionLoss(self)`**。

```python
def init_criterion(self):
    if getattr(self, 'end2end', False):
        return E2EDetectLoss(self)
    if self.nk == 3:
        return v8PoseLoss(self)
    if isinstance(self, (YOLOv11_SEG, ...)):  # 简写
        return v8SegmentationLoss(self)
    if isinstance(self, YOLO11_OBB):
        return v8OBBLoss(self)
    return v8DetectionLoss(self)
```

## 与 v12/v11/v8 的对应关系速查

| 维度 | v8 | v11 | v12 | **v13** |
|------|-----|-----|-----|---------|
| `v8DetectionLoss` | 引入 | 沿用 | 沿用 | **沿用** |
| tal_topk 参数位 | 无 | **首次引入** | 沿用 | **沿用** |
| TAL 类默认 | topk=10 α=0.5 | topk=13 α=1.0 | topk=13 α=1.0 | topk=13 α=1.0 |
| 训练实际 TAL 值 | topk=10 α=0.5 | topk=10 α=0.5（loss.py 覆盖） | topk=10 α=0.5 | topk=10 α=0.5 |
| 三分量增益 | 7.5/0.5/1.5 | 相同 | 相同 | **相同** |
| E2EDetectLoss | 无 | 引入 | 保留 | 保留 |
| **文件行数** | 630 | 745 | 743 | **743** |

## 想快速对上位读者的话

- **v8 读者**：把 [v8 loss.py](/versions/v8/source/loss-py) 的内容整体搬过来用即可；
- **v11/v12 读者**：v13 loss.py 与 v11/v12 完全一致；
- **v13 结构改动如何影响损失？**——**不影响**：v13 论文的超图消息传递与 FullPAD 门控都是「结构骨架层的替换」，损失侧完全沿用 v8 的三分量总装。

## 参考

- [YOLOv13 损失函数](/versions/v13/losses)
- [tal.py 源码页](/versions/v13/source/tal-py)
- [v12 loss.py](/versions/v12/source/loss-py)
- Yang W. *YOLOv13.* arXiv:2506.17733v2, 2025-11. https://arxiv.org/abs/2506.17733
