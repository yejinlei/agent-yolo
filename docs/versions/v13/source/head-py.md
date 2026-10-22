# ultralytics/nn/modules/head.py

YOLOv13 的检测头仓库：**625 行文件里，v13 没有引入任何专属头类**——`__all__`（L18）仍是 `("Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect")`，与 v8/v10/v11/v12 一字不差。v13 的实际检测头走 `Detect` 基类（L21-173），分类分支 cv3 默认走 `legacy=False` 的 depthwise-separable 路径（L46-57，与 v11/v12 完全相同）。`grep "v13|HyperACE|YOLOv13|yolov13"` 在 head.py 里零匹配——**v13 头侧代码零增量**。

> 快照路径：`source/yolov13/snapshot/ultralytics/nn/modules/head.py`（625 行，commit `73289949`，2025-11-18）。点击左侧行号可复制/分享行锚点。

**这个文件在 v13 里的角色**：**v13 论文（arXiv 2506.17733）只改结构骨干与注意力范式，检测头完全沿用 v8 Detect 的三分量输出（cv2 回归 + cv3 分类 + DFL）**。因此读者读 v13 的 head.py 与读 v12 的 head.py 得到的信息完全一致——区别仅在 Detect 输入层的通道来源变了（从 A2C2f 换成 DSC3k2 + FullPAD_Tunnel 的 Concat 结果），但 Detect 类内部不感知这个差异。

**给读者的三件事**：

1. **Detect 类的 `legacy = False` 是默认**（L32）；cv3 走 depthwise-separable 分支（L46-57，两段 DW-separable + 1×1）；
2. **v13 与 v11/v12 的 Detect 类完全一致**（L21-173 逐行相同）；
3. **Segment / Pose / OBB / Classify / WorldDetect / RTDETRDecoder / v10Detect 全部保留**——v13 的其它 yaml（`yolov13-seg.yaml`、`yolov13-pose.yaml` 等）通过继承 Detect 复用这些头。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="1" :end="18" /> | 文件头 + 导入 + `__all__` | **v13 无新增头类；`__all__` 与 v12 一字不差** |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="21" :end="74" /> | **Detect 基类** | **v13 主用**：anchor-free 检测头 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="24" :end="32" /> | Detect 类属性 | `end2end=False`、`max_det=300`、`legacy=False` |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="34" :end="45" /> | Detect.__init__ 上半 | nc/nl/reg_max/no/stride 初始化 + **cv2 回归分支** |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="46" :end="57" /> | Detect.__init__ 下半 | **cv3 分类分支：`legacy` 二选一，默认 DW-separable** |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="58" /> | DFL 挂载 | `DFL(reg_max)` if reg_max > 1 else Identity |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="60" :end="62" /> | end2end 分支 | `if self.end2end: one2one_cv2/one2one_cv3 = deepcopy(...)` |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="64" :end="74" /> | Detect.forward | 训练态 cat cv2/cv3；推理态分流到 _inference |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="76" :end="98" /> | forward_end2end | v10 双分支，v13 不使用但保留 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="100" :end="131" /> | _inference | make_anchors → split → dist2bbox 解码 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="133" :end="144" /> | bias_init | box bias=1.0、cls bias=log(5/nc/(640/s)²) |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="146" :end="148" /> | decode_bboxes | 导出 xyxy、推理 xywh |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="150" :end="172" /> | postprocess | 静态方法；end2end 无 NMS 直接选 top max_det |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="175" :end="229" /> | Segment / OBB | v13 未动 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="230" :end="306" /> | Pose / Classify | v13 未动 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="307" :end="360" /> | WorldDetect | 开放词汇检测，v13 未用 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="361" :end="590" /> | RTDETRDecoder | 端到端检测器，v13 未用 |
| <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="592" :end="624" /> | v10Detect | v10 双分支头，v13 未引用 |

<SourceViewer file="v13:ultralytics/nn/modules/head.py" />

## Detect 基类（L21-173）：v13 实际使用的检测头

### 类属性（L24-32）

```python
dynamic = False
export = False
format = None
end2end = False
max_det = 300
shape = None
anchors = torch.empty(0)
strides = torch.empty(0)
legacy = False  # backward compatibility for v3/v5/v8/v9 models
```

**与 v11/v12 完全一致**——v13 没有引入新的类属性。`legacy=False` 意味着 cv3 默认走 depthwise-separable 分支。

### __init__（L34-62）

- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="39" /> `reg_max = 16`——DFL 每边 16 个 bin；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="40" /> `no = nc + 4 × reg_max`——80 类时 144 通道；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="42" /> `c2, c3 = max((16, ch[0] // 4, reg_max*4)), max(ch[0], min(nc, 100))`——回归通道 c2 = 64、分类通道 c3 = 100（nc=80 时）；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="43" :end="45" /> **cv2 回归分支**：`Conv(x, c2, 3) → Conv(c2, c2, 3) → nn.Conv2d(c2, 4 * reg_max, 1)`——**两个全 3×3 + 1×1**，与 v8/v11/v12 一字不差；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="46" :end="57" /> **cv3 分类分支**：

```python
self.cv3 = (
    nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
    if self.legacy
    else nn.ModuleList(
        nn.Sequential(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),     # 3×3 DW + 1×1
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)), # 3×3 DW + 1×1
            nn.Conv2d(c3, self.nc, 1),                           # 1×1
        )
        for x in ch
    )
)
```

- `legacy=True`：与 v8 相同的全 3×3 分类头；
- `legacy=False`（**v13 默认**）：两段 depthwise-separable + 一个 1×1——**计算量与参数量较全 3×3 大幅下降**；
- 这个分支与 v11 Detect 完全相同（v11 没有 legacy 开关、无条件走 DW-separable），也与 v12 完全相同（v12 加了 legacy 开关、默认 False）。

- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="58" /> `self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()`；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="60" :end="62" /> `if self.end2end: one2one_cv2/one2one_cv3 = deepcopy(...)`——v13 默认 `end2end=False`，不触发。

### forward（L64-74）

```python
def forward(self, x):
    if self.end2end:
        return self.forward_end2end(x)
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    y = self._inference(x)
    return y if self.export else (y, x)
```

- 与 v11/v12 完全一致；训练态返回 x 列表（cv2/cv3 已经 cat 到 144 通道），推理态返回 `(y, x)`；
- **v13 检测训练实际走的分支**：默认 `end2end=False` → 走标准三尺度 cat 分支 → loss.py 的 `v8DetectionLoss.__call__` 吃 `(y, x)` 的 `preds[1]`。

### _inference / bias_init / decode_bboxes / postprocess（L100-172）

- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="100" :end="131" /> `_inference`：make_anchors → split → dist2bbox 解码，与 v11/v12 一字不差；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="133" :end="144" /> `bias_init`：box bias 置 1.0、分类 bias 置 `log(5/nc/(640/s)²)`——与 v8 完全一致；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="146" :end="148" /> `decode_bboxes`：`dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)`；
- <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="150" :end="172" /> `postprocess`：静态方法，`amax + topk + gather` 三步选出前 max_det 个候选——**无 NMS**，仅 end2end 分支使用；v13 默认 end2end=False，走外部 NMS。

## v10Detect（L592）：v13 保留但不使用

<SrcLink file="v13:ultralytics/nn/modules/head.py" :line="592" :end="624" /> `class v10Detect(Detect)`——v10 的双分支头仍在文件末尾。**v13 的 `yolov13.yaml` 末行是 `Detect`（L50）而非 `v10Detect`**，`guess_model_task` 遇到 `yolov13` 前缀不会走 v10 的 task_map 特判；`DetectionModel.init_criterion` 返回 `v8DetectionLoss` 而非端到端损失。

## 各头类与 v13 的关系

| 头类 | 行段 | 任务 | v13 是否使用 |
|------|------|------|-------------|
| **Detect** | <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="21" :end="173" /> | 目标检测 | **✅ yolov13.yaml 主用**（`legacy=False` DW-separable） |
| Segment | <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="175" :end="199" /> | 实例分割 | ✅ yolov13-seg.yaml 使用（继承 Detect） |
| Pose | <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="230" :end="281" /> | 关键点检测 | ✅ yolov13-pose.yaml 使用 |
| OBB | <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="200" :end="229" /> | 旋转框检测 | ✅ yolov13-obb.yaml 使用 |
| Classify | 独立文件 | 图像分类 | 与检测头无关 |
| RTDETRDecoder | <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="361" :end="590" /> | 端到端检测器 | 未接入 yolov13 |
| v10Detect | <SrcLink file="v13:ultralytics/nn/modules/head.py" :line="592" :end="624" /> | v10 双分支 | ❌ v13 不使用 |

Segment / Pose / OBB 三个变体都继承自 Detect，只覆写各自的头部分支。**它们的底座 `__init__` 与 `_inference` 全部继承 Detect L21-173**，因此 Detect 的 `legacy=False` 会**同时影响这四个任务的分类头**——v13 里 Segment/Pose/OBB 的分类头也会自动走 DW-separable。

## 与 v8 / v10 / v11 / v12 Detect 的对比

| 维度 | v8 Detect | v11 Detect | v12 Detect | **v13 Detect** | v10 v10Detect |
|------|-----------|---------------|---------------|---------------|---------------|
| 文件位置 | head.py L21-101 | head.py L21-162 | head.py L21-173 | **head.py L21-173** | head.py L592-624 |
| 通道公式 c3 | `max(ch[0], min(nc, 100))` | 一字未改 | 一字未改 | **一字未改** | 一字未改 |
| 分类头结构 | 两个全 3×3 + 1×1 | 两段 DW-separable + 1×1（无开关） | 两段 DW-separable + 1×1（legacy=False） | **两段 DW-separable + 1×1（legacy=False，同 v12）** | 两段 DW-separable |
| reg_max | 16 | 16 | 16 | **16** | 16 |
| forward 分支 | 单分支 | 单分支 | 单分支 + end2end 预留 | **单分支 + end2end 预留** | 双分支 |
| 文件行数 | 600 | 600 | 625 | **625** | — |
| v13 专属头 | — | — | — | **无** | — |

## 与 v13 骨架的连接点

v13 的 Detect 头虽然本身没变，但它吃的输入通道变了：

- **v12 Detect 输入**：来自骨干/neck 的 P3/P4/P5 三层特征（分别来自 A2C2f a2=False 的 C3k 骨架和 C3k2）；
- **v13 Detect 输入**：来自 [yolov13.yaml L50](/versions/v13/source/yolov13-yaml) 的 `[[23, 27, 31], 1, Detect, [nc]]`——分别是 head 里三个 `DSC3k2 + FullPAD_Tunnel` 组合的产物（P3-small 来自 layer 23、P4-medium 来自 layer 27、P5-large 来自 layer 31）。

**这些特征图的空间尺寸是 (64×64, 32×32, 16×16)，通道数是 (256, 512, 1024)**——`Detect.__init__` 的 `ch[0] = 256`，因此 c2=64、c3=100 与 v12 相同（因为 `ch[0] // 4 = 64`，与 v12 一致）。

> **v1~v12 读者路标**：v8/v9/v11/v12 读者读 Detect 就是读 L21-74，v13 与 v12 快照的结构**完全同源**——v13 头侧代码零增量。v10 读者请把 L592 的 v10Detect 当作「旧权重兼容层」看待，yolov13 训练时不会碰到它。

## 与其他文件的关系

- 检测头输入源：<SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="50" /> `[[23, 27, 31], 1, Detect, [nc]]`——引用 Detect 类而非 v10Detect
- 分配器：<SrcLink file="v13:ultralytics/utils/tal.py" :line="14" :end="258" /> `TaskAlignedAssigner`（v13 完全沿用 v11/v12）
- 损失总装：<SrcLink file="v13:ultralytics/utils/loss.py" :line="157" :end="259" /> `v8DetectionLoss`（v13 主用）；<SrcLink file="v13:ultralytics/utils/loss.py" :line="728" :end="743" /> `E2EDetectLoss`（v13 未使用）
- 建模入口：<SrcLink file="v13:ultralytics/nn/tasks.py" /> `DetectionModel`（继承 BaseModel，`init_criterion` 返回 v8DetectionLoss）
- DSConv（cv3 分类分支的 dw 部分来自 DWConv，非 DSConv 但结构类似）见 [conv-py 源码页](/versions/v13/source/conv-py)；骨干与注意力见 [block-py 源码页](/versions/v13/source/block-py)；结构配置见 [yolov13-yaml 源码页](/versions/v13/source/yolov13-yaml)；损失总装见 [loss-py 源码页](/versions/v13/source/loss-py)；分配器见 [tal-py 源码页](/versions/v13/source/tal-py)。
