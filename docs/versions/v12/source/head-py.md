# ultralytics/nn/modules/head.py

YOLOv12 的检测头仓库：**600 行左右的文件里，v12 实际使用的是 L21 起的 `Detect` 基类**——anchor-free 网格中心回归、[DFL](/glossary#dfl-distribution-focal-loss) 分布表示、sigmoid 分类分；与 v8 的 Detect 同源，**默认走 `legacy=False` 的 depthwise-separable 分类头**——v11 的 cv3 本来就是 DW-separable，v12 只是补了 `legacy` 开关。v10 增量的 `v10Detect` 仍在文件末尾（L592）但 **`yolov12.yaml` 不引用它**，属于「历史保留但不使用」。

> 快照路径：`source/yolov12/snapshot/ultralytics/nn/modules/head.py`（625 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v12 里的角色**：v12 对 head.py 的净增量只有**一个类属性**——L32 新增 `legacy = False`，让 L46-57 的 cv3 走 depthwise-separable 分支。**v11 的 Detect 没有 `legacy` 属性，但 cv3 本来就是 DW-separable**（无开关、无条件走同一分支）——所以两代默认行为相同，v12 只是把「不可切」变成「可切但默认仍是 DW」。

具体地，v12 读者读这一页的目的是确认三件事：

1. **Detect 类的 `legacy = False` 是默认**（L32）；cv3 走 depthwise-separable（L46-57，与 v11 相同），与 v10 的 v10Detect 分类头思路相同；
2. **Detect 新增 `end2end` 属性**（L27 类属性 + L60-62 __init__ 双分支 deepcopy + L64-68 forward 分流）——但 v12 的 yaml 未触发它，属于「类级预留」；
3. **`_inference` 方法化**（L100）：v11 里叫 `inference`，v12 拆成 `_inference` + `postprocess`（L150）两个静态方法，语义与 v11 完全一致；
4. Segment / Pose / OBB / Classify / RTDETR 五个任务头保留，v12 论文不涉及它们的改动。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="21" :end="74" /> | **Detect 基类** | **v12 主用**：anchor-free 检测头，`legacy=False` 默认 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="24" :end="32" /> | Detect 类属性 | `legacy = False`（L32）——**v12 新增的属性，v11 没有** |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="34" :end="45" /> | Detect.__init__ 上半 | nc/nl/reg_max/no/stride 初始化 + **cv2 回归分支** |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="46" :end="57" /> | Detect.__init__ 下半 | **cv3 分类分支：`legacy` 二选一** |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="60" :end="62" /> | end2end 分支 | `if self.end2end: one2one_cv2/one2one_cv3 = deepcopy(...)` |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="64" :end="74" /> | Detect.forward | 训练态 cat cv2/cv3；推理态分流到 _inference |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="76" :end="98" /> | forward_end2end | v10 双分支，v12 不使用但保留 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="100" :end="131" /> | _inference | make_anchors → split → dist2bbox 解码 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="133" :end="144" /> | bias_init | box bias=1.0、cls bias=log(5/nc/(640/s)²) |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="146" :end="148" /> | decode_bboxes | 导出 xyxy、推理 xywh |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="150" :end="172" /> | postprocess | 静态方法；end2end 无 NMS 直接选 top max_det |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="175" :end="199" /> | Segment 头 | 实例分割，v12 未动 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="200" :end="229" /> | OBB 头 | 旋转框检测，v12 未动 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="230" :end="281" /> | Pose 头 | 关键点，v12 未动 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="282" :end="306" /> | Classify 头 | 图像分类，v12 未动 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="307" :end="360" /> | WorldDetect 头 | 开放词汇检测，v12 未用 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="361" :end="590" /> | RTDETRDecoder | 端到端检测器，v12 未用 |
| <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="592" :end="624" /> | v10Detect | v10 双分支头，v12 未引用 |

<SourceViewer file="v12:ultralytics/nn/modules/head.py" />

## Detect 基类（L21-173）：v12 实际使用的检测头

### 类属性（L24-32）

```python
dynamic = False     # force grid reconstruction
export = False      # export mode
format = None       # export format
end2end = False     # end2end                     # v12 新增
max_det = 300       # max_det                     # v12 新增（v10 是 -1 哨兵）
shape = None
anchors = torch.empty(0)
strides = torch.empty(0)
legacy = False      # backward compatibility for v3/v5/v8/v9 models   # L32
```

**v12 新增两个类属性 `end2end` 与 `max_det`**——这两个原本是 v10 v10Detect 的实例属性，v12 把它们提到基类 Detect 上做「类级预留」：

- `end2end=False` 是默认，v12 yaml 训练不会触发；
- `max_det=300` 是默认（v10 v10Detect 是 `-1`，导出时由 `engine/exporter.py` 注入）——**v12 把 max_det 的默认值固化到 300**，导出时无需再走 sentinel 注入流程。

**L32 `legacy = False`** 是 v12 相对 v11 唯一的 head.py 增量——注释写的是「backward compatibility for v3/v5/v8/v9 models」，说明这个属性是「老模型兼容开关」，`legacy=True` 时才走全 3×3 分类头（v3/v5/v8/v9 时代的路径）；v12 默认 `legacy=False` 意味着**新训的 v12 模型走 DW-separable 分类头**——与 v11 的默认行为一致（v11 无开关，无条件 DW）。

### __init__（L34-62）

- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="39" /> `reg_max = 16`——[DFL](/glossary#dfl-distribution-focal-loss) 每边 16 个 bin，v12 与 v8/v9/v10/v11 一字不差。
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="40" /> `no = nc + 4 × reg_max`——每 anchor 输出通道数（80 类时 80+64=144），公式与 v8 相同。
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="42" /> `c2, c3 = max((16, ch[0] // 4, reg_max*4)), max(ch[0], min(nc, 100))`——回归通道 c2 = 64、分类通道 c3 = 100（nc=80 时）；公式与 v11 一字不差。
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="43" :end="45" /> 回归分支 `cv2 = nn.ModuleList(Conv(x, c2, 3) → Conv(c2, c2, 3) → nn.Conv2d(c2, 4 * self.reg_max, 1) for x in ch)`——**与 v11 完全相同**（v11 L40-42 同样是 `nn.Conv2d` 收尾）。
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="46" :end="57" /> 分类分支 cv3，`legacy` 二选一：

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

- `legacy=True`：与 v8 相同的全 3×3 分类头（v11 没有这个开关）；
- `legacy=False`（**v12 默认**）：两段 depthwise-separable + 一个 1×1——**计算量与参数量较全 3×3 大幅下降**，与 v10 v10Detect 的分类头思路相同，但 v12 把这个思路作为默认分支下沉到基类。

- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="58" /> `self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()`——DFL 模块挂在类属性上。
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="60" :end="62" /> `if self.end2end: one2one_cv2/one2one_cv3 = deepcopy(...)`——v10 双分支的骨架，v12 默认 `end2end=False` 不会走到。

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

- 与 v11 相比：v11 的 `forward` 直接 `if not self.training and not self.export: return self.inference(x)`；v12 把「训练态返回 cat 结果 / 推理态返回 (y, x)」拆得更明确，并**支持 `end2end` 分流**（但 v12 默认不启用）。
- 训练态返回值是 `x` 列表（cv2/cv3 已经 cat 到 144 通道），推理态返回值是 `(y, x)` 元组——这个约定与 v11 相同，`v8DetectionLoss.__call__`（[loss.py L209](/versions/v12/source/loss-py)）就是 `feats = preds[1] if isinstance(preds, tuple) else preds` 来吃这个。

### _inference（L100-131）

- L104 `torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)`——三尺度特征拉平后按 anchor 维拼接，得到 `(b, H*W_total, 144)`；
- L105-107 网格锚点（`make_anchors` 生成）与 stride 张量挂在类属性缓存；
- L113 `box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)`——按通道切成 `box` 与 `cls` 两份；
- L128-129 `dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides`——DFL softmax 期望成框 + 锚点加 stride 缩放；
- L131 `torch.cat((dbox, cls.sigmoid()), 1)`——输出 `(b, 8400, 4+nc)`；
- L115-127 那几段 `self.export and self.format in (...)` 是给 TF/TFLite/IMX 等后端兜底的兼容分支，读的时候可以跳过。

### bias_init / decode_bboxes / postprocess（L133-172）

- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="133" :end="144" /> `bias_init`：box bias 置 1.0、分类 bias 置 `log(5 / nc / (640/s)²)`——每格先验约 0.01 个目标，与 v8/v11 完全一致；L141-144 额外给 `end2end` 分支的 one2one 头也做一遍 bias_init；
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="146" :end="148" /> `decode_bboxes`：`dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)`——导出走 xyxy、推理走 xywh；
- <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="150" :end="172" /> `postprocess`：静态方法，`amax + topk + gather` 三步选出前 max_det 个候选——**无 NMS**，这是 v10 的 end2end 路径专用；v12 默认 `end2end=False`，训练与标准推理都走基类的 NMS 后处理路径。

## v10Detect（L592）：v12 保留但不使用

<SrcLink file="v12:ultralytics/nn/modules/head.py" :line="592" :end="624" /> `class v10Detect(Detect)`——v10 的双分支头仍在文件末尾。**v12 已经通过基类的 `end2end` 属性把 v10Detect 的核心逻辑吸收进来**，所以 v10Detect 在 v12 里更像「旧权重兼容层」。

**v12 与 v10Detect 的关系**：v12 的 `yolov12.yaml` 末行是 `Detect` 而非 `v10Detect`；`guess_model_task` 遇到 `yolov12` 前缀不会走 v10 的 task_map 特判；`DetectionModel.init_criterion`（L386-388）返回三元表达式的结果——`yolov12.yaml` 的 Detect 没有 `end2end`，实际拿到 `v8DetectionLoss` 而非端到端损失。**v10Detect 在 v12 快照里纯粹是「保留以便加载 v10 权重」的历史代码**，新训练 yolo12 完全不会实例化它。

> **v1~v11 读者路标**：v8/v9/v11 读者读 Detect 就是读 L21-74，v12 与 v11 快照的**结构完全同源，唯一差别是 `legacy=False` 是默认**——这意味着 v12 新训的模型 cv3 走 DW-separable 分类头；v10 读者请把 L592 的 v10Detect 当作「旧权重兼容层」看待，yolo12 训练时不会碰到它；v12 里 v10Detect 的功能大部分已经吸收到基类 Detect 的 `end2end` 属性上。

## 各 Detect 变体与 v12 的关系

| 头类 | 行段 | 任务 | v12 是否使用 |
|------|------|------|-------------|
| **Detect** | <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="21" :end="173" /> | 目标检测 | **✅ yolov12.yaml 主用**（`legacy=False` DW-separable） |
| Segment | <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="175" :end="199" /> | 实例分割 | ✅ yolov12-seg.yaml 使用（继承 Detect） |
| Pose | <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="230" :end="281" /> | 关键点检测 | ✅ yolov12-pose.yaml 使用（继承 Detect） |
| OBB | <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="200" :end="229" /> | 旋转框检测 | ✅ yolov12-obb.yaml 使用（继承 Detect） |
| Classify | 独立文件 | 图像分类 | 与检测头无关 |
| RTDETR | <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="361" :end="590" /> | 端到端检测器 | 未接入 yolov12 |
| v10Detect | <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="592" :end="624" /> | v10 双分支 | ❌ **v12 不使用（仅加载 v10 权重时命中）** |

Segment / Pose / OBB 三个变体都继承自 Detect，各自只覆写：Segment 加 proto 掩码分支 + loss 计算；Pose 加 kpt 关键点头；OBB 把回归通道换成旋转框 5 维参数。**它们的底座 `__init__` 与 `_inference` 全部继承 Detect L21-173**，因此 Detect 的 `legacy=False` 会**同时影响这四个任务的分类头**——v12 里 Segment/Pose/OBB 的分类头也会自动走 DW-separable。

## 与 v8 / v10 / v11 Detect 的对比

| 维度 | v8 Detect | **v11 Detect** | **v12 Detect** | v10 v10Detect |
|------|-----------|---------------|---------------|---------------|
| 文件位置 | head.py L21-101 | **head.py L21-162** | **head.py L21-173** | head.py L592-624 |
| 通道公式 c3 | `max(ch[0], min(nc, 100))` | 一字未改 | 一字未改 | 一字未改 |
| 分类头结构 | 两个全 3×3 + 1×1 | **两段 DW-separable + 1×1（无开关）** | **两段 DW-separable + 1×1（legacy=False 默认）** | 两段 DW-separable |
| reg_max | 16 | 16 | 16 | 16 |
| forward 分支 | 单分支 | 单分支 | 单分支 + end2end 预留 | 双分支（one2one + one2many） |
| detach 梯度 | 无 | 无 | 无（end2end 分支下 detach） | one2one 输入 detach |
| 导出后处理 | 图外 NMS | 图外 NMS | 图外 NMS（end2end 分支下图内 topk） | 图内 v10postprocess（无 NMS） |
| bias_init | box=1.0 / cls=log(5/nc/(640/s)²) | 一字未改 | 一字未改（+ end2end 分支） | 基类 + one2one 各一遍 |
| max_det 默认 | — | — | **300** | -1（sentinel） |
| legacy 默认 | — | **无此属性（恒 DW-separable）** | **False** | — |

## 与其他文件的关系

- 分配器：<SrcLink file="v12:ultralytics/utils/tal.py" :line="14" /> `TaskAlignedAssigner`（v12 默认 topk=13 α=1.0 β=6.0）
- 损失总装：<SrcLink file="v12:ultralytics/utils/loss.py" :line="157" /> `v8DetectionLoss`（v12 主用）；<SrcLink file="v12:ultralytics/utils/loss.py" :line="728" :end="743" /> `E2EDetectLoss`（v10 专用，v12 不使用）
- 建模入口：<SrcLink file="v12:ultralytics/nn/tasks.py" :line="298" /> `DetectionModel`（继承 BaseModel，`init_criterion` 返回 v8DetectionLoss）
- 结构描述：<SrcLink file="v12:ultralytics/cfg/models/v12/yolov12.yaml" :line="45" /> 末行 `[[14, 17, 20], 1, Detect, [nc]]`——引用的是 `Detect` 类，**不是 v10Detect**

逐行注释见本页上方；训练链路与推理链路的全景见 [code-walkthrough](/versions/v12/code-walkthrough)；结构配置见 [yolov12.yaml 源码页](/versions/v12/source/yolov12-yaml)；A2C2f / ABlock / AAttn 三件套见 [block.py 源码页](/versions/v12/source/block-py)；损失总装见 [loss.py 源码页](/versions/v12/source/loss-py)；分配器见 [tal.py 源码页](/versions/v12/source/tal-py)。
