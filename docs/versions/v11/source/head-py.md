# ultralytics/nn/modules/head.py

YOLOv11 的检测头仓库：**597 行文件里，v11 实际使用的是 L21-162 的 `Detect` 基类**——anchor-free 网格中心回归、[DFL](/glossary#dfl-distribution-focal-loss) 分布表示、sigmoid 分类分；外壳与 v8 的 Detect 一致，但 `cv3` 分类分支沿用 v10 的 DW-separable（无 legacy 开关）；v10 增量的 `v10Detect` 仍在文件末尾（L564）但 **`yolo11.yaml` 不引用它**，属于「历史保留但不使用」。

> 快照路径：`source/yolov11/snapshot/ultralytics/nn/modules/head.py`（597 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v11 里的角色**：v11 对 head.py 的净增量为零——`Detect`（L21-162）的外壳（cv2 / dfl / forward / _inference）与 v8.0.100、v9、v10 同源，但 `cv3` 是 v10 式 DW-separable；v10Detect 在 L564 存在但无调用点。因此 v11 读者读这一页的目的**不是找新东西**，而是确认三件事：

1. v11 的 Detect 仍是 v8 的老 Detect，通道公式 `c2, c3 = max((16, ch[0] // 4, reg_max*4)), max(ch[0], min(nc, 100))` 一字未改；
2. v10Detect 在 L564 保留（对历史权重兼容），但 `yolo11.yaml` 末行写的是 `Detect` 而非 `v10Detect`；
3. Segment / Pose / OBB / WorldDetect / RTDETRDetect 五个任务头全部原样保留，v11 论文不涉及它们的改动。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="21" :end="162" /> | **Detect 基类** | **v11 主用**：外壳同 v8，cv3 为 DW-separable |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="32" :end="56" /> | Detect.__init__ | 通道公式 c2/c3；cv2 回归 + cv3 分类 + DFL |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="58" :end="68" /> | Detect.forward | 训练/推理两态切换 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="70" :end="90" /> | forward_end2end | v10 双分支路径（v11 yaml 不触发） |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="94" :end="120" /> | Detect._inference | make_anchors → split → dist2bbox 解码 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="122" :end="133" /> | Detect.bias_init | box bias=1.0、cls bias=log(5/nc/(640/s)²) |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="135" :end="137" /> | decode_bboxes | 导出 xyxy、推理 xywh |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="139" :end="162" /> | postprocess | conf 过滤 + class-wise NMS |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="164" :end="187" /> | Segment 头 | 实例分割，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="189" :end="217" /> | OBB 头 | 旋转框检测，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="219" :end="257" /> | Pose 头 | 关键点，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="259" :end="277" /> | Classify 头 | 图像分类，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="279" :end="331" /> | WorldDetect 头 | 开放词汇检测 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="333" :end="562" /> | RTDETRDecoder | 端到端检测器 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="564" :end="597" /> | **v10Detect** | v10 专属双分支头，**v11 不使用但保留** |

<SourceViewer file="v11:ultralytics/nn/modules/head.py" />

## Detect 基类（L21-162）：v11 实际使用的检测头

### 类属性与 __init__（L24-52）

- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="24" :end="30" /> 类属性 `dynamic / export / end2end / max_det / shape / anchors / strides` 挂在类上做缓存——网格锚点只在输入尺寸变化时重算，与 v8 相同。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="37" /> `reg_max = 16`——[DFL](/glossary#dfl-distribution-focal-loss) 每边 16 个 bin，v11 与 v8/v9/v10 一字不差。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="38" /> `no = nc + 4 × reg_max`——每 anchor 输出通道数（80 类时 80+64=144），公式与 v8 相同。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="40" /> `c2, c3 = max((16, ch[0] // 4, reg_max*4)), max(ch[0], min(nc, 100))`——回归通道 c2 = 64（reg_max*4）、分类通道 c3 = 100（nc=80 时）；**v10 的「轻量分类头」不改这个公式**，只改后面 cv3 的卷积结构；v11 的通道公式不变，cv3 内部卷积结构沿用 v10 的 DW-separable。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="41" :end="43" /> 回归分支 `cv2 = nn.ModuleList(Conv(x, c2, 3) → Conv(c2, c2, 3) → nn.Conv2d(c2, 4*reg_max, 1) for x in ch)`——两层全 3×3 卷积 + 1×1 输出 4×reg_max 通道给 DFL。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="44" :end="51" /> 分类分支 `cv3` 是**两段 depthwise-separable + 一个 1×1 头**（`DWConv 3×3 → 1×1` 重复两次，再 `Conv2d 1×1`）——沿用 v10 的轻量分类头思路；v11 **没有 `legacy` 开关**，无条件走这条分支（v12 才补上 `legacy` 类属性）。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="52" /> `self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()`——DFL 模块挂在类属性上，推理时把 4×reg_max 的分布回归成 4 个框坐标。

### inference（L94-120）：解码一条龙

<SrcLink file="v11:ultralytics/nn/modules/head.py" :line="94" :end="120" /> 推理态的完整解码：

- L97-98 `shape = x[0].shape`；`x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)`——三尺度特征拉平后按 anchor 维拼接，得到 `(b, H*W_total, 144)`；
- L99-101 `make_anchors(x, self.stride, 0.5)`——网格锚点与 stride 张量挂在类属性缓存，输入尺寸不变就不重算；
- L103-107 export 后端兜底：TF 系后端用切片代替 `split`（避免 FlexSplitV 算子），其余走 `x_cat.split((reg_max*4, nc), 1)`；
- L109-118 `self.dfl(box)` 做 4×16 分布期望积分，再 `decode_bboxes(..., anchors) * strides` 还原到像素尺度；tflite/edgetpu 分支额外乘归一化因子；
- L120 `torch.cat((dbox, cls.sigmoid()), 1)`——回归结果与 sigmoid 分类分拼接，输出 `(b, 8400, 4+nc)`。

### forward / bias_init / decode_bboxes（L58-137）

- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="58" :end="68" /> `forward`：L60-61 `end2end` 时转 `forward_end2end`；L63-64 训练态逐尺度 `torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1)`；L65-66 训练直接返回 list；L67-68 推理走 `_inference` 并按 `export` 决定返回形态。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="70" :end="90" /> `forward_end2end`：`one2one = [xi.detach() for xi in x]` 掐断梯度后过 `one2one_cv2/cv3`，`one2many` 走正常 cv2/cv3；训练返回 `{"one2many": x, "one2one": one2one}` 字典。**v11 的 yaml 不触发此分支**。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="122" :end="133" /> `bias_init`：box bias 置 1.0、分类 bias 置 `log(5 / nc / (640/s)²)`——每格先验约 0.01 个目标，v11 与 v8 完全一致；`end2end` 时对 one2one 头也做同样初始化。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="135" :end="137" /> `decode_bboxes`：`dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)`——v11 检测（end2end=False）走 xywh 再转，v10Detect 走 xyxy。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="139" :end="162" /> `postprocess`（staticmethod）：`non_max_suppression` 按类 NMS 后取 top `max_det`——v11 检测唯一的后处理入口。

## v10Detect（L564）：v11 保留但不使用

<SrcLink file="v11:ultralytics/nn/modules/head.py" :line="564" :end="597" /> `class v10Detect(Detect)`——v10 的双分支头仍在文件末尾：

- **end2end = True**（L582）：类属性覆写，基类 `forward` 据此转到 `forward_end2end`；
- **cv3 重写**（L587-596）：把分类分支换成两段 depthwise-separable（`Conv(x, x, 3, g=x) + Conv(x, c3, 1)` → `Conv(c3, c3, 3, g=c3) + Conv(c3, c3, 1)` → `Conv2d(c3, nc, 1)`）——v10 论文的「轻量分类头」；
- **one2one 头 deepcopy**（L597 附近）：`one2one_cv2/one2one_cv3 = copy.deepcopy(...)`；
- **forward 双分支**：直接复用基类 <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="70" :end="90" /> `forward_end2end`——`one2one` 走 `xi.detach()` 掐断梯度、`one2many` 走正常路径；导出态走 `ops.v10postprocess` 无 NMS 直接选 max_det。

**v11 与 v10Detect 的关系**：v11 的 `yolo11.yaml` 末行是 `Detect` 而非 `v10Detect`；`guess_model_task` 遇到 `yolo11` 前缀不会走 v10 的 task_map 特判；`DetectionModel.init_criterion`（tasks.py L386-388）是三元表达式，`yolo11.yaml` 的 Detect 没有 `end2end`，实际返回 `v8DetectionLoss(self)`。**v10Detect 在 v11 快照里纯粹是「保留以便加载 v10 权重」的历史代码**，新训练 yolo11 完全不会实例化它。

> **v1~v10 读者路标**：v8 读者读 Detect 就是读 L21-162，v11 的外壳（cv2 / dfl / forward / _inference）与 v8 一致，但 **cv3 换成 v10 式 DW-separable（无 legacy 开关）**；v9 读者注意 v9 的 head.py 也是同样布局（`Detect` + `Segment` + `Pose` + `OBB`），没有 v10Detect；v9 的 C3k 分流（`if c2 > 128: return self.attn(...)`）在 head.py 里也没有——那个分流写在 `block.py` 的 C3k 类里，v11 的 C3k2 干脆去掉。v10 读者请把 L564 的 v10Detect 当作「旧权重兼容层」看待，yolo11 训练时不会碰到它。

## 各 Detect 变体与 v11 的关系

| 头类 | 行段 | 任务 | v11 是否使用 |
|------|------|------|-------------|
| **Detect** | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="21" :end="162" /> | 目标检测 | **✅ yolo11.yaml 主用** |
| Segment | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="164" :end="187" /> | 实例分割 | ✅ yolo11-seg.yaml 使用（继承 Detect） |
| OBB | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="189" :end="217" /> | 旋转框检测 | ✅ yolo11-obb.yaml 使用（继承 Detect） |
| Pose | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="219" :end="257" /> | 关键点检测 | ✅ yolo11-pose.yaml 使用（继承 Detect） |
| Classify | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="259" :end="277" /> | 图像分类 | ✅ yolo11-cls.yaml 使用 |
| WorldDetect | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="279" :end="331" /> | 世界模型检测 | 未接入 yolo11 |
| RTDETRDecoder | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="333" :end="562" /> | 端到端检测器 | 未接入 yolo11 |
| v10Detect | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="564" :end="597" /> | v10 双分支 | ❌ **v11 不使用（仅加载 v10 权重时命中）** |

Segment / Pose / OBB 三个变体都继承自 Detect，各自只覆写：Segment 加 proto 掩码分支 + loss 计算；Pose 加 kpt 关键点头；OBB 把回归通道换成旋转框 5 维参数。**它们的底座 `__init__` 与 `_inference` 全部继承 Detect L21-162**，因此改动 Detect 会同时影响这四个任务——v11 的 Detect 外壳沿用 v8，但 cv3 内部卷积沿用 v10 的 DW-separable，所以这四个任务头的分类分支也随之走 DW-separable。

## 与 v8 / v10 Detect 的对比

| 维度 | v8 Detect | **v11 Detect** | v10 v10Detect |
|------|-----------|---------------|---------------|
| 文件位置 | head.py L21-101 | **head.py L21-162** | head.py L564-597 |
| 通道公式 c3 | `max(ch[0], min(nc, 100))` | **一字未改** | 一字未改 |
| 分类头结构 | 两个全 3×3 + 1×1 | **两段 depthwise-separable（无 legacy 开关）** | 两段 depthwise-separable |
| reg_max | 16 | **16** | 16 |
| forward 分支 | 单分支 | **单分支** | 双分支（one2one + one2many） |
| detach 梯度 | 无 | **无** | one2one 输入 detach |
| 导出后处理 | 图外 NMS | **图外 NMS** | 图内 v10postprocess（无 NMS） |
| bias_init | box=1.0 / cls=log(5/nc/(640/s)²) | **一字未改** | 基类 + one2one 各一遍 |

## 与其他文件的关系

- 分配器：<SrcLink file="v11:ultralytics/utils/tal.py" :line="13" /> `TaskAlignedAssigner`（v11 默认 topk=13）
- 损失总装：<SrcLink file="v11:ultralytics/utils/loss.py" :line="157" /> `v8DetectionLoss`（v11 主用）；<SrcLink file="v11:ultralytics/utils/loss.py" :line="730" /> `E2EDetectLoss`（v10 专用，v11 不使用）
- 建模入口：<SrcLink file="v11:ultralytics/nn/tasks.py" :line="298" /> `DetectionModel`（继承 BaseModel，`init_criterion` 返回 v8DetectionLoss）
- 结构描述：<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="47" /> 末行 `[[16, 19, 22], 1, Detect, [nc]]`——引用的是 `Detect` 类，**不是 v10Detect**

逐行注释见本页上方；训练链路与推理链路的全景见 [code-walkthrough](/versions/v11/code-walkthrough)；结构配置见 [yolo11.yaml 源码页](/versions/v11/source/yolo11-yaml)；损失总装见 [loss.py 源码页](/versions/v11/source/loss-py)；分配器见 [tal.py 源码页](/versions/v11/source/tal-py)。
