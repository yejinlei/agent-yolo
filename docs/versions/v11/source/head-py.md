# ultralytics/nn/modules/head.py

YOLOv11 的检测头仓库：**597 行文件里，v11 实际使用的是 L21-68 的 `Detect` 基类**——anchor-free 网格中心回归、[DFL](/glossary#dfl-distribution-focal-loss) 分布表示、sigmoid 分类分，与 v8 的 Detect 一字不差；v10 增量的 `v10Detect` 仍在文件末尾（L564）但 **`yolo11.yaml` 不引用它**，属于「历史保留但不使用」。

> 快照路径：`source/yolov11/snapshot/ultralytics/nn/modules/head.py`（597 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v11 里的角色**：v11 对 head.py 的净增量为零——L21-68 的 `Detect` 与 v8.0.100、v9、v10 三个前代版本完全同源；v10Detect 在 L564 存在但无调用点。因此 v11 读者读这一页的目的**不是找新东西**，而是确认三件事：

1. v11 的 Detect 仍是 v8 的老 Detect，通道公式 `c2, c3 = max((16, ch[0] // 4, reg_max*4)), max(ch[0], min(nc, 100))` 一字未改；
2. v10Detect 在 L564 保留（对历史权重兼容），但 `yolo11.yaml` 末行写的是 `Detect` 而非 `v10Detect`；
3. Segment / Pose / OBB / WorldDetect / RTDETRDetect 五个任务头全部原样保留，v11 论文不涉及它们的改动。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="21" :end="68" /> | **Detect 基类** | **v11 主用**：anchor-free 检测头，与 v8 完全一致 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="30" :end="43" /> | Detect.__init__ | 通道公式 c2/c3；cv2 回归 + cv3 分类 + DFL |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="45" :end="71" /> | Detect.inference | make_anchors → split → dist2bbox 解码 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="73" :end="77" /> | forward_feat | 训练态逐尺度 cat cv2/cv3 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="79" :end="86" /> | Detect.forward | 训练/推理两态切换 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="88" :end="95" /> | Detect.bias_init | box bias=1.0、cls bias=log(5/nc/(640/s)²) |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="97" :end="101" /> | decode_bboxes | 导出 xyxy、推理 xywh |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="104" :end="262" /> | Segment 头 | 实例分割，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="265" :end="388" /> | Pose 头 | 关键点，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="391" :end="452" /> | OBB 头 | 旋转框检测，v11 未动 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="455" :end="561" /> | WorldDetect / RTDETRDetect | 世界模型 / 端到端检测器头 |
| <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="564" :end="602" /> | **v10Detect** | v10 专属双分支头，**v11 不使用但保留** |

<SourceViewer file="v11:ultralytics/nn/modules/head.py" />

## Detect 基类（L21-68）：v11 实际使用的检测头

### 类属性与 __init__（L21-43）

- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="24" :end="28" /> 类属性 `dynamic / export / shape / anchors / strides` 挂在类上做缓存——网格锚点只在输入尺寸变化时重算，与 v8 相同。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="35" /> `reg_max = 16`——[DFL](/glossary#dfl-distribution-focal-loss) 每边 16 个 bin，v11 与 v8/v9/v10 一字不差。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="36" /> `no = nc + 4 × reg_max`——每 anchor 输出通道数（80 类时 80+64=144），公式与 v8 相同。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="38" /> `c2, c3 = max((16, ch[0] // 4, reg_max*4)), max(ch[0], min(nc, 100))`——回归通道 c2 = 64（reg_max*4）、分类通道 c3 = 100（nc=80 时）；**v10 的「轻量分类头」不改这个公式**，只改后面 cv3 的卷积结构；v11 更简单，公式与 cv3 结构都不改。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="39" :end="41" /> 回归分支 `cv2 = nn.ModuleList(Conv(x, c2, 3) → Conv(c2, c2, 3) → nn.Identity() for x in ch)`——两层全 3×3 卷积 + Identity，输出 4×reg_max 通道给 DFL。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="42" /> 分类分支 `cv3 = nn.ModuleList(Conv(x, c3, 3) → Conv(c3, c3, 3) → nn.Conv2d(c3, nc, 1) for x in ch)`——**两层全 3×3 + 一个 1×1 头**，与 v8 一字不差，v10 的 depthwise-separable 轻量化在 v11 里不存在。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="43" /> `self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()`——DFL 模块挂在类属性上，推理时把 4×reg_max 的分布回归成 4 个框坐标。

### inference（L45-71）：解码一条龙

<SrcLink file="v11:ultralytics/nn/modules/head.py" :line="45" :end="71" /> 推理态的完整解码：

- L48 `torch.cat([xi.view(sh[0], no, -1) for xi, sh in zip(x, self.shape)], 2).permute(0, 2, 1)`——三尺度特征拉平后按 anchor 维拼接，得到 `(b, H*W_total, 144)`；
- L50 `self.anchors, self.strides = (a.repeat(shape[0], 1), s.repeat(shape[0])) if self.anchors.shape[0] == 1 else ...`——网格锚点（`make_anchors` 生成）与 stride 张量挂在类属性缓存；
- L57 `x.split([self.reg_max * 4, self.nc], 1)`——按通道切成 `box` 与 `cls` 两份；
- L68 `boxes = self.decode_bboxes(box, self.anchors, self.strides)`——`decode_bboxes`（L97-101）走 `dist2bbox`（tal.py）把 4×reg_max 分布回归成 4 坐标；
- L70 `torch.cat((boxes, cls.sigmoid()), 1)`——回归结果与 sigmoid 分类分拼接，输出 `(b, 8400, 4+nc)`；
- L53-66 那几段 `self.export and self.format in (...)` 是给 TF/TFLite/MNN 等后端兜底的兼容分支，读的时候可以跳过。

### forward_feat / forward / bias_init（L73-101）

- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="73" :end="77" /> `forward_feat`：训练态逐尺度 `torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1)`——这是给损失总装 v8DetectionLoss 的输入形态。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="79" :end="86" /> `forward`：`if not self.training and not self.export: return self.inference(x)`；训练态直接返回 `self.forward_feat(x)`。v11 的 Detect **没有 v10 的 `forward` 覆写**，全部逻辑在基类里。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="88" :end="95" /> `bias_init`：box bias 置 1.0、分类 bias 置 `log(5 / nc / (640/s)²)`——每格先验约 0.01 个目标，v11 与 v8 完全一致。
- <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="97" :end="101" /> `decode_bboxes`：导出走 `xywh=False`（xyxy 直接进 NMS）、推理走 `xywh=True`（先 wh 后 xywh2xyxy），v11 完全一致。

## v10Detect（L564）：v11 保留但不使用

<SrcLink file="v11:ultralytics/nn/modules/head.py" :line="564" :end="602" /> `class v10Detect(Detect)`——v10 的双分支头仍在文件末尾：

- **max_det = -1**（L566 附近）：哨兵默认值，导出时由 `engine/exporter.py` 注入 `args.max_det`（默认 300）；
- **cv3 重写**（L570-573）：把分类分支两个全 3×3 换成两段 depthwise-separable（`Conv(x, x, 3, g=x) + Conv(x, c3, 1)` → `Conv(c3, c3, 3, g=c3) + Conv(c3, c3, 1)` → `Conv2d(c3, nc, 1)`）——v10 论文的「轻量分类头」；
- **one2one deepcopy**（L575-576）：`one2one_cv2/one2one_cv3 = copy.deepcopy(self.cv2/self.cv3)`；
- **forward 双分支**（L578-592）：`one2one = self.forward_feat([xi.detach() for xi in x], ...)`——detach 掐断梯度到骨干；`one2many = super().forward(x)` 走 v8 路径；导出态走 `ops.v10postprocess` 无 NMS 直接选 max_det。

**v11 与 v10Detect 的关系**：v11 的 `yolo11.yaml` 末行是 `Detect` 而非 `v10Detect`；`guess_model_task` 遇到 `yolo11` 前缀不会走 v10 的 task_map 特判；`DetectionModel.init_criterion` 返回 `v8DetectionLoss` 而非 `v10DetectLoss`。**v10Detect 在 v11 快照里纯粹是「保留以便加载 v10 权重」的历史代码**，新训练 yolo11 完全不会实例化它。

> **v1~v10 读者路标**：v8 读者读 Detect 就是读 L21-68，v11 与 v8 完全一致，一行注释都没改。v9 读者注意 v9 的 head.py 也是同样布局（`Detect` + `Segment` + `Pose` + `OBB`），没有 v10Detect；v9 的 C3k 分流（`if c2 > 128: return self.attn(...)`）在 head.py 里也没有——那个分流写在 `block.py` 的 C3k 类里，v11 的 C3k2 干脆去掉。v10 读者请把 L564 的 v10Detect 当作「旧权重兼容层」看待，yolo11 训练时不会碰到它。

## 各 Detect 变体与 v11 的关系

| 头类 | 行段 | 任务 | v11 是否使用 |
|------|------|------|-------------|
| **Detect** | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="21" :end="68" /> | 目标检测 | **✅ yolo11.yaml 主用** |
| Segment | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="104" :end="262" /> | 实例分割 | ✅ yolo11-seg.yaml 使用（继承 Detect） |
| Pose | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="265" :end="388" /> | 关键点检测 | ✅ yolo11-pose.yaml 使用（继承 Detect） |
| OBB | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="391" :end="452" /> | 旋转框检测 | ✅ yolo11-obb.yaml 使用（继承 Detect） |
| Classify | 独立文件 | 图像分类 | 与检测头无关 |
| WorldDetect | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="455" :end="510" /> | 世界模型检测 | 未接入 yolo11 |
| RTDETRDetect | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="513" :end="561" /> | 端到端检测器 | 未接入 yolo11 |
| v10Detect | <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="564" :end="602" /> | v10 双分支 | ❌ **v11 不使用（仅加载 v10 权重时命中）** |

Segment / Pose / OBB 三个变体都继承自 Detect，各自只覆写：Segment 加 proto 掩码分支 + loss 计算；Pose 加 kpt 关键点头；OBB 把回归通道换成旋转框 5 维参数。**它们的底座 `__init__` 与 `inference` 全部继承 Detect L21-68**，因此改动 Detect 会同时影响这四个任务——v11 没动 Detect，所以这四个任务头在 v11 里与 v8 一字不差。

## 与 v8 / v10 Detect 的对比

| 维度 | v8 Detect | **v11 Detect** | v10 v10Detect |
|------|-----------|---------------|---------------|
| 文件位置 | head.py L21-101 | **head.py L21-68** | head.py L564-602 |
| 通道公式 c3 | `max(ch[0], min(nc, 100))` | **一字未改** | 一字未改 |
| 分类头结构 | 两个全 3×3 + 1×1 | **一字未改** | 两段 depthwise-separable |
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
