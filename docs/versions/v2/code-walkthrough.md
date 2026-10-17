# YOLOv2 源码走读

基于仓库内 darknet 快照 `source/yolov2/snapshot/`（commit `c6afc7f`，即论文"72.8% VOC mAP @ 416"的发布版），走读四个关键文件。**本页所有代码均为快照原文截取**，头部可跳转完整源码页或 GitHub 对应行。

> 快照与论文口径的差异：cfg 实为 **31 层**（22 conv + 5 maxpool + 2 route + 1 reorg + 1 region），非流传的"24 conv"；推理入口 `test_yolo()` 有 detection_layer 强转残留 bug（见 §3）。一切以快照为准。

## 文件清单

| 文件 | 行数 | 作用 | 完整源码 |
|------|------|------|----------|
| `cfg/yolo.cfg` | 244 | 网络结构 + 训练超参 + region 头 | [浏览](/versions/v2/source/yolo-cfg) |
| `src/region_layer.c` | 426 | 检测头：前向里算 loss/delta，推理解码 | [浏览](/versions/v2/source/region-layer) |
| `src/yolo.c` | 355 | 训练/推理/验证入口 | [浏览](/versions/v2/source/yolo) |
| `src/box.c` | 317 | IoU、NMS、DIoU 实验草稿 | [浏览](/versions/v2/source/box) |

---

## 1. `yolo.cfg` — 网络与训练配置

`[net]` 段：输入 416、batch=64、lr 阶梯衰减、`hue=.1`（v1 没有的色相抖动）。

<SrcSnippet file="v2:cfg/yolo.cfg" :start="1" :end="18" hl="2-5,9,12" />

- `height=416, width=416`：比 v1 的 448 小，5 次减半得 13×13（奇数，中心 cell 唯一）
- `hue=.1`：色相 ±10% 抖动，v2 新增的数据增强

backbone 每层 conv 都是 `batch_normalize=1 + leaky`（第一层示例）：

<SrcSnippet file="v2:cfg/yolo.cfg" :start="20" :end="26" hl="24-25" />

passthrough 三连：route 回连 → reorg 重排 → route 拼接（见[架构 §3.3](/versions/v2/architecture)）：

<SrcSnippet file="v2:cfg/yolo.cfg" :start="203" :end="210" hl="203,206,209" />

检测头与 region 参数（决定输出 13×13×425 与四项损失权重）：

<SrcSnippet file="v2:cfg/yolo.cfg" :start="212" :end="244" hl="213-214,221-223,228-240" />

- `anchors`：5 组 K-means 先验（对 COCO 聚类）
- `bias_match=1`：GT 分配只看 anchor 形状（见 §2.5）
- `softmax=1`：类头互斥（v1 是裸线性）
- `rescore=1`：objectness 目标取实时 IoU
- `coord_scale=5 / noobject_scale=1`：no-object 权重从 v1 的 .5 提到 **1**（anchor 多、负样本更极端）
- `random=0`：多尺度训练**不**由 cfg 触发，靠训练脚本换分辨率

---

## 2. `region_layer.c` — 前向即损失

与 v1 的 detection_layer 同一哲学：训练时 **`forward_region_layer()` 一边前向一边写 `l.delta`**，`backward_region_layer()` 只做一次 axpy（<SrcLink file="v2:src/region_layer.c" :line="322" :end="325" />）。v2 新增的是：anchor 解码、softmax 类头、sigmoid 坐标。

### 2.1 构造：输出维度 13×13×425

<SrcLink file="v2:src/region_layer.c" :line="14" /> `make_region_layer()` 里 `outputs = h·w·n·(classes + coords + 1)`（<SrcLink file="v2:src/region_layer.c" :line="28" />）。425 = 5×(80+4+1)，**每个 anchor 独立携带类别**（v1 是 cell 内 box 共享类别）。

### 2.2 解码 get_region_box：sigmoid + exp + anchor

<SrcSnippet file="v2:src/region_layer.c" :start="73" :end="85" hl="76-83" />

::: details 逐行注释（L73–L85）
- **L76**：`index + n*stride` 定位到第 n 个 anchor 的 5 维起点（stride = 85 = 4+1+80）。
- **L78-79**：`b.x = (i + σ(x[0]))/w`——σ 把偏移压进 (0,1)，加 cell 索引 i 得网格坐标，除 w=13 归一到全图。
- **L80-81**：`b.w = exp(x[2]) × biases[2n]`——e 的预测次幂 × anchor 宽；`DOABS` 时再除 w 归一（坐标系 hack，见[痛点 §6](/versions/v2/painpoints)）。
- **L83**：`b.prob` 指向 objectness 槽位，解码后由 NMS 消费。
:::

- `b.x = (i + σ(x[0]))/w`：中心 = cell 索引 + sigmoid 偏移，再归一化
- `b.w = exp(x[2])·biases[2n]`：宽 = e 的预测次幂 × anchor 先验
- `DOABS`（<SrcLink file="v2:src/region_layer.c" :line="12" />）时再除一次 grid 尺寸做归一——v2 早期坐标系 hack（见[痛点 §6](/versions/v2/painpoints)）

### 2.3 坐标 delta：sigmoid/log 链式法则

<SrcSnippet file="v2:src/region_layer.c" :start="87" :end="106" hl="94-104" />

- 中心目标 `tx = truth.x·w − i`，delta 乘 `logistic_gradient`（sigmoid 导数）
- 宽高目标 `tw = log(truth.w/anchor)`，delta 是**线性差**（log 空间无激活）
- 推导细节见[反向传播推导 §3](/versions/v2/loss-gradient)

### 2.4 类别 delta：softmax 交叉熵 + WordTree 分支

<SrcSnippet file="v2:src/region_layer.c" :start="108" :end="131" hl="111-130" />

- 普通分支：`delta = scale·(one-hot − softmax)`（交叉熵对 softmax 的梯度）
- `hier` 分支（YOLO9000）：沿 WordTree 父节点回溯写梯度（见[痛点 §7](/versions/v2/painpoints)）

### 2.5 前向主体：先激活、再写 delta

`forward_region_layer()`（<SrcLink file="v2:src/region_layer.c" :line="144" />）开头先把输入拷入 output，然后对**每个 cell × anchor** 的类头做 softmax、objectness 做 sigmoid：

<SrcSnippet file="v2:src/region_layer.c" :start="163" :end="177" hl="168-174,177" />

`if(!state.train) return`（L177）——**推理时到这里就结束**，后面全是训练逻辑。

### 2.6 noobject：先给所有 anchor 背债

<SrcSnippet file="v2:src/region_layer.c" :start="230" :end="241" hl="231-232,234-236" />

每个 anchor 先算它与所有 GT 的最大 IoU，objectness delta 先写成 `noobject·(0−conf)·σ'`；IoU 超 `thresh` 的 anchor 可被豁免（`delta=0`）。

### 2.7 12800 warmup：anchor 先学"占位"

<SrcSnippet file="v2:src/region_layer.c" :start="243" :end="254" hl="243,248-253" />

训练前 12800 张图：用 cell 中心 + anchor 尺寸当伪 GT，scale 仅 `.01`，让 anchor 先学会待在原地。

### 2.8 GT 分配：bias_match 选 anchor，再写三项 delta

核心循环 <SrcLink file="v2:src/region_layer.c" :line="258" :end="312" />。先用 **anchor 形状**选 best_n（不看预测框）：

<SrcSnippet file="v2:src/region_layer.c" :start="272" :end="292" hl="275-282,288-291" />

然后对 best_n 依次写坐标 delta（§2.3）、objectness delta（rescore 用 IoU）与类别 delta（§2.4）：

<SrcSnippet file="v2:src/region_layer.c" :start="294" :end="312" hl="296-304,306-311" />

### 2.9 cost 与 backward

<SrcLink file="v2:src/region_layer.c" :line="318" />：`cost = Σdelta²`（sum of squared deltas），就是训练日志里的 loss。backward 只做 `axpy`（<SrcLink file="v2:src/region_layer.c" :line="322" :end="325" />）。

### 2.10 推理解码 get_region_boxes

<SrcLink file="v2:src/region_layer.c" :line="327" />：把 425 维逐 anchor 解码成像素框（用 §2.2 的公式），类概率乘 objectness，低于阈值清零。**注意本快照的 `test_yolo()` 没走这条路**（见 §3 bug）。

---

## 3. `yolo.c` — 入口与那个著名的 bug

### 3.1 训练入口 train_yolo

<SrcLink file="v2:src/yolo.c" :line="15" />：解析 cfg → 可选加载预训练权重 → 循环 `load_data + network_train` → 按周期存权重。与 v1 结构相同。

### 3.2 验证 validate_yolo

<SrcLink file="v2:src/yolo.c" :line="111" />：多线程读图 → predict → `get_detection_boxes`（<SrcLink file="v2:src/yolo.c" :line="188" />）→ `do_nms_sort`（<SrcLink file="v2:src/yolo.c" :line="189" />，`iou_thresh=.5` 在 <SrcLink file="v2:src/yolo.c" :line="146" />）→ 写 VOC 结果文件。

### 3.3 test_yolo：detection_layer 强转残留 bug

<SrcSnippet file="v2:src/yolo.c" :start="285" :end="302" hl="292,299-302" />

- L292 `detection_layer l = net.layers[net.n-1];` —— **应为 region_layer**，pjreddie 忘改
- 能跑是因为两结构体头部字段布局重合；`l.side`、`get_detection_boxes` 都是 v1 残留
- 后果：本 commit 直接跑 `yolo test` 结果不可靠；论文数字来自后续修正版
- 详见[痛点 §5](/versions/v2/painpoints)

NMS 阈值 `.4`（L299），与 valid 的 `.5` 不同。

### 3.4 run_yolo 分发

<SrcLink file="v2:src/yolo.c" :line="336" :end="355" />：`test / train / valid / recall / demo` 五个子命令分发。

---

## 4. `box.c` — IoU、NMS 与 DIoU 草稿

### 4.1 IoU

<SrcLink file="v2:src/box.c" :line="94" /> `box_iou()` = inter/union，NMS 与 anchor 匹配的共同基础。

### 4.2 NMS：按类排序、重叠置零

<SrcSnippet file="v2:src/box.c" :start="249" :end="277" hl="254-256,268-271" />

::: details 逐行注释（L249–L277）
- **L254-256**：逐类别遍历：该类概率低于阈值的框先清零，再按该类概率 `qsort` 降序。
- **L258-271**：每个框与**排在自己前面**的已保留框算 IoU，超阈值则**把该类概率置零**——框不删除，它在其他类别上的分数还活着。
- **为什么按类做**：跨类别的重叠（人抱着狗）不该互相压制。
:::

- 逐类别：按该类概率 qsort → 与已保留框 IoU > thresh 则**该类概率置零**（框不删，其他类仍存活）
- 阈值由调用方传（test .4 / valid .5）

### 4.3 DIoU 实验草稿

<SrcLink file="v2:src/box.c" :line="140" /> 起一批 `dxa/dya/dwa/dha` 函数——pjreddie 在尝试把"中心距离"写进框回归（即后来的 DIoU/CIoU 思想），但**本快照的检测路径未调用**。这是 v4 CIoU 的史前草稿，见[痛点 §2](/versions/v2/painpoints)。

---

## 走读路线建议

1. **先 cfg 后代码**：`yolo.cfg` 看懂 31 层与 region 头参数 → 对照[架构](/versions/v2/architecture)
2. **训练主线**：`region_layer.c` L144 前向 → L177 train 分支 → L230 noobj → L258 GT 循环 → L318 cost，配合[反向传播推导](/versions/v2/loss-gradient)
3. **推理主线**：`region_layer.c` L327 `get_region_boxes` → `box.c` L249 NMS；**警惕** `yolo.c` L292 的强转 bug
4. **对照 v1**：[v1 源码走读](/versions/v1/code-walkthrough)，重点看 detection→region 的差分

## 延伸阅读

- [论文精读](/versions/v2/paper) / [理论 Concept](/versions/v2/concepts)
- [损失函数](/versions/v2/losses) / [反向传播推导](/versions/v2/loss-gradient)
- [训练技巧](/versions/v2/tricks) / [痛点追踪](/versions/v2/painpoints)
