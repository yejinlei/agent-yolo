# YOLOv4 源码走读

基于仓库内 darknet 快照 `source/yolov4/snapshot/`（commit `4131ae1`，2020-04-23，论文发布版），走读四个关键文件。**本页所有代码均为快照原文截取**，头部可跳转完整源码页或 GitHub 对应行。

> 快照与论文口径的差异：SAT 代码就绪但 cfg 未启用；DIoU-NMS 代码就绪但用 greedy；CIoU 长宽比梯度缺 1/(1+ar²) 因子。一切以快照为准。

## 文件清单

| 文件 | 行数 | 作用 | 完整源码 |
|------|------|------|----------|
| `cfg/yolov4.cfg` | 1155 | 162 层结构 + 训练超参 + 3 个 yolo 头 | [浏览](/versions/v4/source/yolo-cfg) |
| `src/yolo_layer.c` | 900+ | 检测头：前向里算 CIoU loss/delta，推理解码 | [浏览](/versions/v4/source/yolo-layer) |
| `src/yolo.c` | 300+ | 训练/验证/测试入口（**v4 回到 src/**） | [浏览](/versions/v4/source/yolo) |
| `src/box.c` | 700+ | IoU 家族、NMS、CIoU 梯度手推 | [浏览](/versions/v4/source/box) |

> ⚠️ v4 的目录结构与 v1/v2 相同（`yolo.c` 在 `src/`），与 v3 不同（v3 在 `examples/`）——本站所有 `v4:` 引用使用 `v4:src/yolo.c` 路径。

---

## 1. `yolov4.cfg` — 162 层结构与训练配置

`[net]` 段：输入 608、batch=64、burn_in 升温 + steps 两级衰减、Mosaic + 色彩增强：

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="8" :end="26" hl="8-9,14-16,18-19,26" />

- `burn_in=1000`：前 1000 iter 线性升温（与 v3 相同）
- `mosaic=1`：**v4 新增**——四图拼接增强
- `hue/saturation/exposure`：与 v3 相同，全部真实生效

骨干 CSP 段示例（CSP 分流 + 残块 + concat + 过渡）：

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="186" :end="207" hl="186-190,204-206" />

- `conv 3×3 s2 → 256ch`：CSP 段 3 入口下采样
- `conv 1×1 → 128ch`：CSP 分流（通道减半）
- `route layers=-2`：取回输入作为旁路

SPP 段（三尺度 maxpool + concat）：

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="763" :end="790" hl="763-766,772-775,779-782,786-789" />

三个 yolo 头之一（76×76 小目标，mask 0,1,2）：

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="967" :end="982" hl="968,974-975,978" />

- `num=9` + `mask=0,1,2`：9 个 anchor 全网络共享，本头只认领第 1/2/3 组
- `scale_x_y=1.2`：**v4 新增**——消网格敏感
- `iou_thresh=0.213`：**v4 新增**——multi-positive 阈值
- `iou_loss=ciou`：**v4 新增**——CIoU 替代 SSE
- `nms_kind=greedynms`：DIoU-NMS 代码就绪但未启用

---

## 2. `yolo_layer.c` — 前向即 CIoU 损失

与 v1/v2/v3 同一哲学：训练时 **`forward_yolo_layer()` 一边前向一边写 `l.delta`**，`backward_yolo_layer()` 只做一次 axpy（<SrcLink file="v4:src/yolo_layer.c" :line="616" :end="619" />）。v4 的新东西：CIoU 梯度、scale_x_y、multi-positive 第三遍循环。

### 2.1 构造：三头同构，255 = 3×(4+1+80)

<SrcLink file="v4:src/yolo_layer.c" :line="13" /> `make_yolo_layer(batch, w, h, n, total, mask, classes)`——`n=3`（本头 anchors）、`total=9`（全网）、`mask` 从 cfg 读入。

### 2.2 前置激活 + scale_x_y

<SrcSnippet file="v4:src/yolo_layer.c" :start="334" :end="340" hl="336-337" />

::: details 逐行注释（L334–L340）
- **L336**：对 tx/ty 通道原地做 logistic——与 v3 相同。
- **L337**：**v4 新增**——`scal_add_cpu(2*l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y-1), ...)`，即 `y = scale_x_y·σ(tx) − 0.5·(scale_x_y−1)`，把范围从 (0,1) 扩到 (-0.1, 1.1)。
- **L339**：对 objectness + 80 类通道原地做 logistic（与 v3 相同）。
:::

### 2.3 CIoU 梯度：delta_yolo_box 的分支

<SrcSnippet file="v4:src/yolo_layer.c" :start="183" :end="236" hl="187,196-203,206-209" />

::: details 逐行注释（L183–L236）
- **L187**：`dx_box_iou(pred, truth, iou_loss)` —— 调 box.c 的 CIoU 梯度计算。
- **L196-199**：Jacobian 转置——`dx = dt, dy = db, dw = dl, dh = dr`。
- **L202-203**：`dw *= exp(x[...])` —— tw/th 是 log 空间，梯度乘 e^tw 才能作用到原始 w/h。
- **L206-209**：`dx *= iou_normalizer` —— 乘归一化系数 0.07。
- **L233-236**：累加 delta——multi-positive 的多个 anchor 会叠加。
:::

### 2.4 三遍循环：noobj + best_n + multi-positive

第一遍（<SrcLink file="v4:src/yolo_layer.c" :line="369" :end="410" />）：全场 noobj，IoU>0.7 豁免。

第二遍（<SrcLink file="v4:src/yolo_layer.c" :line="447" :end="490" />）：每个 GT 找 best_n，mask 命中才监督。

第三遍（<SrcLink file="v4:src/yolo_layer.c" :line="501" :end="545" />）：同头其余 anchor IoU>0.213 也监督。

<SrcSnippet file="v4:src/yolo_layer.c" :start="501" :end="520" hl="504-517" />

::: details 逐行注释（L501–L520）
- **L504**：`mask_n >= 0 && n != best_n` —— 只处理"在本头 mask 内、但不是 best_n"的 anchor。
- **L508-511**：形状 IoU > 0.213 才参与监督。
- **L517**：调 `delta_yolo_box` 写 delta（与 best_n 相同的逻辑）。
:::

delta 平均（<SrcLink file="v4:src/yolo_layer.c" :line="556" />）：所有正样本的 delta 经 `averages_yolo_deltas` 平均。

---

## 3. `yolo.c` — 入口（回到 src/）

v4 把训练/验证/测试入口从 v3 的 `examples/` 挪回 `src/`：`train_yolo`（<SrcLink file="v4:src/yolo.c" :line="5" />）、`validate_yolo`、 `test_yolo`（<SrcLink file="v4:src/yolo.c" :line="285" />）、`run_yolo`。

**test_yolo 的祖传 bug 已修复**：v4 的 `test_yolo` 不再用 `l.side`（v2/v3 的 bug），改用 `get_network_boxes` 返回的 `nboxes`。

---

## 4. `box.c` — CIoU 梯度手推

核心函数：`box_iou`（<SrcLink file="v4:src/box.c" :line="163" />）、`do_nms_sort`、`diounms_sort`（DIoU-NMS，代码就绪）、`dx_box_iou`（<SrcLink file="v4:src/box.c" :line="258" :end="476" />，CIoU 梯度全手推）。

**CIoU 梯度实现**（<SrcLink file="v4:src/box.c" :line="430" :end="450" />）：

```c
if (iou_loss == CIOU) {
    float ar_gt = truth.w / truth.h;
    float ar_pred = pred.w / pred.h;
    float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
    float alpha = ar_loss / (1 - I/U + ar_loss + 0.000001);
    float ar_dw = 8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.h;
    float ar_dh = -8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.w;
    // ... DIoU 项 + CIoU 长宽比项 ...
}
```

**注意**：`ar_dw/ar_dh` 缺 1/(1+ar²) 因子（见 [痛点 §7](/versions/v4/painpoints#_7-快照事实-ciou-长宽比梯度缺-1-1-ar2-因子-pain-ciou-gradient-missing-factor)）。

---

## 阅读路线建议

1. 先看 [Architecture 逐层图解](/versions/v4/architecture)建立 162 层地图
2. `cfg` 的 [yolo] 段（<SrcLink file="v4:cfg/yolov4.cfg" :line="967" :end="982" />）→ `yolo_layer.c` 的 forward（<SrcLink file="v4:src/yolo_layer.c" :line="327" />）→ 三遍循环（<SrcLink file="v4:src/yolo_layer.c" :line="369" :end="545" />）
3. 对照 [losses](/versions/v4/losses) / [loss-gradient](/versions/v4/loss-gradient) 逐行验证 CIoU 公式
4. 最后读 [painpoints](/versions/v4/painpoints) 复核三个快照级发现

## 延伸阅读

- [源码页：yolo_layer.c](/versions/v4/source/yolo-layer) / [yolo.c](/versions/v4/source/yolo) / [box.c](/versions/v4/source/box) / [yolov4.cfg](/versions/v4/source/yolo-cfg)
- [痛点追踪](/versions/v4/painpoints)
- [v3 源码走读](/versions/v3/code-walkthrough)
