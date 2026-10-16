# YOLOv1 源码走读（独立版）

> 基于 `source/yolov1/snapshot/` 快照，逐行走读 4 个关键文件。
> 站点版见 [docs/versions/v1/code-walkthrough.md](../../docs/versions/v1/code-walkthrough.md)，本版补充更多代码细节。

---

## 1. `cfg/yolo.cfg` — 模型与训练配置

### 1.1 网络超参数

```ini
[net]
batch=1
subdivisions=1
height=448
width=448
channels=3
momentum=0.9
decay=0.0005
saturation=1.5
exposure=1.5
hue=.1
```

- `height=448, width=448`：输入尺寸固定，因 FC 层不可变
- `saturation=1.5, exposure=1.5, hue=.1`：HSV 数据增强参数，训练时随机扰动

### 1.2 学习率策略

```ini
learning_rate=0.0005
policy=steps
steps=200,400,600,20000,30000
scales=2.5,2,2,.1,.1
max_batches = 40000
```

5 步调度的含义：

| Step | Batch | Scale | 效果 | 目的 |
|------|-------|-------|------|------|
| 1 | 200 | ×2.5 | 0.0005→0.00125 | warmup |
| 2 | 400 | ×2 | →0.0025 | warmup |
| 3 | 600 | ×2 | →0.005 | warmup |
| 4 | 20000 | ×0.1 | →0.0005 | decay |
| 5 | 30000 | ×0.1 | →0.00005 | final decay |

前 3 步是 warmup（防止早期梯度爆炸），后 2 步是 decay（收敛阶段降速）。

### 1.3 卷积层模板

```cfg
[convolutional]
batch_normalize=1
filters=64
size=7
stride=2
pad=1
activation=leaky
```

每一层四要素：
- `filters`：输出通道数
- `size`：卷积核尺寸
- `stride`：步长
- `activation`：激活函数（leaky = LeakyReLU, slope=0.1）

`batch_normalize=1` 是 v1 引入 BN 的标志——在每层 Conv 后、激活前加 BN。

### 1.4 检测头配置

```cfg
[detection]
classes=20
coords=4
rescore=1
side=7
num=2
softmax=0
sqrt=1
jitter=.2

object_scale=1
noobject_scale=.5
class_scale=1
coord_scale=5
```

参数速查表：

| 参数 | 值 | 含义 | 对应本体实体 |
|------|-----|------|-------------|
| `side` | 7 | Grid S=7 | Model `yolo-v1` |
| `num` | 2 | 每格 B=2 个 box | Model `yolo-v1` |
| `classes` | 20 | VOC 20 类 | Model `yolo-v1` |
| `coords` | 4 | x, y, w, h | Loss `coord-l2` |
| `rescore` | 1 | conf delta 用 IoU 修正 | Loss `rescore` |
| `sqrt` | 1 | w/h 开方回归 | Trick `sqrt-one` |
| `softmax` | 0 | 类别用 sigmoid 不用 softmax | — |
| `jitter` | 0.2 | 20% 平移缩放增强 | Trick `region-jitter` |
| `coord_scale` | 5 | 坐标 loss 权重 | Trick `loss-weight-balance` |
| `object_scale` | 1 | 有目标 conf 权重 | — |
| `noobject_scale` | 0.5 | 无目标 conf 权重 | — |
| `class_scale` | 1 | 类别 loss 权重 | — |

> **注意**：仓库 `source/_repos/` 中的 `cfg/yolov1.cfg` 已被 v2+ 覆盖为 `num=3, dropout=0.5`。快照保留了 v1 原始值。

---

## 2. `src/detection_layer.c` — 检测头

这是 YOLOv1 最核心的文件——**同时负责前向、loss 计算、GT 分配**。

### 2.1 数据结构

```c
// detection_layer 结构（简化）
typedef struct {
    int classes;    // 20
    int coords;     // 4
    int rescore;    // 1
    int side;       // 7
    int n;          // 2 (B)
    int sqrt;       // 1
    float jitter;   // 0.2

    float object_scale;     // 1
    float noobject_scale;   // 0.5
    float class_scale;      // 1
    float coord_scale;      // 5

    int count;  // 训练迭代计数器
    // ...
} detection_layer;
```

### 2.2 前向核心：GT 分配与 loss 计算

```c
// forward_detection_cpu 核心逻辑（简化）
for (b = 0; b < batch; ++b) {
    for (n = 0; n < l.n * l.side * l.side; ++n) {
        float best_iou = 0;
        int best_index = 0;

        // 遍历所有 GT，找 IoU 最大的 box
        for (i = 0; i < truths; ++i) {
            box truth = float_to_box(truths + i * truth_size, l.coords, 1, 0);
            box pred = float_to_box(net_output + n * l.outputs, l.coords, 1, 1);
            float iou = box_iou(pred, truth);
            if (iou > best_iou) {
                best_index = n;
                best_iou = iou;
            }
        }

        // ★ Trick: 前 100k 迭代随机 best_index
        if (l.count < 100000) best_index = rand() % l.n;

        // 计算 coord delta
        if (l.sqrt) {
            box.w = pow(boxes[i].w, 0.5);
            box.h = pow(boxes[i].h, 0.5);
        }
        delta[0] = l.coord_scale * (truth.x - pred.x);
        delta[1] = l.coord_scale * (truth.y - pred.y);
        delta[2] = l.coord_scale * (truth.w - pred.w);
        delta[3] = l.coord_scale * (truth.h - pred.h);

        // 计算 confidence delta
        // ★ rescore=1: conf 目标 = IoU（不是 1）
        if (l.rescore) {
            delta[4] = l.object_scale * (best_iou - pred.conf);
        } else {
            delta[4] = l.object_scale * (1 - pred.conf);
        }

        // 计算 class delta
        for (c = 0; c < l.classes; ++c) {
            float val = (c == truth_class) ? 1 : 0;
            delta[5 + c] = l.class_scale * (val - pred.prob[c]);
        }

        // 无目标格降权
        if (n != best_index) {
            delta[4] = l.noobject_scale * (0 - pred.conf);
            // class delta = 0（不计算类别 loss）
        }
    }
    l.count++;
}
```

### 2.3 代码走读要点

| 行号区域 | 功能 | 对应本体实体 |
|---------|------|-------------|
| 73-75 | 随机 best_index warmup | Trick `random-best-index` |
| 82-83 | conf 目标 = IoU (rescore) | Loss `rescore-objectness` |
| 85-88 | conf delta 修正 | Loss `object-l2` |
| 94-98 | class delta (one-hot) | Loss `class-l2` |
| 76-77 | 无目标格降权 | Loss `noobject-l2` |
| 101-105 | coord delta + sqrt | Loss `coord-l2`, Trick `sqrt-one` |

### 2.4 推理时 NMS 调用

```c
detection *detection_layer_forward(...) {
    // 收集所有 conf > thresh 的 box
    for (i = 0; i < l.n * l.side * l.side; ++i) {
        detections[i].bbox = get_region_box(...);
        detections[i].prob = ...;
    }
    // 交给 do_nms_sort 做后处理
    do_nms_sort(detections, total, l.classes, 0, 0.5);
    return detections;
}
```

推理时把 98 个 box 收集起来，交给 `do_nms_sort`。

---

## 3. `src/yolo.c` — 端到端推理入口

### 3.1 核心函数

```c
detection *detect_objects(image im, layer l, int w, int h, int net, float thresh) {
    // Step 1: 缩放到 448×448
    image sized = resize_image(im, w, h);

    // Step 2: 加灰色边框（论文未提的冷门 trick）
    image crop = crop_image(sized, -(sized.w - w) / 2, -(sized.h - h) / 2, w, h);
    box border = border_image(crop, 4, 127, 0);  // 128 灰色边框

    // Step 3: 前向推理
    predictions = network_predict(net, border);

    // Step 4: 从网络输出提取检测框
    detection *res = detections_from_network(l, predictions);
    return res;
}
```

### 3.2 三步走详解

**Step 1 — resize_image**：

```c
image resize_image(image im, int w, int h) {
    // 双线性插值缩放到 448×448
    image resized = make_image(w, h, im.c);
    for (k ...) for (r ...) for (c ...) {
        float val = bilinear_interpolate(im, x, y, k);
        set_pixel(resized, c, r, k, val);
    }
    return resized;
}
```

**Step 2 — border_image**（论文未提的 trick）：

```c
image border_image(image im, int border, float r, float g, float b) {
    // 在图像四周加 border 像素的灰色边框
    image bordered = make_image(im.w + 2*border, im.h + 2*border, im.c);
    // 填充灰色 (127)
    for (k ...) for (r ...) for (c ...) {
        set_pixel(bordered, c, r, k, r/255.0f);
    }
    // 将原图贴到中间
    embed_image(im, bordered, border, border);
    return bordered;
}
```

**为什么加 border**：v1 的 7×7 grid 在图像边缘会被截断。灰色边框给网络一个稳定的边缘参考，防止边缘物体被裁切。这是论文没有提及但源码中存在的 trick。

**Step 3 — network_predict**：

标准的前向传播，从输入层到 detection 层。输出 1470 维向量。

**Step 4 — detections_from_network**：

```c
detection *detections_from_network(layer l, float *predictions) {
    // reshape 1470 → 7×7×30
    // 对每个 cell:
    //   对每个 box (B=2):
    //     if conf > thresh:
    //       extract (x, y, w, h, conf, class_probs)
    //       add to detections[]
    // 调用 do_nms_sort 去重
    return detections;
}
```

---

## 4. `src/box.c` — IoU 和 NMS

### 4.1 `box_iou` — 核心 IoU 计算

```c
float box_iou(box a, box b) {
    // 转为中心点 + 宽高 → 角点坐标
    float a_x1 = a.x - a.w / 2;
    float a_y1 = a.y - a.h / 2;
    float a_x2 = a.x + a.w / 2;
    float a_y2 = a.y + a.h / 2;

    float b_x1 = b.x - b.w / 2;
    float b_y1 = b.y - b.h / 2;
    float b_x2 = b.x + b.w / 2;
    float b_y2 = b.y + b.h / 2;

    // 交集
    float inter_x1 = max(a_x1, b_x1);
    float inter_y1 = max(a_y1, b_y1);
    float inter_x2 = min(a_x2, b_x2);
    float inter_y2 = min(a_y2, b_y2);

    float inter_w = max(0.f, inter_x2 - inter_x1);
    float inter_h = max(0.f, inter_y2 - inter_y1);
    float inter = inter_w * inter_h;

    // 并集
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - inter;

    return inter / union_area;
}
```

**标准 IoU**：交集面积 / 并集面积。

**v1 只用了 IoU**，没有 CIoU/GIoU/DIoU。这是 v1 定位粗糙的直接原因：
- IoU 对 box 的方向不敏感（两个 box 方向不同但面积相同，IoU 可能相同）
- IoU 无法度量"距离但无重叠"的情况（IoU=0 时梯度为 0）

### 4.2 `do_nms_sort` — 非极大值抑制

```c
void do_nms_sort(detection *dets, int total, int classes, float overlap) {
    int i, j, k;

    // Step 1: 对每个检测框按 confidence 降序排序
    // 使用 sort_detections，按 prob[k] * object 降序
    sort_detections(dets, total);

    // Step 2: 贪心 NMS
    for (i = 0; i < total; ++i) {
        if (dets[i].sorted) continue;     // 已被删除
        if (!dets[i].prob[0]) continue;   // 置信度为 0

        for (j = i + 1; j < total; ++j) {
            if (dets[j].sorted) continue;

            for (k = 0; k < classes; ++k) {
                // 两个 box 都检测到类别 k
                if (dets[j].prob[k] > 0 && dets[i].prob[k] > 0) {
                    float iou = box_iou(dets[i].bbox, dets[j].bbox);

                    // IoU 超过阈值 → 删除低分框
                    if (iou > overlap) {
                        dets[j].sorted = 1;   // 标记删除
                    }
                }
            }
        }
    }
}
```

**核心逻辑**：
1. 按 `confidence` 降序排序
2. 对每个未删除的 box，与所有排在其后的 box 计算 IoU
3. IoU > `overlap`（默认 0.5）则标记删除

**性能分析**：

| 版本 | 候选框数 | NMS 计算量 |
|------|---------|-----------|
| v1 | 98 | 98²/2 ≈ 4,800 |
| v3 (3 scale) | 10,647 | ~56M |
| v8 (3 scale) | 8,400 | ~35M |

v1 的 NMS 开销可忽略，但 v3+ 成为瓶颈。v10 的 NMS-free 检测头彻底解决了这个问题。

### 4.3 NMS 的后续演进

| 版本 | NMS 方案 | 变化 |
|------|---------|------|
| v1 | 硬阈值 0.5 | O(n²) |
| v3 | 硬阈值 + per-class | 按类别独立做 |
| v4 | SOFT-NMS（可选） | 不硬删，软降权 |
| v8 | DIoU-NMS（可选） | 用 DIoU 替代 IoU |
| v10 | NMS-free | 检测头直接输出唯一 box |

---

## 5. 完整调用链

```
darknet detector test cfg/yolo.cfg backup/yolo.weights data/dog.jpg
  │
  ├── yolo.c: detect_objects()
  │     ├── image.c: resize_image()         → 448×448
  │     ├── image.c: border_image()         → 加灰色边框
  │     ├── network.c: network_predict()    → 前向传播
  │     │     ├── convolutional_layer.c     → 24 层 Conv
  │     │     ├── connected_layer.c         → 2 层 FC
  │     │     └── detection_layer.c         → Detection 层
  │     │           ├── reshape 1470 → 7×7×30
  │     │           └── 提取 98 个 box
  │     └── box.c: do_nms_sort()            → NMS 去重
  │           └── box.c: box_iou()          → IoU 计算
  │
  └── 输出检测结果
```

```
训练时:
darknet detector train cfg/yolo.cfg backup/yolo.weights
  │
  ├── parser.c: parse_network_cfg()         → 解析 cfg
  ├── data.c: load_data()                   → 加载 VOC
  │     └── data.c: random_distort()       → jitter 增强
  ├── network.c: train_network()            → 训练循环
  │     └── detection_layer.c: forward_detection_cpu()
  │           ├── GT 分配 (best_index)
  │           ├── 随机 best_index (前 100k)
  │           ├── coord delta + sqrt
  │           ├── conf delta + rescore
  │           ├── class delta
  │           └── noobject delta
  │     └── detection_layer.c: backward_detection_cpu()
  └── 保存权重
```

---

## 6. 源码中的"隐藏" trick

论文没提但源码中存在的 trick：

| Trick | 源码位置 | 作用 |
|-------|---------|------|
| 随机 best_index | `detection_layer.c:73-75` | 前 100k 防梯度塌缩 |
| border_image | `yolo.c: detect_objects()` | 加灰色边框防边缘裁切 |
| rescore | `detection_layer.c:85-88` | conf delta 用 IoU 修正 |
| sqrt 权重 | `detection_layer.c:101-105` | w/h 开方均衡量级 |

这些 trick 在论文中被忽略，但在源码中起到了关键作用。随机 best_index 尤其有意思——它暴露了"多 box 结构不稳定"的本质，v2 引入 anchor 后彻底废弃。

---

## 相关资源

- [VitePress 源码走读页](../../docs/versions/v1/code-walkthrough.md)：站点版速查
- [架构与层](../../docs/versions/v1/layers.md)：每层 cfg 详解
- [损失函数](../../docs/versions/v1/losses.md)：detection_layer.c loss 公式
- [训练技巧](../../docs/versions/v1/tricks.md)：7 个 trick 的源码定位
- [痛点追踪](../../docs/versions/v1/painpoints.md)：源码暴露的实现局限
- [本体论 SourceFile 实例](../../ontology/instances/yolov1/sources.yaml)
