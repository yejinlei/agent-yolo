# YOLOv5 训练技巧

## 全部 Trick 速查

| Trick | 代码位置 | 说明 |
|-------|----------|------|
| Focus | common.py L163 | 切片下采样 |
| C3 | common.py L126 | CSP Bottleneck |
| SiLU | common.py L39 | x·sigmoid(x) |
| SPP | common.py L149 | 多尺度池化 |
| PANet | yaml head | 双向聚合 |
| CIoU | general.py L343 | 坐标回归 |
| Autoanchor | autoanchor.py L23 | k-means + GA |
| 多正样本 | loss.py L185 | wh 比例 + 邻居偏移 |
| IoU ratio obj | loss.py L136 | tobj = iou |
| HSV 增强 | datasets.py L646 | hue/sat/val |
| Mosaic | datasets.py L671 | 4 图拼接 |
| random_perspective | datasets.py L851 | 仿射变换 |
| EMA | torch_utils.py L269 | 参数滑动平均 |
| AMP | train.py L301 | 混合精度 |
| OneCycle LR | train.py L139 | 学习率调度 |
| 多尺度 | train.py L293 | 随机 resize |
| 图像权重 | train.py L252 | 难例加权采样 |

## 1. Autoanchor (`trick-autoanchor`)

```python
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    m = model.module.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1. / thr).float().sum(1).mean()
        bpr = (best > 1. / thr).float().mean()
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)
    bpr, aat = metric(anchors)
    if bpr < 0.98:
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000)
        ...
```

- BPR < 0.98 时触发重算
- k-means 初始化 + 遗传算法进化 1000 代
- 完全自动化，无需人工干预

## 2. 多正样本 Anchor 匹配 (`trick-anchor-matching`)

```python
r = t[:, :, 4:6] / anchors[:, None]
j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # 4.0
```

- 与 v4 的 `iou_thresh=0.213` 不同，v5 用 **wh 比例** 而非 IoU
- 更宽松：wh 比例 < 4.0 即入选，而 v4 要求 IoU > 0.213
- 加上邻居偏移，一个 GT 最多产生 15 个正样本

## 3. IoU Ratio Objectness (`trick-iou-ratio`)

```python
tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0)
```

- `gr = 1.0` 时，objectness target = IoU
- 比 v4 的固定 1.0 更精细
- 推理时 `conf = obj_conf * cls_conf`，obj 高意味着 box 与 GT 重叠度高

## 4. HSV 色彩增强 (`trick-hsv`)

```python
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    ...
```

- 使用 OpenCV LUT 实现，比逐像素循环快 10 倍
- 默认 `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`

## 5. Mosaic 四图拼接 (`trick-mosaic`)

```python
def load_mosaic(self, index):
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
    indices = [index] + random.choices(self.indices, k=3)
    ...
```

- 与 v4 的 Mosaic 完全一致，但实现是 Python + NumPy
- 随机中心点 `yc, xc`，4 张图拼成 2s × 2s

## 6. random_perspective 仿射变换 (`trick-perspective`)

```python
def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    ...
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    M = T @ S @ R @ P
    ...
```

- 旋转 + 缩放 + 平移 + 剪切 + 透视，5 种变换组合
- 默认 `degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0`

## 7. EMA（指数移动平均）(`trick-ema`)

```python
class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        ...

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
```

- 训练时更新 ema，eval 时使用 ema 模型
- 提高模型泛化能力，稳定训练

## 8. AMP（自动混合精度）(`trick-amp`)

```python
with amp.autocast(enabled=cuda):
    pred = model(imgs)
    loss, loss_items = compute_loss(pred, targets.to(device))
    ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- `torch.cuda.amp` 自动选择 fp16/fp32
- 训练速度提升 2-3 倍，显存减半

## 9. OneCycle LR 调度 (`trick-onecycle`)

```python
if opt.cos_lr:
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
else:
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
```

- 默认 linear，可选 cosine
- `lrf = 0.2` 表示最终 lr = lr0 × 0.2

## 10. 多尺度训练 (`trick-multiscale`)

```python
if opt.multi_scale:
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
    sf = sz / max(imgs.shape[2:])
    if sf != 1:
        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
```

- 每 batch 随机 resize 到 320-1280
- 与 v2 的 `random=1` 类似，但实现更灵活

## 11. 图像权重采样 (`trick-image-weights`)

```python
if opt.image_weights:
    cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
```

- 按 mAP 加权采样：mAP 低的类别对应的图像被采样概率更高
- 解决类别不平衡

## 延伸阅读

- [理论 Concept](/versions/v5/concepts)
- [损失函数](/versions/v5/losses)
- [源码走读](/versions/v5/code-walkthrough)
