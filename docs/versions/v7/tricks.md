# YOLOv7 训练技巧

## 全部 Trick 速查

| Trick | 代码位置 | 说明 | 推理是否零开销 |
|-------|----------|------|----------------|
| E-ELAN 扩展高效层聚合 | yolov7.yaml L21-28 / L82-89 | expand/shuffle/merge 的 cardinality | 是（纯结构） |
| compound scaling 缩放配方 | cfg/training/yolov7-{w6,e6,d6,e6e}.yaml | computational block 缩 depth、transition 缩 width | 是（纯结构） |
| planned re-param 选位 | common.py L463-582 + yolov7.yaml L135-137 | RepConv/RepConvN 规则，只插 head 末端 | 训练期三分支，推理折叠零开销 |
| lead-head-guided 标签分配 | loss.py L1176-1697 | 双头共用 lead 头引导的分配 | 是（纯训练期） |
| coarse-to-fine 双尺度正样本 | loss.py L1593-1697 | fine（g=0.5）给 lead、coarse（g=1.0）给 aux | 是（纯训练期） |
| partial auxiliary head | yolo.py L311-430 | aux 纯 1×1 头只接部分层 | 是（推理丢 aux） |
| conv-BN 融合 | common.py L560-581 | 训练分开、推理折叠 | 是（推理免费） |
| YOLOR 隐式知识并入卷积 | yolo.py L178-190 | ImplicitA/ImplicitM 并进 bias/weight | 是（推理免费） |
| EMA 平均教师 | train.py L200 | 最终推理用 EMA 权重 | 是（训练期开销） |
| AMP 混合精度 | train.py L299 / L360 | GradScaler + autocast | 是（训练期开销） |
| 超参文件分档 | data/hyp.scratch.{p5,p6,tiny,custom}.yaml | 按模型档位选 hyp，含 `loss_ota` 总开关 | 是（纯配置） |
| 从零训练只用 COCO | — | 不依赖额外数据 | 是（数据侧约束） |
| label smoothing 与可选 Focal | loss.py L432-438 | 默认全关，按需开启 | 是（纯训练期） |

## 1. E-ELAN 扩展高效层聚合 (`trick-eelan`)

**动机**：深网络一味堆层会遇到梯度路径退化。论文 §3.1 的 E-ELAN 不改变梯度路径本身，只用 expand/shuffle/merge 的 cardinality（配合 group conv）增加梯度传播路径的多样性，让超深网络仍然训得动。

**实现**：基础版 yolov7.yaml 的 ELAN 块是"1×1 分叉 → 4×3×3 堆叠 → 多深度 Concat → 1×1 transition"（<SrcLink file="v7:cfg/training/yolov7.yaml" :line="20" :end="28" />）：

```yaml
[-1, 1, Conv, [64, 1, 1]],        # 分叉 1
[-2, 1, Conv, [64, 1, 1]],        # 分叉 2（跨层取输入）
[-1, 1, Conv, [64, 3, 1]],        # 4 个 3×3 逐级堆叠
...
[[-1, -3, -5, -6], 1, Concat, [1]],   # 多深度合并
[-1, 1, Conv, [256, 1, 1]],       # transition 层
```

- head 里的 ELAN 块合并路数更多（六路 Concat，<SrcLink file="v7:cfg/training/yolov7.yaml" :line="82" :end="89" />）
- 扩展模型（`cfg/training/yolov7-{w6,e6,e6e,d6}.yaml`）的 E-ELAN 把分叉加宽、ELAN 段加深（如 e6e 每段 6 个 3×3、两段之间再接 `Shortcut`），下采样换 `DownC`，对应论文给 W6/E6/D6/E6E 家族的结构基座

**效果**：论文 §5.1 的 W6/E6/D6/E6E 全部建立在 E-ELAN 之上，最大号 E6E 达到 56.8 AP。

**推理开销**：零开销。E-ELAN 是纯结构设计，训练与推理形态一致，没有训练期临时分支；成本只体现为模型本身更大。

## 2. compound scaling 缩放配方 (`trick-compound-scaling`)

**动机**：concat 型模型（ELAN 家族）不能照搬 CSP 系"depth、width 独立缩放"的做法。论文 §3.2 的配方：**computational block 只缩 depth（堆叠数），transition layer 按与 computational block 相同的变化量缩 width**。

**实现**：
- 快照里 `cfg/training/*.yaml` 全部是 `depth_multiple: 1.0` / `width_multiple: 1.0`（<SrcLink file="v7:cfg/training/yolov7.yaml" :line="2" :end="4" />）——缩放直接写进结构：base 版每段 ELAN 堆 4 个 3×3、transition 通道 256/512/1024；W6/E6/D6/E6E 加深 ELAN 段并成倍加宽 transition（e6e 用 160/320/640…）
- 缩放钩子仍保留在 `parse_model`：depth 走 `n = max(round(n * gd), 1)`、width 走 `c2 = make_divisible(c2 * gw, 8)`（<SrcLink file="v7:models/yolo.py" :line="751" :end="765" />），但官方配置不再依赖倍率

**效果**：论文 Table 3 消融 base 51.7 → compound 52.9 AP（+1.2）；同一配方推演出 W6（54.9）→ E6（56.0）→ D6（56.6）→ E6E（56.8）的完整谱系。

**推理开销**：零开销。纯结构缩放，训练推理同形态。

## 3. planned re-param 与 RepConvN 选位规则 (`trick-planned-reparam`)

**动机**：RepVGG 式三分支重参数化不能随手乱插。论文 §4.1 的"planned re-parameterized convolution"给出选位规则：**带 residual / concat 连接的卷积层不能有 identity 分支**，否则破坏 ResNet 残差与 DenseNet 拼接的梯度多样性（论文 Fig.4）——遵守该规则的 RepConv 即论文命名的 RepConvN。

**实现**（<SrcLink file="v7:models/common.py" :line="463" :end="582" />）：快照里没有 `RepConvN` 类，规则用条件 identity 实现（<SrcLink file="v7:models/common.py" :line="486" />）：

```python
self.rbr_identity = (nn.BatchNorm2d(num_features=c2) if c2 == c1 and s == 1 else None)
self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False), nn.BatchNorm2d(num_features=c2))
self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False), nn.BatchNorm2d(num_features=c2))
```

- c2≠c1 或 s≠1 时 `rbr_identity` 自动为 `None`，三分支退化为两分支，等价 RepConvN；forward 见 <SrcLink file="v7:models/common.py" :line="498" :end="507" />
- **选位**：yolov7.yaml 只在 head 末端、六路 Concat 聚合之后插三处 RepConv，且全部 c2≠c1（128→256 / 256→512 / 512→1024）→ 自动无 identity（<SrcLink file="v7:cfg/training/yolov7.yaml" :line="135" :end="137" />）：

```yaml
[75, 1, RepConv, [256, 3, 1]],
[88, 1, RepConv, [512, 3, 1]],
[101, 1, RepConv, [1024, 3, 1]],
```

- 骨干/颈部的残差（Shortcut）与拼接（Concat）位置不放 RepConv，正是 RepConvN 规则的体现

**效果**：论文 Table 4/5 的消融支持"planned"选位收益；三处 RepConv 部署时折叠成普通 3×3。

**推理开销**：训练期三分支多付算力/显存，推理期零开销——`get_equivalent_kernel_bias` + `_fuse_bn_tensor`（<SrcLink file="v7:models/common.py" :line="509" :end="552" />）折叠为单 3×3 卷积。

## 4. lead-head-guided 标签分配 (`trick-lead-guided`)

**动机**：lead 头与 auxiliary 头各算各的标签会互相矛盾。论文 §4.2：用 **lead head 的预测 + GT** 生成一套标签，两个 head 共用（不再独立分配）。

**实现**（<SrcLink file="v7:utils/loss.py" :line="1202" :end="1218" />）：`ComputeLossAuxOTA.__call__` 里两套 targets 都拿 **lead 头的预测** `p[:self.nl]` 参与 cost 匹配生成：

```python
bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets, imgs)
bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets, imgs)
...
pi = p[i]              # lead 头预测
pi_aux = p[i+self.nl]  # aux 头预测
```

- `build_targets`（lead，<SrcLink file="v7:utils/loss.py" :line="1287" :end="1289" />）与 `build_targets2`（aux，<SrcLink file="v7:utils/loss.py" :line="1440" :end="1442" />）共用同一 cost 匹配流程，aux 的分配由 lead 头预测引导——这正是 "lead head guided label assigner" 的代码落点
- 软标签仍是 v3/v5 式 `tobj = (1-gr) + gr·iou`，lead 与 aux 各按自己预测的 IoU 生成（aux 侧 <SrcLink file="v7:utils/loss.py" :line="1262" />）

**效果**：双头目标一致、不再互相拉扯；配合 partial aux 头，消融里逐步涨点（base 55.6 → 56.8，见 [trick 6](#_6-partial-auxiliary-head-trick-partial-aux)）。

**推理开销**：零开销。标签分配只发生在训练期。

## 5. coarse-to-fine 双尺度正样本 (`trick-coarse-to-fine`)

**动机**：论文 §4.2 的 coarse-for-auxiliary & fine-for-lead：约束严的 fine label 给 lead head 精修定位，放宽的 coarse label 给 auxiliary head 保 recall，让 aux 头学得更全、lead 头学得更准。

**实现**：差异全在候选格偏置 `g` 上（两者都是 5 邻域候选，对角偏移被注释停用）：

- fine（lead）：`find_3_positive`，`g = 0.5`，候选格收紧（<SrcLink file="v7:utils/loss.py" :line="1646" :end="1658" />）

```python
g = 0.5  # bias
off = torch.tensor([[0, 0],
                    [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                    ], device=targets.device).float() * g  # offsets
```

- coarse（aux）：`find_5_positive`，`g = 1.0`，同样 5 邻域但偏置更宽（<SrcLink file="v7:utils/loss.py" :line="1592" :end="1604" />）

```python
g = 1.0  # bias
off = torch.tensor([[0, 0],
                    [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                    ], device=targets.device).float() * g  # offsets
```

- 两个函数后续都做同样的 anchor 宽高比匹配（`hyp['anchor_t']`）+ cost 匹配，只在候选格宽严上分叉

**效果**：论文消融 Table 6-8：base 55.6 → coarse-to-fine 55.9 → +partial aux 56.3 → 56.8。lead 定位更精、aux recall 更高。

**推理开销**：零开销。纯训练期的分配策略。

## 6. partial auxiliary head (`trick-partial-aux`)

**动机**：辅助头全接太贵。论文 §4.2 的 partial auxiliary head 只在网络部分层引出 aux 分支，且 aux 分支**不含隐式知识**（YOLOR 的 ImplicitA/ImplicitM 只给 lead），进一步省训练成本。

**实现**（<SrcLink file="v7:models/yolo.py" :line="311" :end="430" />）：
- lead 头 `m` 带 `ia`/`im` 隐式知识（<SrcLink file="v7:models/yolo.py" :line="328" :end="332" />），aux 头 `m2` 是**纯 1×1 卷积**（<SrcLink file="v7:models/yolo.py" :line="329" />）：

```python
self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)   # aux 头：纯 1×1
self.ia = nn.ModuleList(ImplicitA(x) for x in ch)   # 仅 lead
self.im = nn.ModuleList(ImplicitM(x) for x in ch)   # 仅 lead
```

- 训练期双份输出、推理期只留 lead（<SrcLink file="v7:models/yolo.py" :line="362" />）：

```python
return x if self.training else (torch.cat(z, 1), x[:self.nl])
```

- 训练走 `train_aux.py`（构造 `ComputeLossAuxOTA`，<SrcLink file="v7:utils/loss.py" :line="1176" :end="1202" />）；普通 `train.py` 不带 aux 头
- aux 损失统一 ×0.25 降权：lbox（<SrcLink file="v7:utils/loss.py" :line="1259" />）、lcls（<SrcLink file="v7:utils/loss.py" :line="1269" />）、lobj（<SrcLink file="v7:utils/loss.py" :line="1273" />）

**效果**：论文消融 Table 6-8 的三连涨（55.6 → 55.9 → 56.3 → 56.8）一半来自 partial aux。

**推理开销**：零开销。aux 头训练完直接丢弃，推理图里只有 lead 头。

## 7. conv-BN 融合 (`trick-conv-bn-fuse`)

**动机**：论文 §4.3 三条"推理免费" freebies 之一。训练期 conv 与 BN 分开算数值稳，推理期 BN 可折叠进卷积核，白赚一次算子合并。

**实现**（<SrcLink file="v7:models/common.py" :line="560" :end="581" />）：

```python
def fuse_conv_bn(conv, bn):
    fusedconv = nn.Conv2d(..., bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    ...
```

- `Conv.fuseforward`（<SrcLink file="v7:models/common.py" :line="110" :end="111" />）跳过 BN 直接过激活；RepConv 的 `fuse_repvgg_block`（<SrcLink file="v7:models/common.py" :line="584" />）同样基于它折叠

**效果**：数学恒等变换，精度不掉、延迟降低。

**推理开销**：零开销（负开销——推理反而更快）。

## 8. YOLOR 隐式知识并入卷积 (`trick-implicit-fuse`)

**动机**：论文 §4.3 freebie 之二：YOLOR 的隐式知识（ImplicitA 加性、ImplicitM 乘性）训练期作为独立参数学出来，推理期可以合并进检测头卷积的 bias/weight，不增加任何算子。

**实现**：
- ImplicitA（`return self.implicit + x`，<SrcLink file="v7:models/common.py" :line="433" :end="443" />）与 ImplicitM（`return self.implicit * x`，<SrcLink file="v7:models/common.py" :line="446" :end="456" />）插在 IDetect 每层卷积前后（<SrcLink file="v7:models/yolo.py" :line="124" :end="125" />）
- `IDetect.fuse()` 把加性隐变量并进 conv bias、乘性隐变量并进 weight/bias（<SrcLink file="v7:models/yolo.py" :line="178" :end="190" />）：

```python
# fuse ImplicitA and Convolution
self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2), self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)
# fuse ImplicitM and Convolution
self.m[i].bias *= self.im[i].implicit.reshape(c2)
self.m[i].weight *= self.im[i].implicit.transpose(0,1)
```

- 注意 aux 头不带 ia/im（见 [trick 6](#_6-partial-auxiliary-head-trick-partial-aux)），隐式知识只给 lead

**效果**：推理图与"无隐式知识"的模型完全同构，精度白拿。

**推理开销**：零开销。fuse 后隐式知识已固化进卷积参数。

## 9. EMA 平均教师 (`trick-ema`)

**动机**：论文 §4.3 freebie 之三：把 EMA（mean teacher）模型作为最终推理模型，泛化更稳。

**实现**（<SrcLink file="v7:train.py" :line="199" :end="200" />）：

```python
# EMA
ema = ModelEMA(model) if rank in [-1, 0] else None
```

- `ModelEMA` 来自 `utils/torch_utils.py`（v5 同款滑动平均实现），每个 batch 后 `ema.update(model)`
- 评估与最终部署取 EMA 权重而非原始权重（`ckpt = {'ema': ema.ema, ...}` 一路传递给 test/detect）

**效果**：与 v5/v6 同源的成熟配方；论文把 EMA 显式列为三条推理免费 freebies 之一。

**推理开销**：零开销。EMA 只存在于训练态，推理时就是普通权重加载。

## 10. AMP 混合精度 (`trick-amp`)

**动机**：P6 大模型（W6/E6/D6/E6E）训练显存吃紧，FP16 混合精度放大吞吐。

**实现**（<SrcLink file="v7:train.py" :line="299" />、<SrcLink file="v7:train.py" :line="360" :end="361" />）：

```python
scaler = amp.GradScaler(enabled=cuda)
...
with amp.autocast(enabled=cuda):
    pred = model(imgs)
```

- `train_aux.py` 同款（L299 GradScaler、L360 autocast），双头训练同样吃 AMP 红利

**效果**：训练速度/显存双收益，精度几乎无损。

**推理开销**：零开销。AMP 只影响训练，推理精度由部署时的 FP32/FP16/INT8 决定。

## 11. 超参文件分档 (`trick-hyp-files`)

**动机**：不同档位模型/输入尺度的增强与损失超参不同，v7 用四份 hyp 文件分档，训练时 `--hyp` 指定。

**实现**：
- `data/hyp.scratch.p5.yaml`（yolov7 / yolov7-x，P3-P5）、`hyp.scratch.p6.yaml`（W6/E6/D6/E6E，P3-P6）、`hyp.scratch.tiny.yaml`（tiny）、`hyp.scratch.custom.yaml`（自定义数据）
- 关键开关在四份文件里一致：`anchor_t: 4.0`（anchor 宽高比匹配阈值）、`fl_gamma: 0.0`、`loss_ota: 1`（注释：use 0 for faster training）
- `loss_ota` 是标签分配的总开关（<SrcLink file="v7:train.py" :line="362" :end="365" />）：

```python
if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
    compute_loss_ota(pred, targets.to(device), imgs)   # ComputeLossOTA 动态匹配
else:
    compute_loss(pred, targets)                        # ComputeLoss 静态 anchor 匹配
```

**效果**：一条命令切换"动态匹配（慢而准）/ 静态匹配（快）"，以及按模型档位换整套增强超参。

**推理开销**：零开销。纯训练配置。

## 12. 从零训练只用 COCO (`trick-coco-only`)

**动机**：同期很多高分检测器靠 Object365 等额外数据堆上去。论文 §5.1 明确：**从零训练只用 COCO trainval**，不加载额外数据/预训练权重，公平对比。

**实现**：无特殊代码——`train.py --data coco.yaml` 标准流程；论文报 AP 时给 test-dev / val 双口径（如 YOLOv7 51.4 / 51.2）。

**效果**：Table 2 的成绩全在"仅 COCO"约束下取得：YOLOv7 36.9M / 104.7G / 161 FPS / 51.4 AP；E6E 151.7M / 36 FPS / 56.8 AP。"trainable bag-of-freebies" 的含金量正在于此。

**推理开销**：零开销（数据侧约束，与推理无关）。

## 13. label smoothing 与可选 Focal (`trick-smooth-focal`)

**动机**：v3/v5 老三样（CIoU + BCEobj + BCEcls）的 BCE 侧保留两个经典正则开关：label smoothing 防过拟合、Focal loss 平衡难易样本。

**实现**（<SrcLink file="v7:utils/loss.py" :line="432" :end="438" />）：

```python
# Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

# Focal loss
g = h['fl_gamma']  # focal loss gamma
if g > 0:
    BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
```

- 默认全关：四份 hyp 均无 `label_smoothing` 键（走默认 0.0）、`fl_gamma: 0.0`
- FocalLoss 包装式实现（<SrcLink file="v7:utils/loss.py" :line="121" />），乘在 BCEcls/BCEobj 外层
- 损失本体仍是 v3/v5 老三样：CIoU 回归 + obj 目标 `(1-gr)+gr·iou`（<SrcLink file="v7:utils/loss.py" :line="472" />），v7 的创新全在标签分配与 aux 头（见 [trick 4](#_4-lead-head-guided-标签分配-trick-lead-guided)）

**效果**：开关式正则，默认配置即论文口径。

**推理开销**：零开销。纯训练期损失选项。

## 延伸阅读

- [理论 Concept](/versions/v7/concepts)
- [损失函数](/versions/v7/losses)
- [源码走读](/versions/v7/code-walkthrough)
- [痛点追踪](/versions/v7/painpoints)
- [YOLOv6 训练技巧](/versions/v6/tricks)（对照看配方传承）
