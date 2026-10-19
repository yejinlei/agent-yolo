# YOLOv6 训练技巧

## 全部 Trick 速查

| Trick | 代码位置 | 说明 |
|-------|----------|------|
| RepVGGBlock 重参数化 | common.py L197-319 | 训练三分支 → 推理单 3×3 |
| BottleRep 加权残差 | common.py L590-608 | 可学习 alpha 残差 |
| EfficientRep 骨干 | efficientrep.py L7-118 | RepVGG 风格 stage 堆叠 |
| BepC3 / RepBlock | common.py L569-650 | CSP 式重参数化聚合 |
| CSPSPPF | common.py L133-176 | CSP 结构的 SPPF |
| BiFusion / RepBiFPANNeck | reppan.py L132-238 | 三路双向跨层融合 |
| EffiDeHead 解耦头 | effidehead.py L10-139 | 硬件感知混合通道 |
| TAL 标签分配 | tal_assigner.py L6-150 | 对齐度量 top-13 |
| VarifocalLoss | loss.py L201-211 | IoU 感知软标签 |
| GIoU + 损失加权 | loss.py L26-29 | iou×2.5 / dfl×0.5 |
| DFL 分布回归开关 | loss.py L244-259 | 蒸馏续训才启用 |
| Mosaic / MixUp / 仿射 | data_augment.py L61-113 | 强增强组合 |
| 关闭强增强 | engine.py L323-329 | 末 N epoch 停 mosaic |
| EMA | ema.py L21-37 | 衰减 warmup 的滑动平均 |
| Cosine LR + warmup | solver/build.py L36-46 | bias lr 0.1 起步插值 |
| AMP + 梯度累积 | engine.py L149-175 | 混合精度 + 累积到 64 |
| QARepVGGBlock | common.py L322-393 | 量化友好重参数化 |
| 知识蒸馏 | engine.py L306-321 | channel-wise distillation |

## 1. RepVGGBlock 重参数化 (`trick-reparam`)

**动机**：训练时多分支结构正则化效果好、精度高；部署时希望只有一次 3×3 卷积，吃满 GPU/TensorRT 的算子效率。YOLOv6 把"训练态"和"部署态"做成同一个类的两种形态。

**实现**（<SrcLink file="v6:yolov6/layers/common.py" :line="197" :end="319" />）：训练态维护三个分支——3×3 卷积、1×1 卷积、identity（BN）：

```python
self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, ...)
self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, ...)
```

前向是三分支相加再过 ReLU（<SrcLink file="v6:yolov6/layers/common.py" :line="244" :end="254" />）：

```python
return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
```

**源码锚点**：
- 等价核融合 `get_equivalent_kernel_bias`（<SrcLink file="v6:yolov6/layers/common.py" :line="256" :end="260" />）：1×1 核 zero-pad 成 3×3、BN 折叠进卷积核（`_fuse_bn_tensor`，<SrcLink file="v6:yolov6/layers/common.py" :line="277" :end="299" />）后三路相加
- `switch_to_deploy`（<SrcLink file="v6:yolov6/layers/common.py" :line="301" :end="318" />）：把等价核写进一个新的 `nn.Conv2d`（`rbr_reparam`），再 `__delattr__` 删掉三个训练分支，`self.deploy = True`

**效果**：部署后每个 block 只有一次 3×3 卷积 + ReLU，精度与训练态完全等价（数学恒等变换，无精度损失）；推理速度大幅提升是 YOLOv6 "工业友好"定位的基石。代价见 [痛点追踪](/versions/v6/painpoints)：训练期要跑三分支，显存与算力开销更高。

## 2. BottleRep 加权残差 (`trick-bottlerep`)

**动机**：RepBlock 深堆叠时朴素残差容易退化，YOLOv6 给残差加一个可学习权重，让网络自己决定每层"跳过多少"。

**实现**（<SrcLink file="v6:yolov6/layers/common.py" :line="590" :end="608" />）：

```python
if weight:
    self.alpha = Parameter(torch.ones(1))
...
def forward(self, x):
    outputs = self.conv1(x)
    outputs = self.conv2(outputs)
    return outputs + self.alpha * x if self.shortcut else outputs
```

- BottleRep 内部是两个 basic_block（RepVGGBlock）串联 + 加权 shortcut
- BottleRep3 是三卷积版（<SrcLink file="v6:yolov6/layers/common.py" :line="611" :end="631" />），供 MBLABlock 使用
- RepBlock 检测到 `block == BottleRep` 时自动把重复次数减半（`n = n // 2`，<SrcLink file="v6:yolov6/layers/common.py" :line="578" :end="581" />），控制计算量

**效果**：残差强度可学习，深层 stage（如 num_repeats=18 的 ERBlock_4）训练更稳。

## 3. EfficientRep 骨干 (`trick-efficientrep`)

**动机**：v5 的 Focus 切片算子在推理引擎里难优化（见 [v5 痛点](/versions/v5/painpoints)）；YOLOv6 换成全 3×3 卷积的重参数化骨干，对 GPU 友好。

**实现**（<SrcLink file="v6:yolov6/models/efficientrep.py" :line="7" :end="118" />）：

```python
self.stem = block(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2)
self.ERBlock_2 = nn.Sequential(block(..., kernel_size=3, stride=2), RepBlock(...))
...
channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
if cspsppf:
    channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF
```

- stem 是 3×3 stride-2 的 RepVGGBlock（推理折叠成普通卷积），逐步下采样
- ERBlock_2~5 各含一个 stride-2 block + 一个 RepBlock；ERBlock_5 末端接 CSPSPPF 做感受野聚合（<SrcLink file="v6:yolov6/models/efficientrep.py" :line="80" :end="102" />）
- `fuse_P2=True` 时多输出一层 P2 特征（<SrcLink file="v6:yolov6/models/efficientrep.py" :line="104" :end="118" />）

**效果**：全卷积结构对 TensorRT/NCNN 友好；配合 `depth_multiple=0.33, width_multiple=0.50` 得到 YOLOv6s（见 [源码走读](/versions/v6/code-walkthrough)）。

## 4. BepC3 与 RepBlock (`trick-bepc3`)

**动机**：把 v5 C3 的 CSP 思路与重参数化结合：训练期用多分支 Bottleneck 堆叠，部署期折叠。

**实现**：
- RepBlock 是"rep-style 基本块"的堆叠容器（<SrcLink file="v6:yolov6/layers/common.py" :line="569" :end="587" />）
- BepC3（CSPStackRep）把输入 1×1 拆成两支：一支直连、一支过 RepBlock(BottleRep)，再 concat + 1×1 融合（<SrcLink file="v6:yolov6/layers/common.py" :line="634" :end="650" />）：

```python
self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)
def forward(self, x):
    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
```

- CSP 系骨干/颈部（CSPBepBackbone、CSPRepPANNeck）通过 `stage_block_type` 选择 BepC3 或 MBLABlock（build_network 默认 `"BepC3"`，<SrcLink file="v6:yolov6/models/yolo.py" :line="75" :end="78" />）

**效果**：CSP 的梯度分流 + 重参数化的部署效率兼得，是 P6 大模型与 MBLA 系列的公共底座。

## 5. CSPSPPF (`trick-cspsppf`)

**动机**：SPPF 的多次 MaxPool 串行依赖长，CSP 化后把大感受野聚合拆成两条路，延迟更低。

**实现**（<SrcLink file="v6:yolov6/layers/common.py" :line="133" :end="156" />）：

```python
def forward(self, x):
    x1 = self.cv4(self.cv3(self.cv1(x)))
    y0 = self.cv2(x)
    ...
    y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
    return self.cv7(torch.cat((y0, y3), dim=1))
```

- 一路 1×1 直连（`cv2`），一路经 3×3 + 4×5×5 MaxPool 串联聚合（`cv5/cv6`），最后 concat 融合
- `cspsppf=True` 时骨干末端用 CSPSPPF/SimCSPSPPF 替代 SPPF（<SrcLink file="v6:yolov6/models/efficientrep.py" :line="80" :end="82" />）

**效果**：YOLOv6s 默认开启（`configs/yolov6s.py` L12），速度/精度双赢的轻量改进。

## 6. BiFusion 与 RepBiFPANNeck (`trick-bifusion`)

**动机**：PANet 的自顶向下融合只做"上采样 + concat"，YOLOv6 让高层、同层、低层三路都参与融合，且每路先各自变换再合并。

**实现**：
- BiFusion：高层过转置卷积上采样、同层过 1×1、低层过 3×3 下采样，三路 concat 后 1×1 融合（<SrcLink file="v6:yolov6/layers/common.py" :line="695" :end="718" />）：

```python
def forward(self, x):
    x0 = self.upsample(x[0])
    x1 = self.cv1(x[1])
    x2 = self.downsample(self.cv2(x[2]))
    return self.cv3(torch.cat((x0, x1, x2), dim=1))
```

- RepBiFPANNeck 用两个 BiFusion 做自顶向下，再用 stride-2 下采样 + concat 做自底向上（<SrcLink file="v6:yolov6/models/reppan.py" :line="215" :end="237" />）：

```python
f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
f_out0 = self.Rep_p4(f_concat_layer0)
```

**效果**：neck 中间全是 RepBlock/ConvBNReLU，部署期同样可折叠或直转，无特殊算子。

## 7. EffiDeHead 解耦头 (`trick-effidehead`)

**动机**：v5 的 Detect 头把分类/回归挤在同一层卷积里（见 [v5 痛点](/versions/v5/painpoints)），且通道分配一刀切。YOLOv6 做显式解耦，并按硬件实测调每层通道。

**实现**（<SrcLink file="v6:yolov6/models/effidehead.py" :line="10" :end="47" />）：每个检测层五件套 `stem / cls_conv / reg_conv / cls_pred / reg_pred`，分类回归走不同卷积塔：

```python
cls_feat = self.cls_convs[i](cls_x)
cls_output = self.cls_preds[i](cls_feat)
reg_feat = self.reg_convs[i](reg_x)
reg_output = self.reg_preds[i](reg_feat)
```

- 训练 forward 输出 `(featmaps, cls_score_list, reg_distri_list)`（<SrcLink file="v6:yolov6/models/effidehead.py" :line="71" :end="92" />）；推理 forward 直接解码出 `xywh` 框（<SrcLink file="v6:yolov6/models/effidehead.py" :line="93" :end="139" />）
- 偏置初始化：cls bias 设为 $-\log\frac{1-p}{p}$（prior_prob=1e-2）、reg bias=1.0（<SrcLink file="v6:yolov6/models/effidehead.py" :line="49" :end="65" />）
- 默认 anchor-free：预测是到网格中心的 ltrb 距离（`dist2bbox`），配置里的 `anchors_init` 只被 `--fuse_ab` 辅助分支使用

**效果**：解耦 + 混合通道（hybridchannels）在同精度下比"均分通道"更快，是 YOLOv6 速度指标领先的关键之一。

## 8. TaskAlignedAssigner 标签分配 (`trick-tal`)

**动机**：SimOTA 之类的分配依赖动态 topk 与 Sinkhorn 迭代，开销大且超参敏感；TAL 用"分类-定位对齐度量"一次选正样本，训练更快更稳。

**实现**（<SrcLink file="v6:yolov6/assigners/tal_assigner.py" :line="6" :end="150" />）：

```python
align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)   # alpha=1.0, beta=6.0
...
mask_pos = mask_topk * mask_in_gts * mask_gt
```

- 三重掩码：anchor 中心落在 GT 框内 ∧ 对齐度量排 top-13 ∧ GT 有效（<SrcLink file="v6:yolov6/assigners/tal_assigner.py" :line="97" :end="115" />）
- 冲突消解：一个 anchor 被多个 GT 选中时只保留 IoU 最高者（`select_highest_overlaps`）
- 软标签：`target_scores = one_hot × norm_align_metric`（<SrcLink file="v6:yolov6/assigners/tal_assigner.py" :line="76" :end="81" />），直接喂给 VarifocalLoss
- 保留 ATSS warmup 通道（`ATSSAssigner(9)`，<SrcLink file="v6:yolov6/models/losses/loss.py" :line="41" :end="42" />），但配置 `atss_warmup_epoch=0` 时全程 TAL

**效果**：分配在单次 forward 内完成（`@torch.no_grad()`），比 SimOTA 省显存省时间；n_max_boxes>100 时按 batch 切块防 OOM（<SrcLink file="v6:yolov6/assigners/tal_assigner.py" :line="55" :end="58" />）。

## 9. VarifocalLoss (`trick-varifocal`)

**动机**：分类分支需要区分"正样本的定位质量"与"负样本的抑制强度"，BCE 一视同仁不够精细。

**实现**（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="201" :end="211" />）：

```python
def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()
```

- 负样本权重 $\alpha p^\gamma$（聚焦难负样本），正样本权重是 TAL 给的软 IoU 分数 $t$
- 调用处 one-hot 编码后除以 `target_scores_sum` 归一（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="160" :end="169" />）

**效果**：配合 TAL 软标签，分类分支学的是"对齐度量"而不只是 0/1，mAP 提升明显且收敛快。

## 10. GIoU 与损失加权 (`trick-loss-weight`)

**动机**：回归损失选型上 YOLOv6 回到 GIoU，避免 CIoU 那套长宽比项的数值不稳定（承接 [v5 痛点](/versions/v5/painpoints)）；再用手工权重平衡三项。

**实现**：
- 配置固定 `iou_type='giou'`（<SrcLink file="v6:configs/yolov6s.py" :line="31" />），`IOUloss` 走 GIoU 分支
- 总损失权重 `{'class': 1.0, 'iou': 2.5, 'dfl': 0.5}`（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="26" :end="29" />）：

```python
loss = self.loss_weight['class'] * loss_cls + \
       self.loss_weight['iou'] * loss_iou + \
       self.loss_weight['dfl'] * loss_dfl
```

- IoU 损失以 TAL 的 `target_scores` 为权重、除以 `target_scores_sum` 归一（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="234" :end="241" />）

**效果**：定位项权重（2.5）高于分类项，引导前期快速收敛框质量。

## 11. DFL 分布式回归开关 (`trick-dfl`)

**动机**：把坐标回归从单值变成 $reg\_max+1$ 个 bin 的分布，表达不确定性；但 DFL 与蒸馏续训绑定，标准训练默认关闭。

**实现**：
- 解码：`softmax(pred.view(..., 4, reg_max+1)) @ proj`，proj 为 `linspace(0, reg_max)`（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="194" :end="198" />）
- 损失：左右整数 bin 的 cross_entropy 加权（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="267" :end="278" />），仅 `if self.use_dfl` 才计算，否则 `loss_dfl = pred_dist.sum() * 0.` 占位（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="244" :end="259" />）
- 配置开关 `use_dfl=False, reg_max=0`，注释写明"set to True if you want to further train with distillation"（<SrcLink file="v6:configs/yolov6s.py" :line="32" :end="33" />）

**效果**：默认关 → 无 DFL 开销；蒸馏续训开 → 精度再涨。开关状态容易被忽略，是 v6 的一个认知负担（见 [痛点](/versions/v6/painpoints)）。

## 12. 数据增强组合 (`trick-augment`)

**动机**：沿用 v5/v4 验证过的增强组合，保证小模型泛化。

**实现**：
- Mosaic 四图拼接（`mosaic_augmentation`，<SrcLink file="v6:yolov6/data/data_augment.py" :line="146" />；按 `hyp["mosaic"]` 概率触发，<SrcLink file="v6:yolov6/data/datasets.py" :line="177" :end="179" />）
- MixUp：`beta(32.0, 32.0)` 混合比例（<SrcLink file="v6:yolov6/data/data_augment.py" :line="61" :end="66" />），嵌在 Mosaic 分支里按 `hyp["mixup"]` 触发（<SrcLink file="v6:yolov6/data/datasets.py" :line="182" :end="187" />）；**注意 `configs/yolov6s.py` 默认 `mixup=0.0` 关闭**，只在 mosaic=1.0
- random_affine：旋转/缩放/平移/剪切（<SrcLink file="v6:yolov6/data/data_augment.py" :line="77" :end="113" />），参数 `degrees=0.0, translate=0.1, scale=0.5, shear=0.0`
- HSV LUT 色彩抖动（<SrcLink file="v6:yolov6/data/data_augment.py" :line="13" :end="26" />）+ flipud/fliplr（<SrcLink file="v6:yolov6/data/datasets.py" :line="483" :end="493" />）

**效果**：与 v5 同源（代码注释标明 based on ultralytics/yolov5 dataloaders），成熟稳定。

## 13. 关闭强增强 (`trick-stop-aug`)

**动机**：Mosaic/MixUp 这类强增强在最后阶段会拖累收敛到最高精度，YOLOv6 训末尾自动"熄火"。

**实现**（<SrcLink file="v6:yolov6/core/engine.py" :line="323" :end="329" />）：

```python
if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
    self.cfg.data_aug.mosaic = 0.0
    self.cfg.data_aug.mixup = 0.0
    self.args.cache_ram = False
    self.train_loader, self.val_loader = self.get_data_loader(...)
```

- 直接改配置并重建 dataloader；resume 时若已进入该区间也同步关闭（<SrcLink file="v6:yolov6/core/engine.py" :line="87" :end="89" />）
- 该区间内的 best ckpt 单独存为 `best_stop_aug_ckpt`（<SrcLink file="v6:yolov6/core/engine.py" :line="206" :end="210" />）

**效果**：官方消融显示关强增强能再涨零点几个点，几乎零成本。

## 14. EMA (`trick-ema`)

**动机**：与 v5 相同——评估用滑动平均权重，泛化更稳。

**实现**（<SrcLink file="v6:yolov6/utils/ema.py" :line="21" :end="37" />）：

```python
self.decay = lambda x: decay * (1 - math.exp(-x / 2000))   # decay=0.9999
...
item *= decay
item += (1 - decay) * state_dict[k].detach()
```

- 衰减系数带 warmup（随更新次数逼近 0.9999），每步 `ema.update(model)`（<SrcLink file="v6:yolov6/core/engine.py" :line="373" :end="374" />）
- 每 epoch 同步 `nc/names/stride` 属性（<SrcLink file="v6:yolov6/core/engine.py" :line="181" />）

**效果**：与 v5 同款成熟配方，评估指标用 ema 权重。

## 15. Cosine LR + warmup (`trick-lr-warmup`)

**动机**：大 batch + 重参数化结构训练初期不稳，warmup + cosine 衰减是标准解。

**实现**：
- Cosine 调度 `lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1`（<SrcLink file="v6:yolov6/solver/build.py" :line="36" :end="46" />），`lr0=0.01, lrf=0.01`
- warmup 期内对每个参数组线性插值：bias 组从 `warmup_bias_lr=0.1` 起、momentum 从 0.8 爬到 0.937（<SrcLink file="v6:yolov6/core/engine.py" :line="359" :end="368" />）；warmup 总步数 `max(warmup_epochs × max_stepnum, 1000)`（<SrcLink file="v6:yolov6/core/engine.py" :line="274" />）
- 优化器三参数组：BN 权重不加 weight decay、普通权重加 `weight_decay=0.0005`、bias 不加（<SrcLink file="v6:yolov6/solver/build.py" :line="12" :end="33" />）

**效果**：SGD nesterov + Cosine + 分组 weight decay，与 v5 配方同源但参数组划分更讲究。

## 16. AMP + 梯度累积 (`trick-amp`)

**动机**：重参数化训练态分支多、显存占用高，混合精度 + 累积变相放大 batch。

**实现**（<SrcLink file="v6:yolov6/core/engine.py" :line="149" :end="175" />）：

```python
with amp.autocast(enabled=self.device != 'cpu'):
    preds, s_featmaps = self.model(images)
    total_loss, loss_items = self.compute_loss(...)
self.scaler.scale(total_loss).backward()
self.update_optimizer()
```

- 梯度累积步数 `max(1, round(64 / batch_size))`，即等效 batch 64（<SrcLink file="v6:yolov6/core/engine.py" :line="361" />）
- `GradScaler` 在 before_train_loop 创建（<SrcLink file="v6:yolov6/core/engine.py" :line="277" />）

**效果**：小显存也能训到等效大 batch，速度/精度近似大 batch 训练。

## 17. QARepVGGBlock 量化友好重参数化 (`trick-qarepvgg`)

**动机**：经典 RepVGG 折叠后没有 BN，PTQ 校准与 QAT 都容易掉点；QARepVGG 保留一个"后置 BN"给量化器当校准锚点。

> ⚠️ 术语消歧：这里的 "QARepVGG" 是 YOLOv6 **release v4.0**（2023）引入的块，与 2020 年的 **YOLOv4**（v4 系列检测器）毫无关系。本页凡提到"v4.0"均指 YOLOv6 自己的版本号。

**实现**（<SrcLink file="v6:yolov6/layers/common.py" :line="321" :end="392" />）：

```python
if not deploy:
    self.bn = nn.BatchNorm2d(out_channels)   # 后置 BN
    self.rbr_1x1 = nn.Conv2d(..., bias=False)
    self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
...
# keep post bn for QAT
```

- 前向：`nonlinearity(bn(se(rbr_dense + rbr_1x1 + id_out)))`——三分支先加，BN 放在求和之后（<SrcLink file="v6:yolov6/layers/common.py" :line="336" :end="345" />）
- `switch_to_deploy` 折叠后**保留** `self.bn`（注释 `keep post bn for QAT`，<SrcLink file="v6:yolov6/layers/common.py" :line="389" :end="392" />）
- 还有加平均池化分支的 QARepVGGBlockV2（<SrcLink file="v6:yolov6/layers/common.py" :line="395" :end="409" />）
- 通过 `training_mode='qarepvgg'` 切换（`get_block`，<SrcLink file="v6:yolov6/layers/common.py" :line="721" :end="737" />）

**效果**（configs/qarepvgg/README.md 实测）：float 精度微降（N 37.5→37.1、S 45.0→44.7、M 50.0→49.7），INT8 PTQ 大涨（N 34.3→36.4、S 41.3→44.0、M 48.1→49.4）——用零点几个 float 点换两三个量化点。

## 18. 知识蒸馏 (`trick-distill`)

**动机**：工业落地要求小模型精度尽可能逼近大模型；YOLOv6 把蒸馏做成一等公民（同样是 release v4.0 的招牌能力，注意与 YOLOv4 无关）。

**实现**：
- teacher/student 双前向 + `ComputeLoss_distill(_ns)`（<SrcLink file="v6:yolov6/core/engine.py" :line="152" :end="158" />），n/s 模型用 `ComputeLoss_distill_ns`（<SrcLink file="v6:yolov6/core/engine.py" :line="306" :end="310" />）
- channel-wise 特征蒸馏（`--distill_feat`），蒸馏权重来自配置 `distill_weight={'class': 1.0, 'dfl': 1.0}`（<SrcLink file="v6:configs/yolov6s.py" :line="34" :end="37" />）
- 蒸馏续训时配套打开 DFL：`use_dfl=True, reg_max=16`（见 [trick 11](#_11-dfl-分布式回归开关-trick-dfl)）
- 量化蒸馏配方（tools/qat/README.md）：RepOpt-S PTQ 掉 0.5% → QAT + channel-wise 蒸馏拉回 43.0% mAP → TensorRT 引擎 43.3%（仅差 float 0.1）

**效果**：小模型 + 蒸馏 + QAT 的组合拳，是 YOLOv6 "工业部署"叙事的核心；但也把 DFL 开关、蒸馏头（`effidehead_distill_ns`/`effidehead_fuseab`）等复杂度带进了主干流程。

## 延伸阅读

- [理论 Concept](/versions/v6/concepts)
- [损失函数](/versions/v6/losses)
- [源码走读](/versions/v6/code-walkthrough)
- [痛点追踪](/versions/v6/painpoints)
- [YOLOv5 训练技巧](/versions/v5/tricks)（对照看配方传承）
