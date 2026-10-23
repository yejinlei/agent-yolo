# 架构与层速查

本页以**表格 + YAML 拆解**形式快速查阅 YOLO-World 的所有层——按 backbone / neck / head 三大块组织，对照 v1 / v2 两个分支的差异，并附**Ultralytics 版**与 **AILab-CVC 版**的代码落点。

## 一、v1 YAML 全量拆解（`yolov8-world.yaml`）

引用：<SrcLink file="yoloworld:ultralytics/cfg/models/v8/yolov8-world.yaml" :line="1" :end="48"/>

### 1.1 参数区

```yaml
nc: 80
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `nc` | 80 | 类别数（COCO 默认） |
| `depth` | 0.33~1.0 | 层重复次数缩放 |
| `width` | 0.25~1.25 | 通道数缩放 |
| `max_channels` | 512~1024 | 通道上限 |

### 1.2 Backbone 层表

| 层号 | 模块 | 参数 | 输出通道 | 输出分辨率 | 与 v8 一致 |
|------|------|------|---------|-----------|-----------|
| 0 | Conv | [64, 3, 2] | 64 | 320×320 | ✅ |
| 1 | Conv | [128, 3, 2] | 128 | 160×160 | ✅ |
| 2 | C2f | [128, True] | 128 | 160×160 | ✅ |
| 3 | Conv | [256, 3, 2] | 256 | 80×80 | ✅ |
| 4 | C2f | [256, True] | 256 | 80×80 | ✅ |
| 5 | Conv | [512, 3, 2] | 512 | 40×40 | ✅ |
| 6 | C2f | [512, True] | 512 | 40×40 | ✅ |
| 7 | Conv | [1024, 3, 2] | 1024 | 20×20 | ✅ |
| 8 | C2f | [1024, True] | 1024 | 20×20 | ✅ |
| 9 | SPPF | [1024, 5] | 1024 | 20×20 | ✅ |

### 1.3 Head 层表（v1）

| 层号 | 模块 | 参数 | 输出通道 | 输出分辨率 | 备注 |
|------|------|------|---------|-----------|------|
| 10 | nn.Upsample | [None, 2, "nearest"] | 1024 | 40×40 | 上采样 |
| 11 | Concat | [1] | 1536 | 40×40 | 拼接 #6 |
| 12 | **C2fAttn** | [512, 256, 8] | 512 | 40×40 | ★ v1 独有：VL 融合 |
| 13 | nn.Upsample | [None, 2, "nearest"] | 512 | 80×80 | 上采样 |
| 14 | Concat | [1] | 768 | 80×80 | 拼接 #4 |
| 15 | **C2fAttn** | [256, 128, 4] | 256 | 80×80 | ★ v1 独有 |
| 16 | **ImagePoolingAttn** | [256] | 256 | 80×80 | ★ v1 独有（导出失败） |
| 17 | Conv | [256, 3, 2] | 256 | 40×40 | 下采样 |
| 18 | Concat | [1] | 768 | 40×40 | 拼接 #12 |
| 19 | **C2fAttn** | [512, 256, 8] | 512 | 40×40 | ★ v1 独有 |
| 20 | Conv | [512, 3, 2] | 512 | 20×20 | 下采样 |
| 21 | Concat | [1] | 1536 | 20×20 | 拼接 #9 |
| 22 | **C2fAttn** | [1024, 512, 16] | 1024 | 20×20 | ★ v1 独有 |
| 23 | **WorldDetect** | [nc, 512, False] | — | — | ★ with_bn=False |

### 1.4 v1 关键差异

- **`C2fAttn` ×4**：全部替换 v8 的 `C2f`，加 VL attention；
- **`ImagePoolingAttn` ×1**（L38）：**v1 独有**，无法 export；
- **`WorldDetect [nc, 512, False]`**：最后一个参数 `False` 表示用 `ContrastiveHead`（L2 归一化）。

## 二、v2 YAML 全量拆解（`yolov8-worldv2.yaml`）

引用：<SrcLink file="yoloworld:ultralytics/cfg/models/v8/yolov8-worldv2.yaml" :line="1" :end="46"/>

### 2.1 Backbone 层表（与 v1 完全一致）

（略——完全相同，见上文 1.2 节）

### 2.2 Head 层表（v2）

| 层号 | 模块 | 参数 | 输出通道 | 输出分辨率 | 备注 |
|------|------|------|---------|-----------|------|
| 10 | nn.Upsample | [None, 2, "nearest"] | 1024 | 40×40 | 上采样 |
| 11 | Concat | [1] | 1536 | 40×40 | 拼接 #6 |
| 12 | **C2fAttn** | [512, 256, 8] | 512 | 40×40 | ★ v2 保留 |
| 13 | nn.Upsample | [None, 2, "nearest"] | 512 | 80×80 | 上采样 |
| 14 | Concat | [1] | 768 | 80×80 | 拼接 #4 |
| 15 | **C2fAttn** | [256, 128, 4] | 256 | 80×80 | ★ v2 保留 |
| 16 | Conv | [256, 3, 2] | 256 | 40×40 | 下采样（**替代 v1 的 ImagePoolingAttn**） |
| 17 | Concat | [1] | 768 | 40×40 | 拼接 #12 |
| 18 | **C2fAttn** | [512, 256, 8] | 512 | 40×40 | ★ v2 保留 |
| 19 | Conv | [512, 3, 2] | 512 | 20×20 | 下采样 |
| 20 | Concat | [1] | 1536 | 20×20 | 拼接 #9 |
| 21 | **C2fAttn** | [1024, 512, 16] | 1024 | 20×20 | ★ v2 保留 |
| 22 | **WorldDetect** | [nc, 512, True] | — | — | ★ with_bn=True |

### 2.3 v2 关键差异

- **`C2fAttn` ×4**（同 v1）；
- **去掉 `ImagePoolingAttn`**：v1 的 L38 位置替换成普通 `Conv`；
- **`WorldDetect [nc, 512, True]`**：最后一个参数 `True` 表示用 `BNContrastiveHead`（BN 归一化）；
- **总层数少 1**（v1: 23 层，v2: 22 层）。

## 三、v1 vs v2 层对比表

| 层 | v1 | v2 |
|----|----|----|
| Backbone（#0~#9） | C2f ×4 + SPPF | C2f ×4 + SPPF（**相同**） |
| Upsample ×2 | ✅ | ✅ |
| C2fAttn ×4 | ✅ | ✅ |
| ImagePoolingAttn ×1 | ✅ | ❌（**去掉**） |
| Conv 3×3 s2 ×2 | ✅ | ✅ |
| WorldDetect with_bn | **False**（ContrastiveHead） | **True**（BNContrastiveHead） |
| 总层数 | 23 | 22 |

## 四、`WorldDetect` 内部层速查

引用：<SrcLink file="yoloworld:ultralytics/nn/modules/head.py" :line="211" :end="260"/>

### 4.1 分支结构

```
WorldDetect(x, text)
  ├─ cv2[i]  (回归分支，与 v8 Detect 一致)
  │   ├─ Conv(x, c2, 3)
  │   ├─ Conv(c2, c2, 3)
  │   └─ Conv(c2, 4*reg_max=64, 1)
  │
  ├─ cv3[i]  (类别分支，输出 512-dim)
  │   ├─ Conv(x, c3, 3)
  │   ├─ Conv(c3, c3, 3)
  │   └─ Conv(c3, embed=512, 1)
  │
  └─ cv4[i]  (对比头，新增)
      ├─ if with_bn: BNContrastiveHead(512)
      └─ else:      ContrastiveHead()
```

### 4.2 通道维度

| 分支 | 输入通道 | 输出通道 | 说明 |
|------|---------|---------|------|
| `cv2` | ch[0]/1/2/3 | 64 | 回归（DFL 输入） |
| `cv3` | ch[0]/1/2/3 | 512 | 类别（对比头输入） |
| `cv4` | 512 | K（=num_prompts） | 类别 logits |

### 4.3 Forward 流程

```python
def forward(self, x, text):
    for i in range(self.nl):
        x[i] = torch.cat((
            self.cv2[i](x[i]),                                # reg
            self.cv4[i](self.cv3[i](x[i]), text),             # cls
        ), 1)
```

- 输出 shape: `[B, nc + reg_max*4, H, W]`；
- **训练时**：返回 list of `[B, K+64, H, W]`（不 concat）；
- **推理时**：concat 到全局，做 anchor 解码 + sigmoid。

## 五、AILab-CVC 版层速查

AILab-CVC 使用 mmengine config 风格，模块以 Python 类定义。以下表格列出核心模块的**参数**与**代码落点**：

### 5.1 Backbone（MultiModalYOLOBackbone）

| 模块 | 参数 | 说明 |
|------|------|------|
| `image_model` | YOLOv8 CNN | 视觉编码器 |
| `text_model` | `HuggingCLIPLanguageBackbone` | 文本编码器（可置 None） |
| `text_model.frozen_modules` | `['all']` | 冻结所有文本参数 |
| `text_model.max_length` | 77 | CLIP tokenizer 上限 |

- 引用：见 `ailab-cvc/configs/pretrain_v1/*.py` 里的 `model.backbone` 定义。

### 5.2 Neck（YOLOWorldPAFPN / YOLOWorldDualPAFPN）

| 模块 | v1 | v2 |
|------|-----|-----|
| `YOLOWorldPAFPN` | ❌ | ✅ |
| `YOLOWorldDualPAFPN` | ✅ | ❌ |
| `guide_channels` | 512 | 512 |
| `embed_channels` | 512 | 512 |
| `num_heads` | 8 | 8 |
| `text_enhancer` | ImagePoolingAttentionModule | 无 |

- `YOLOWorldPAFPN` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py" :line="16" :end="146"/>
- `YOLOWorldDualPAFPN` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py" :line="150" :end="235"/>

### 5.3 Head Module

| 模块 | v1 | v2 |
|------|-----|-----|
| `YOLOWorldHeadModule` | ✅ | ❌ |
| `RepYOLOWorldHeadModule` | ❌ | ✅ |
| `embed_dims` | 512 | 512 |
| `use_bn_head` | False | True |
| `use_einsum` | True | True |
| `freeze_all` | False | False |
| `reparameterized` | — | True（微调时） |

- `YOLOWorldHeadModule` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="135" :end="289"/>
- `RepYOLOWorldHeadModule` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="293" :end="344"/>

### 5.4 Head（YOLOWorldHead）

| 参数 | 值 | 说明 |
|------|-----|------|
| `reg_max` | 16 | DFL 通道（沿用 v8） |
| `num_classes` | 80（训练时）/ txt_feats.shape[0]（推理时） | 动态类数 |
| `stride` | [8, 16, 32] | 三尺度 stride |
| `topk` | 10 | TAL 参数 |
| `alpha` | 0.5 | TAL 参数 |
| `beta` | 6.0 | TAL 参数 |
| `loss_cls` | CrossEntropyLoss | 分类损失 |
| `loss_bbox` | BboxLoss | 边界框损失 |
| `loss_dfl` | DistributionFocalLoss | DFL 损失 |

- `YOLOWorldHead` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="348" :end="749"/>

### 5.5 Detector（YOLOWorldDetector / SimpleYOLOWorldDetector）

| 参数 | v1 (YOLOWorldDetector) | v2 (SimpleYOLOWorldDetector) |
|------|------------------------|-------------------------------|
| `mm_neck` | 可选 | 可选 |
| `num_train_classes` | 80 | 80 |
| `num_test_classes` | 80 | 80 |
| `prompt_dim` | — | 512 |
| `num_prompts` | — | 80 |
| `embedding_path` | — | 可指定 .npz 文件 |
| `reparameterized` | — | False/True |
| `freeze_prompt` | — | False（可训练） |
| `use_mlp_adapter` | — | False（可选 MLP） |

- `YOLOWorldDetector` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/detectors/yolo_world.py" :line="12" :end="105"/>
- `SimpleYOLOWorldDetector` 引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/detectors/yolo_world.py" :line="109" :end="232"/>

## 六、各档参数规模速查（v2 系列）

| 档位 | Backbone repeats | Width 缩放 | Params | GFLOPs | FPS (V100) | COCO z-shot AP |
|------|-----------------|-----------|--------|--------|------------|----------------|
| n | 0.33 | 0.25 | — | — | — | —（未发布） |
| s | 0.33 | 0.50 | 24.5 M | 74.2 | 74.1 | 37.7 |
| m | 0.67 | 0.75 | 38.2 M | 138.2 | 58.1 | 43.0 |
| l | 1.00 | 1.00 | 55.4 M | 196.4 | 52.0 | 45.8 |
| x | 1.00 | 1.25 | 72.6 M | 263.9 | 47.3 | 47.1 |

**档位选择建议**：

- **移动端**：s（24.5M，74 FPS，37.7 AP）
- **边缘计算**：m（38.2M，58 FPS，43.0 AP）
- **服务器推理**：l（55.4M，52 FPS，45.8 AP）
- **研究 / 精调**：x（72.6M，47 FPS，47.1 AP）

## 七、Layer 类型分类速查

| 类别 | 模块 | 作用 |
|------|------|------|
| **卷积** | Conv | 标准 3×3 卷积 + BN + SiLU |
| **注意力** | C2fAttn | C2f + 内嵌 VL attention |
| **池化** | SPPF | Space Pyramid Pooling |
| **融合** | Concat, ImagePoolingAttn | 多尺度拼接 / 全局 VL 池化 |
| **重参数化** | RepBNContrastiveHead | BN + Conv 折叠 |
| **检测头** | Detect, WorldDetect | 解耦头（reg + cls） |
| **对比头** | ContrastiveHead, BNContrastiveHead | 视觉-文本对比 |
| **归一化** | BN, LN | 标准化 |
| **激活** | SiLU | Sigmoid Linear Unit |

## 八、常见层组合速查

**Backbone 块**：

```
[Conv 3×3 s2] + [C2f repeat=N]
```

**Neck 块**：

```
[Upsample ×2] + [Concat] + [C2fAttn repeat=N]
```

**Head 块**：

```
[Conv 3×3] + [Concat] + [C2fAttn repeat=N]
  ↓ (P3/P4/P5)
[WorldDetect cv2 + cv3 + cv4]
```

## 九、常见报错速查

| 报错 | 原因 | 解决 |
|------|------|------|
| `ImportError: No module named clip` | 未装 CLIP | `pip install git+https://github.com/openai/CLIP.git` |
| `AttributeError: 'WorldModel' has no attribute 'txt_feats'` | 用 `YOLO()` 而非 `WorldModel()` 加载 | 换用 `WorldModel` 入口 |
| `OSError: Unable to open engine` | v1 checkpoint export 失败 | 换用 v2 权重 |
| `ValueError: nc must be <= 80` | Ultralytics 硬编码限制 | 用 AILab-CVC 版跑 >80 类 |
| `RuntimeError: CLIP not available on this device` | CPU-only CLIP 报错 | 用 GPU 训练，或 CPU 上设 `device='cpu'` |
| `KeyError: 'texts'` | Dataset 未提供文本 | 检查 `data.yaml` 的 `texts` 字段 |

## 十、与 Ultralytics 主线层的对应

| YOLO-World 层 | Ultralytics 主线 | 备注 |
|---------------|-----------------|------|
| `C2fAttn` | `C2f` + `Attention` | 快照未收录 block.py |
| `ImagePoolingAttn` | 无（YOLO-World 独有） | v1 独有 |
| `WorldDetect` | `Detect` | Ultralytics 8.1.39 新增 |
| `WorldConv` | 无 | 定义在 head.py，快照中 |
| `WorldAlignMLP` | 无 | 定义在 head.py，快照中 |
| `ContrastiveHead` | 无（YOLO-World 独有） | Ultralytics 版定义在 block.py |
| `BNContrastiveHead` | 无（YOLO-World 独有） | Ultralytics 版定义在 block.py |
| `RepBNContrastiveHead` | 无（YOLO-World 独有） | AILab-CVC 独有 |

## 深入阅读

- [Architecture 逐层图解](./architecture) · [理论 Concept](./concepts) · [损失函数](./losses) · [训练技巧](./tricks) · [痛点追踪](./painpoints) · [源码走读](./code-walkthrough)
