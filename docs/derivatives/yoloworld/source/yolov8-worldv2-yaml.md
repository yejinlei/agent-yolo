# ultralytics/cfg/models/v8/yolov8-worldv2.yaml

YOLO-World v2 的结构配置：**46 行、与 v1 cfg 的差别只在 head 部分 3 处改动、v2 是「能导出 ONNX/TensorRT 的唯一 Ultralytics 版本」**。核心替换三处：L32/L36/L40/L44 的 `C2fAttn` 保留；**L38 的 `ImagePoolingAttn` 删除**（这是 v1 无法导出 ONNX 的唯一元凶）；[L46 WorldDetect [nc, 512, True]](#L46) 的第三个参数从 `False` 变 `True`——触发 `BNContrastiveHead`（BN 归一化），后续 `reparameterized=True` 时 BN 权重折叠进 1×1 Conv。

> 快照路径：`source/yoloworld/snapshot/ultralytics/cfg/models/v8/yolov8-worldv2.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ultralytics/cfg/models/v8/yolov8-worldv2.yaml" />

**阅读路线**：

- [L5 nc=80](#L5) 与 [L6-L12 scales](#L6-L12)：与 v1 / v8 主线一字不差，n/s/m/l/x 五档复合缩放
- Backbone [L15-L26](#L15-L26)：10 层——Conv s2 ×2 → C2f(128,True) → P3/8 → C2f(256,True) → P4/16 → C2f(512,True) → P5/32 → C2f(1024,True) → SPPF(1024, 5)
- Head [L29-L46](#L29-L46)：
  - FPN 上行 [L30-L32](#L30-L32)：Upsample + Concat + `C2fAttn(512, 256, 8)`——v1 的 attention 结构**保留**
  - 下到 P3/8 [L34-L36](#L34-L36)：Upsample + Concat + `C2fAttn(256, 128, 4)`
  - **★ v2 关键差异**：v1 在 L38 有 `ImagePoolingAttn` 汇聚三尺度并反向更新 `txt_feats`；v2 **把这层直接删掉**，改为 [L38 Conv [256, 3, 2]](#L38) 直接下采样——**这是 v2 可 export ONNX 的唯一条件**
  - PAN 下行 [L40-L44](#L40-L44)：两组 Conv s2 + Concat + `C2fAttn`（无 `ImagePoolingAttn`）
  - [L46 WorldDetect [nc, 512, True]](#L46)：`embed=512` + `with_bn=True`——触发 `BNContrastiveHead`

## 与 v1 配置 yolov8-world.yaml 的对比

| 维度 | v1 (world.yaml) | v2 (worldv2.yaml) |
|------|-----------------|-------------------|
| 行数 | 48 | 46 |
| `C2fAttn` 出现次数 | 4 处 | 4 处（保留） |
| `ImagePoolingAttn` | 1 处（L38） | **0 处** ★ |
| 补层 Conv | 无 | 1 处（L38 补 Conv s2） |
| `WorldDetect` 参数 | `[nc, 512, False]` | `[nc, 512, True]` ★ |
| `with_bn` | False（L2 归一化） | True（BN 归一化） |
| 对比头类 | `ContrastiveHead` | `BNContrastiveHead` |
| `logit_scale` 初值 | `log(1/0.07) ≈ 2.659` | `-1.0`（更保守稳定） |
| 可重参数化 | ❌ | ✅（`RepBNContrastiveHead`） |
| ONNX / TensorRT 导出 | ❌ | ✅ |
| `txt_feats` 反向更新 | ✅（L38 汇聚更新） | ❌ |
| 论文 COCO AP | v1-X 34.9 | v2-X **40.9**（finetune 后） |
| 论文 LVIS AP | v1-X 34.7 | v2-X **35.0** |

**核心叙事**：v1 → v2 是**用可部署性换一点点精度**的工程决策。v1 的 `ImagePoolingAttn` 虽然能反向更新 `txt_feats` 提升精度上限，但内含 Python `for` 循环与切片，无法被 ONNX 序列化；v2 用纯 BN + 可重参数化把 `txt_feats` 折叠进 `1×1 Conv` 权重，换来 100% 的导出成功率。逐层解读见 [架构页](../architecture)，attention 与 BN 归一化细节见 [layers 页](../layers)，损失与 `txt_masks` 加权见 [losses 页](../losses)，v1 vs v2 的取舍脉络见 [痛点页](../painpoints)。
