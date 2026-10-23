# ultralytics/cfg/models/v8/yolov8-world.yaml

YOLO-World v1 的结构配置：**48 行、五档 scale 与 v8 主线完全一致、backbone 与 v8 一字不差、head 层数从 v8 的 16 层扩到 19 层**——多出的 3 层全部为「文本感知」而设：两处 `C2fAttn`（把 `txt_feats` 送入 neck 做 attention 加权）+ 一处 `ImagePoolingAttn`（把三尺度特征汇聚后反过来更新 `txt_feats`）。`WorldDetect [nc, 512, False]` 是 v1 的招牌——`embed=512` 是 cv3 输出维度（不再等于 `nc`），`with_bn=False` 走 L2 归一化对比头。

> 快照路径：`source/yoloworld/snapshot/ultralytics/cfg/models/v8/yolov8-world.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ultralytics/cfg/models/v8/yolov8-world.yaml" />

**阅读路线**：

- [L5 nc=80](#L5)：与 v8 相同默认 80 类；[L6-L12 scales 五档](#L6-L12)：`[depth, width, max_channels]` 五档与 [v8 yolov8.yaml](/versions/v8/source/yolov8-yaml) 一字不差，n/s 上限 1024、m 768、l/x 512
- Backbone [L15-L26](#L15-L26)：10 层——Conv s2 ×2 → C2f(128,True) → P3/8 → C2f(256,True) → P4/16 → C2f(512,True) → P5/32 → C2f(1024,True) → SPPF(1024, 5)——**与 v8 主线完全一致，没有任何 YOLO-World 独有模块**
- Head [L29-L48](#L29-L48)：
  - FPN 上行：[L30 Upsample](#L30) + [L31 Concat P4](#L31) → [L32 C2fAttn(512, 256, 8)](#L32) ★ v1 独有 attention
  - 下到 P3/8：[L34 Upsample](#L34) + [L35 Concat P3](#L35) → [L36 C2fAttn(256, 128, 4)](#L36)
  - [L38 ImagePoolingAttn(256)](#L38)：从 [15, 12, 9] 三尺度汇聚，**输出更新后的 `txt_feats`**——这是 v1 唯一的「文本反向更新」模块，也是无法 export ONNX 的元凶
  - PAN 下行：[L40-L46](#L40-L46) 三组 Conv s2 + Concat + `C2fAttn`
  - [L48 WorldDetect [nc, 512, False]](#L48)：`embed=512`、`with_bn=False`（L2 归一化）

## 与 v8 主线 yolov8.yaml 的对比

| 维度 | v8 yolov8.yaml | v1 yolov8-world.yaml |
|------|----------------|----------------------|
| 行数 | 46 | 48 |
| backbone | 10 层，Conv + C2f + SPPF | 同 v8（10 层一字不差） |
| head 层数 | 16 | 19（多 3 层 attention） |
| 融合模块 | `C2f` | `C2fAttn`（4 处） |
| 文本注入 | 无 | `C2fAttn` 4 处 + `ImagePoolingAttn` 1 处 |
| `txt_feats` 反向更新 | 无 | 有（L38，唯一位置） |
| 检测头 | `Detect [nc]` | `WorldDetect [nc, 512, False]` |
| cv3 输出维度 | `nc`（80） | `embed`（512） |
| 归一化 | 无（`sigmoid`） | L2 归一化（`ContrastiveHead`） |
| ONNX 导出 | ✅ | ❌（`ImagePoolingAttn` Python `for` 循环 + 切片） |

## 与 v2 配置 yolov8-worldv2.yaml 的对比

| 维度 | v1（本文件） | v2 |
|------|-------------|----|
| L32/L36/L42/L46 attention | `C2fAttn` | `C2fAttn`（保留） |
| L38 图像池化 | `ImagePoolingAttn(256)` | **删除** |
| L40 补层 | 无（L38 后直接 Conv s2） | `Conv [256, 3, 2]` 补一层 |
| 最终 `WorldDetect` | `[nc, 512, False]` | `[nc, 512, True]`（`with_bn=True`） |
| 对比头 | `ContrastiveHead`（L2 归一化） | `BNContrastiveHead`（BN 归一化） |
| 可重参数化 | ❌ | ✅（推理时 BN 折进 1×1 Conv） |
| ONNX 导出 | ❌ | ✅ |
| 行数 | 48 | 46 |

**核心叙事**：v1 用 `ImagePoolingAttn` 换精度上限，v2 用纯 BN 换部署可行性。逐层对比见 [架构页](../architecture)，方法学见 [concepts 页](../concepts)，损失见 [losses 页](../losses)。
