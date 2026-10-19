# models/yolo.py

YOLOv7 的检测头与建模器：**IDetect（带隐式知识的默认头）+ IAuxDetect（lead+aux 双头）+ Model/parse_model**。v5 式的 Detect 仅作兼容保留，yolov7.yaml 实际用的是 IDetect。

> 快照路径：`source/yolov7/snapshot/models/yolo.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v7:models/yolo.py" />

**阅读路线**：

- 原版头：[L23-94 Detect](#L23-L94)——v3 式结构，[L56-57 解码公式](#L56-L57)：`xy=(σ·2-0.5+grid)·stride`、`wh=(σ·2)²·anchor_grid`
- 默认头 [L97-208 IDetect](#L97-L208)（yolov7.yaml 末行所用）：
  - [L116-117 ia/im](#L116-L117)：每个尺度包一层 ImplicitA + ImplicitM
  - [L119-138 forward](#L119-L138)：`self.m[i](self.ia[i](x[i]))` 后再 `self.im[i](x[i])`，解码仍是 v3 式
  - [L140-176 fuseforward](#L140-L176)：融合后的前向（含 ONNX 导出分支）
  - [L178-190 fuse](#L178-L190)：**ImplicitA 经 matmul 并进卷积 bias、ImplicitM 乘进 weight/bias**——论文 §4.3"推理零开销 freebie"的代码落点
- 姿态头：[L210 IKeypoint](#L210)
- 辅助头 [L311-430 IAuxDetect](#L311-L430)（train_aux.py 用）：[L328-332 头部构造](#L328-L332)——lead 头 `m`+`ia`/`im`，**aux 头 [L329 m2](#L329) 只有纯 1×1 卷积、无隐式知识**（partial auxiliary head 的代码体现）；[L362 推理只回 lead](#L362)（`x[:self.nl]`）
- bin 回归头：[L433 IBin](#L433)
- 建模与解析：[L508 Model](#L508)（[L633 _initialize_biases](#L633)、[L643 _initialize_aux_biases](#L643)、[L693 fuse](#L693)）、[L736 parse_model](#L736)（[L752-762 模块白名单](#L752-L762)、[L790 检测头注册](#L790)）

## 与 v5 models/yolo.py 的对比

| 维度 | v5 yolo.py | v7 yolo.py |
|------|------------|------------|
| 检测头 | Detect 一种 | Detect/IDetect/IAuxDetect/IBin/IKeypoint 五种 |
| 隐式知识 | 无 | IDetect 每层 ia/im，可 fuse 进卷积参数 |
| 双头训练 | 无 | IAuxDetect：lead 带隐式知识、aux 纯 1×1，推理丢弃 aux |
| 解码公式 | v3 式 | 同 v3 式（论文所述"距离限制 objectness"约束未见于代码） |
| bias 初始化 | _initialize_biases | 另有 _initialize_aux_biases / bin / kpt 版 |
| parse_model | Conv/C3/SPP/Focus 白名单 | RepConv/Swin 系等大扩展白名单 |

辅助头的标签分配策略见 [utils/loss.py](/versions/v7/source/loss-py) 的 ComputeLossAuxOTA。
