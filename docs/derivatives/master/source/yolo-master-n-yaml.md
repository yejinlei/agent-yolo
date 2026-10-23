# ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml

YOLO-Master n 档的完整结构配置：**45 行**（nc + scales + backbone 13 层 + head 13 层）。与 v12 的骨干相比，Master 在 P2–P5 每一层之后都插入 `ES_MOE`——**4 处 ES_MOE + 2 处 A2C2f**是这条链路的骨架；head 沿用 v12/v11 标准的 Upsample + Concat + C3k2 双向融合，最终由 `Detect` 三尺度收尾。

> 快照路径：`source/yolomaster/snapshot/ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml" />

**阅读路线**：

- [L4 nc=80](#L4)：COCO 默认 80 类；[L5-7 scales 复合缩放](#L5-L7)：`[depth, width, max_channels]` 三参数——**n 档 `[0.50, 0.25, 1024]`**，官方摘要 344 层 / 2,694,364 参数 / 8.8 GFLOPs
- 骨干 [L10-24](#L10-L24)：
  - [L12-13 Conv s2 ×2](#L12-L13)：64 → 128（P2/4）
  - [L14 C3k2(256, False, 0.25)](#L14) → [L15 ES_MOE(256)](#L15)：第一处 MoE，接在浅层 C3k2 之后
  - [L16-18 P3/8](#L16-L18)：Conv s2 → C3k2(512) → ES_MOE(512)
  - [L19-21 P4/16](#L19-L21)：Conv s2 → **A2C2f(512, True, 4)** → ES_MOE(512)——A2C2f 是 v12 引入的 Area-Attention 分支
  - [L22-24 P5/32](#L22-L24)：Conv s2 → A2C2f(1024, True, 1) → ES_MOE(1024)——最深 4 层重复，最后一层再次 ES_MOE 收尾
- Head [L27-44](#L27-L44)：
  - FPN 上行：[L28 Upsample + L29 Concat 骨干 P4](#L28-L29) → [L30 C3k2(512)](#L30)；[L32-34](#L32-L34) 二次上采样到 P3/8
  - PAN 下行：[L36-38](#L36-L38) 下到 P4；[L40-42](#L40-L42) 下到 P5
  - [L44 Detect([18, 21, 24], nc)](#L44)：三尺度融合，索引分别对应 P3/8、P4/16、P5/32

## 与 v12 yolo12n.yaml 的对比

| 维度 | v12 yolov12n.yaml | Master n 档 |
|------|-------------------|-------------|
| 行数 | ~46 行 | 45 行 |
| 骨干核心 | C3k2 + A2C2f + Conv | C3k2 + A2C2f + Conv **+ ES_MOE × 4** |
| MoE | 无 | 4 处 ES_MOE（P2–P5 各一） |
| Head | Upsample/Concat + C3k2 + Detect | 完全一致 |
| Scales | `[0.33, 0.50, 1024]` | `[0.50, 0.25, 1024]`（width 更瘦） |
| 官方 GFLOPs | 6.9 | 8.8 |

**关键差异**：Master n 档的 depth=0.50 比 v12 n 档的 0.33 更大，width=0.25 比 v12 的 0.50 更小——**层数更多、通道更瘦**，配合 4×ES_MOE 的稀疏计算达到类似总 FLOPs 但更强的表达能力。

**yaml 与建网的耦合**：`parse_model` 遇到 `ES_MOE` 走独立 moe 分支并把 `legacy` 置为 False，Detect 会自动走 DW-separable cv3（[tricks 页 §Head 双分支](../tricks#head-双分支)）。`A2C2f` 与 `A2C2fMoE` 在 yaml 里写法不同——本页用的是 v12 原版 `A2C2f`，MoE 只体现在独立的 ES_MOE 层。

## 常见问题

**Q：为什么骨干里 ES_MOE 之后没有 Conv？**
A：ES_MOE 内部已经包含 BN + SiLU（[modules.py 源码页](./moe-modules-py)），输出直接接下一层的 Conv s2 就够——不需要额外的过渡层。这也是 ES_MOE 相比其他 MoE 的接口简洁之处。

**Q：yaml 里 `True` / `False` 是什么意思？**
A：`A2C2f(512, True, 4)` 中的 `True` 是 `a2` 参数——是否启用 Area Attention 分支；`False` 表示回退到 C3k。`0.25` 是 C3k2 的 `e` 参数（hidden channel 比例）。

**Q：Detect 索引为什么是 [18, 21, 24]？**
A：0-25 层的编号，18 是 head 里第一个 C3k2 层（对应 P3/8），21 是第二个 C3k2（对应 P4/16），24 是最后一个 C3k2（对应 P5/32）。这三层是 head 的最终输出，Detect 直接从它们读取三尺度特征。

**Q：为什么 head 的 C3k2 不传 `True`？**
A：head 的 C3k2 参数是 `[512, True]`——这里的 `True` 是 C3k2 的 `shortcut` 参数（默认 False），不是 `a2`。C3k2 与 A2C2f 是**两个独立的类**，签名不同。

**Q：如何加载这个 yaml？**
A：`from ultralytics import YOLO; model = YOLO("yolo-master-n.yaml")`——`yaml_model_load` 会先尝试找 `yolo-master.yaml`（不存在），失败后 fallback 到 `guess_model_scale` 用 `-master-n` 后缀选 n 档。

架构总览见 [architecture 页](../architecture)，损失集成见 [losses 页](../losses)，方法学见 [concepts 页](../concepts)，痛点见 [painpoints 页](../painpoints)。

**其他相关 yaml**：
- [yolo-master-l.yaml](./yolo-master-l-yaml)：l 档结构
- [yolo-master-esmoe-n.yaml](./yolo-master-esmoe-n-yaml)：COCO profile
- Master MoE 主模块：[moe/modules.py](./moe-modules-py)
