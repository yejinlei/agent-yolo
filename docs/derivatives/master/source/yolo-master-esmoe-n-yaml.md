# ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-coco.yaml

YOLO-Master ES-MoE n 档 COCO profile：**36 行、比 n/l 档更精简**，是 `yolo-master-esmoe-n-visdrone.yaml` 的「同构 COCO 版」。文件头 [L1-2](#L1-L2) 直接注释「Architecture-equivalent COCO profile of yolo-master-esmoe-n-visdrone.yaml」——两条 yaml 的模块序列逐行一致，只切换数据集（VisDrone 60 类 vs COCO 80 类）。

> 快照路径：`source/yolomaster/snapshot/ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-coco.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-coco.yaml" />

**阅读路线**：

- [L1-3 文件头注释](#L1-L3)：注明「ES-MoE-N COCO profile」；[L4 nc=80](#L4)；[L5-6 scales](#L5-L6)：`n: [0.50, 0.25, 1024]`——**与 yolo-master-n.yaml 的 n 档完全一致**
- 骨干 [L8-21](#L8-L21)：13 层——Conv/C3k2/ES_MOE/A2C2f 组合与 n/l 档完全一致：[L12 ES_MOE(256)](#L12)、[L15 ES_MOE(512)](#L15)、[L18 ES_MOE(512)](#L18)、[L21 ES_MOE(1024)](#L21)
- Head [L23-36](#L23-L36)：13 层 Upsample + Concat + C3k2 × 4 组，[L36 Detect([18, 21, 24], nc)](#L36) 收尾

## 与 yolo-master-n.yaml 的对比

| 维度 | yolo-master-n.yaml | yolo-master-esmoe-n-coco.yaml |
|------|---------------------|-------------------------------|
| 行数 | 45 | 36（精简注释） |
| scales n 档 | `[0.50, 0.25, 1024]` | **完全一致** |
| backbone 结构 | 4×ES_MOE + 2×A2C2f | 完全一致 |
| head 结构 | 13 层 Upsample/C3k2 | 完全一致 |
| Detect 索引 | [18, 21, 24] | 完全一致 |
| 用途 | 通用 n 档 | ES-MoE-n 命名的 COCO profile |
| 关联数据集 | COCO | VisDrone 的孪生 COCO 版 |

**这份 yaml 存在的意义**：Master 论文公开了两个数据集（COCO + VisDrone），但**只有一份 COCO profile 有明确命名**——`yolo-master-esmoe-n-coco.yaml` 就是把 visdrone 版的 scales + backbone + head 原封不动搬到 COCO，方便复现。这也是为什么 [痛点 3 §只公开 n/l 两档 + 1 个 esmoe-n](../painpoints#3-只公开-nl--1-个-esmoe-n-三个-yaml-pain-yaml-incomplete)：`yolo-master-esmoe-n-visdrone.yaml` 未在官方发布，`yolo-master-esmoe-n-coco.yaml` 却单独存在。

**结构一致性验证**：把 `yolo-master-n.yaml` 去掉所有 `#` 注释和空行，剩下的模块序列与本文件完全一致——**这两份 yaml 是同一张图**。用户若把 `yolo-master-esmoe-n-coco.yaml` 当作「Master 主推 n 档」使用，得到的模型结构与 `yolo-master-n.yaml` 完全等价。

架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)，痛点见 [painpoints 页](../painpoints)。

## 常见问题

**Q：这份 yaml 和 yolo-master-n.yaml 是完全一样吗？**
A：**结构完全一致**，唯一区别是注释更精简、注释头标注了「ES-MoE-N COCO profile」。两份 yaml 建出来的模型权重形状、FLOPs、参数都完全一致。

**Q：visdrone 版在哪里？**
A：`yolo-master-esmoe-n-visdrone.yaml` 未在官方 repo 中公开——这也是为什么 [痛点 3](../painpoints#3-只公开-nl--1-个-esmoe-n-三个-yaml-pain-yaml-incomplete) 存在。用户若要在 VisDrone 上训练，可把本文件的 `nc: 80` 改成 `nc: 60`。

**Q：Master 论文里提到 ES-MoE-n 是什么？**
A：ES-MoE-n 是论文里的命名——「Efficient Sparse Mixture-of-Experts n scale」，与 Master n 档结构等价，是同一个模型的两种命名。

**Q：为什么官方要单独发一份 esmoe-n-coco.yaml？**
A：推测是给论文复现用的 profile 文件——visdrone 版是主实验配置（[VisDrone 60 类](https://github.com/Ultralytics/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml)），COCO 版是同构的 COCO 复现版本。**命名差异只是文件头注释不同**，结构完全一致。

**Q：这份 yaml 可以直接作为 yolo-master-n 加载？**
A：可以——`YOLO("yolo-master-esmoe-n-coco.yaml")` 与 `YOLO("yolo-master-n.yaml")` 建出的模型结构完全等价。用户可根据文件名偏好选择加载。

**Q：nc=80 与 visdrone 60 的差异？**
A：nc 决定 Detect 输出的类别维度——COCO 80 类、VisDrone 60 类。修改 `nc` 只需改 [L4](#L4) 一处，其余结构完全独立。

**Q：COCO profile 训练命令？**
A：`yolo detect train data=coco128.yaml model=yolo-master-esmoe-n-coco.yaml epochs=100`——数据换成 COCO 完整版本即可（`data=coco.yaml`）。

**Q：官方 COCO 权重在哪下载？**
A：Master repo 的 `releases` 页面提供 `yolo-master-esmoe-n-coco.pt` 权重，可直接加载——`YOLO("yolo-master-esmoe-n-coco.pt")` 会自动读取同名的 yaml 结构。

**其他相关 yaml**：
- [yolo-master-n.yaml](./yolo-master-n-yaml)：通用 n 档（结构一致）
- [yolo-master-l.yaml](./yolo-master-l-yaml)：l 档结构
