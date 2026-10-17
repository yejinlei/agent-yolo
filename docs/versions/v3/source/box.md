# box.c

IoU/NMS 公共库：`box_iou`、`do_nms_sort` 供三个版本共用；**v3 保留了 v2 的 IoU 解析梯度实验函数**（`dunion`/`dintersect`），仍未被检测路径调用——CIoU 的数学在这里躺了两年，等 v4 接线。

> 快照路径：`source/yolov3/snapshot/src/box.c`。点击左侧行号可复制/分享行锚点；如 `#L179-L183` 可高亮多行区间。

<SourceViewer file="v3:src/box.c" />

**阅读路线**：[L6 nms_comparator 排序](/versions/v3/source/box#L6) → [L21 do_nms_obj 全类抑制](/versions/v3/source/box#L21) → [L58 do_nms_sort 按类抑制](/versions/v3/source/box#L58) → [L91 float_to_box](/versions/v3/source/box#L91) → [L152-186 overlap/IoU 家族](/versions/v3/source/box#L152-L186) → [L192-207 dintersect/dunion（IoU 梯度实验）](/versions/v3/source/box#L192-L207) → [L221-266 单元测试](/versions/v3/source/box#L221-L266) → [L319 do_nms](/versions/v3/source/box#L319) → [L339 encode/decode_box](/versions/v3/source/box#L339-L349)。
