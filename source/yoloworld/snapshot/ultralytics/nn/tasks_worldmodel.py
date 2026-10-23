# --- Extracted from ultralytics/nn/tasks.py (commit e9187c12968c1242a2d4dc5f4396cc0e8bb403fb) ---
# YOLOv8-World 定义类：包含 txt_feats、CLIP 编码、text-guided forward。
# 在原仓库中位于 ultralytics/nn/tasks.py:558-646。

class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


