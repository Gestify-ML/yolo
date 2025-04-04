import torch
from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore
from ultralytics.utils import LOGGER  # type: ignore


class PrunedDectectionTrainer(DetectionTrainer):
    def setup_model(self):
        super().setup_model()

        # Extract masks from model
        self.masks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # Create a mask where weights are non-zero
                mask = (torch.abs(module.weight) > 1e-6).float()
                self.masks[name] = mask

    def apply_pruned_mask(self):
        """Apply the masks to block gradients of pruned weights during training."""
        for name, module in self.model.named_modules():
            if (
                isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
                and name in self.masks
            ):
                if module.weight.grad is not None:
                    mask = self.masks[name].to(module.weight.device)
                    if mask.size() != module.weight.grad.size():
                        raise ValueError(
                            f"Mask and gradient size mismatch in layer {name}: "
                            f"mask {mask.size()}, grad {module.weight.grad.size()}"
                        )
                    module.weight.grad *= mask

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        self.apply_pruned_mask()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        pass
