"""
Grad-CAM for the DenseNet121 classifier. Hooks the last conv layer
(features.norm5 output / the last denseblock) to produce a heatmap showing
which region of the X-ray drove the prediction for a given class.
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import eval_transform, CONDITIONS
from model import build_model


class GradCAM:
    """
    Grad-CAM using a forward hook + retain_grad() rather than a backward
    hook. DenseNet applies an in-place ReLU immediately after the layer we
    hook, which conflicts with register_full_backward_hook's autograd
    wrapping (PyTorch raises a "view is being modified in-place" error).
    retain_grad() sidesteps this entirely and is the more robust approach.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output
        self.activations.retain_grad()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        gradients = self.activations.grad
        # Global-average-pool the gradients -> channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, torch.sigmoid(logits)[0, class_idx].item()


def overlay_heatmap(original_pil_image, cam, alpha=0.4):
    """Resize cam to the original image size and blend it in as a heatmap."""
    img = np.array(original_pil_image.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def region_description(cam):
    """
    Rough quadrant description of where the CAM is concentrated, used later
    by the report generator (e.g. 'lower right lung field').
    """
    h, w = cam.shape
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    vertical = "upper" if y < h / 2 else "lower"
    # Note: X-rays are typically displayed with patient's left on the viewer's
    # right -- adjust this mapping if your dataset convention differs.
    horizontal = "right" if x < w / 2 else "left"
    return f"{vertical} {horizontal} lung field"


def run(image_path, checkpoint="models/chest_classifier.pt", class_idx=None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    conditions = ckpt.get("conditions", CONDITIONS)
    model = build_model(num_classes=len(conditions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_layer = model.features.norm5  # last norm layer before pooling

    image = Image.open(image_path).convert("L")
    input_tensor = eval_transform(image).unsqueeze(0)
    input_tensor.requires_grad_(False)

    if class_idx is None:
        with torch.no_grad():
            probs = torch.sigmoid(model(input_tensor))[0]
        class_idx = int(torch.argmax(probs).item())

    cam_engine = GradCAM(model, target_layer)
    cam, confidence = cam_engine.generate(input_tensor, class_idx)
    overlay = overlay_heatmap(image, cam)
    region = region_description(cam)

    print(f"Predicted: {conditions[class_idx]} (confidence {confidence:.2f})")
    print(f"Region of concern: {region}")
    overlay.save("outputs/gradcam_overlay.png")
    print("Saved overlay to outputs/gradcam_overlay.png")

    return overlay, conditions[class_idx], confidence, region


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=False)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint", default="models/chest_classifier.pt")
    args = parser.parse_args()

    if args.image:
        run(args.image, checkpoint=args.checkpoint)
    elif args.test:
        print("Pass --image path/to/xray.png to test Grad-CAM on a real file.")