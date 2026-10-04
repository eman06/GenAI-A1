"""Hard routing (Task 2) and soft mixture-of-experts (Task 3) restoration systems."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardRoutedRestorer(nn.Module):
    """classifier -> argmax -> one specialist; class 0 (clean) is an identity bypass.

    route=None uses the classifier prediction (operational mode); passing the true labels
    gives oracle routing. Only the images routed to an expert are run through it.
    """

    def __init__(self, classifier, experts):
        super().__init__()
        self.classifier = classifier
        self.experts = nn.ModuleList(experts)   # order: salt_pepper, blur, occlusion -> classes 1, 2, 3

    def forward(self, x, route=None):
        probs = F.softmax(self.classifier(x).float(), dim=1)
        pred = probs.argmax(1)
        r = pred if route is None else route
        out = x.clone().float()
        for k, expert in enumerate(self.experts, start=1):
            m = r == k
            if m.any():
                out[m] = expert(x[m]).float()
        return out, probs, pred


class SoftMoERestorer(nn.Module):
    """x_hat = w0 * x + w1 A_salt(x) + w2 A_blur(x) + w3 A_occ(x),  w = softmax(G(x) / tau).

    The gate G has the classifier architecture (initialised from the Task 2 classifier),
    the experts are initialised from the Task 2 specialists. Fully differentiable.
    """

    def __init__(self, gate, experts, tau=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.tau = tau

    def forward(self, x):
        logits = self.gate(x).float()
        w = F.softmax(logits / self.tau, dim=1)
        branches = torch.stack([x.float()] + [e(x).float() for e in self.experts], dim=1)   # (B, 4, 3, H, W)
        out = (w[:, :, None, None, None] * branches).sum(1)
        return out, w, logits


class SoftMoEExport(nn.Module):
    """ONNX wrapper: returns (restored image, routing weights)."""

    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        out, w, _ = self.moe(x)
        return out.clamp(0, 1), w


class ClassifierExport(nn.Module):
    """ONNX wrapper: returns softmax probabilities."""

    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def forward(self, x):
        return F.softmax(self.clf(x), dim=1)
