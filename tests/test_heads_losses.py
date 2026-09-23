"""Tests for heads and losses."""

import numpy as np
import pytest
import torch

from drstage.models.heads import CeHead, CornHead, build_head
from drstage.models.losses import (
    build_criterion,
    class_weights_from_counts,
    corn_loss,
)


class TestCornLoss:
    """Tests for CORN loss implementation."""

    def test_corn_hand_example_k3_zeros(self):
        """CORN with K=3, y=[0,2], logits zeros -> loss = ln(2)."""
        logits = torch.zeros(2, 2, dtype=torch.float32)
        targets = torch.tensor([0, 2], dtype=torch.long)
        num_classes = 3

        loss = corn_loss(logits, targets, num_classes)

        expected = float(np.log(2))
        assert abs(loss.item() - expected) < 1e-6

    def test_corn_hand_example_logits_values(self):
        """CORN with logits [[-2,5],[3,1]], y=[0,2] -> loss ≈ 0.16293."""
        logits = torch.tensor([[-2.0, 5.0], [3.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 2], dtype=torch.long)
        num_classes = 3

        loss = corn_loss(logits, targets, num_classes)

        expected = 0.16293
        assert abs(loss.item() - expected) < 1e-5

    def test_corn_logits_invariance(self):
        """CORN loss invariant to logits[0,1] when y[0] < 1."""
        logits1 = torch.tensor([[-2.0, 5.0], [3.0, 1.0]], dtype=torch.float32)
        logits2 = torch.tensor([[-2.0, 100.0], [3.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 2], dtype=torch.long)
        num_classes = 3

        loss1 = corn_loss(logits1, targets, num_classes)
        loss2 = corn_loss(logits2, targets, num_classes)

        assert abs(loss1.item() - loss2.item()) < 1e-6

    def test_corn_weighted(self):
        """CORN with sample weights."""
        logits = torch.zeros(2, 2, dtype=torch.float32)
        targets = torch.tensor([0, 2], dtype=torch.long)
        sample_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        num_classes = 3

        loss = corn_loss(logits, targets, num_classes, sample_weights)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_corn_float32_computation(self):
        """CORN computation always in float32."""
        logits = torch.tensor([[-2.0, 5.0]], dtype=torch.float16)
        targets = torch.tensor([0], dtype=torch.long)

        loss = corn_loss(logits, targets, num_classes=3)

        assert loss.dtype == torch.float32


class TestCornHeadDecode:
    """Tests for CORN head prediction/decoding."""

    def test_corn_class_probs_sum_to_one(self):
        """CORN class probs sum to 1 and are all >= 0."""
        head = CornHead(in_features=256, num_classes=5, dropout=0.0)

        logits = torch.randn(10, 4)
        grade, prob_dr, class_probs = head.predict(logits)

        assert torch.allclose(class_probs.sum(dim=1), torch.ones(10))
        assert (class_probs >= 0).all()
        assert (class_probs <= 1).all()

    def test_corn_grade_equals_cumulative_threshold(self):
        """CORN grade == count of P(y>k) > 0.5."""
        head = CornHead(in_features=256, num_classes=5, dropout=0.0)

        logits = torch.randn(20, 4)
        grade, _, _ = head.predict(logits)

        cum = torch.cumprod(torch.sigmoid(logits.float()), dim=1)
        expected_grade = (cum > 0.5).sum(dim=1).long()

        assert torch.allclose(grade, expected_grade)

    def test_corn_monotone_logit_invariance(self):
        """Increasing any logit never decreases grade."""
        head = CornHead(in_features=256, num_classes=5, dropout=0.0)

        logits_base = torch.randn(10, 4)
        grade_base, _, _ = head.predict(logits_base)

        for k in range(4):
            logits_inc = logits_base.clone()
            logits_inc[:, k] += 1.0
            grade_inc, _, _ = head.predict(logits_inc)

            assert (grade_inc >= grade_base).all()

    def test_corn_prob_dr_from_cumulative(self):
        """CORN prob_dr equals P(y>0) = sigmoid(logits[:,0])."""
        head = CornHead(in_features=256, num_classes=5, dropout=0.0)

        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        _, prob_dr, _ = head.predict(logits)

        expected = torch.sigmoid(logits[:, :1])

        assert torch.allclose(prob_dr, expected.squeeze(), atol=1e-6)


class TestCeHead:
    """Tests for CE head."""

    def test_ce_forward_shape(self):
        """CE forward returns [N, K] logits."""
        head = CeHead(in_features=256, num_classes=5, dropout=0.0)
        features = torch.randn(10, 256)

        logits = head(features)

        assert logits.shape == (10, 5)

    def test_ce_class_probs_sum_to_one(self):
        """CE class probs (softmax) sum to 1."""
        head = CeHead(in_features=256, num_classes=5, dropout=0.0)

        logits = torch.randn(10, 5)
        _, _, class_probs = head.predict(logits)

        assert torch.allclose(class_probs.sum(dim=1), torch.ones(10))

    def test_ce_grade_is_argmax(self):
        """CE grade is argmax of softmax probs."""
        head = CeHead(in_features=256, num_classes=5, dropout=0.0)

        logits = torch.randn(20, 5)
        grade, _, class_probs = head.predict(logits)

        expected_grade = torch.argmax(class_probs, dim=1).long()

        assert torch.allclose(grade.float(), expected_grade.float())

    def test_ce_prob_dr_from_softmax(self):
        """CE prob_dr = 1 - softmax[:,0]."""
        head = CeHead(in_features=256, num_classes=5, dropout=0.0)

        logits = torch.randn(10, 5)
        _, prob_dr, class_probs = head.predict(logits)

        expected = 1.0 - class_probs[:, 0]

        assert torch.allclose(prob_dr, expected)


class TestBuildHead:
    """Tests for head factory."""

    def test_build_head_corn(self):
        """build_head('corn') creates CornHead."""
        head = build_head("corn", in_features=256, num_classes=5, dropout=0.1)

        assert isinstance(head, CornHead)
        assert head.num_classes == 5
        assert head.out_dim == 4

    def test_build_head_ce(self):
        """build_head('ce') creates CeHead."""
        head = build_head("ce", in_features=256, num_classes=5, dropout=0.1)

        assert isinstance(head, CeHead)
        assert head.num_classes == 5
        assert head.out_dim == 5

    def test_build_head_invalid_kind(self):
        """build_head with invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown head kind"):
            build_head("invalid", in_features=256, num_classes=5)


class TestClassWeights:
    """Tests for class weight computation."""

    def test_class_weights_mean_one(self):
        """Class weights have mean == 1."""
        counts = [3355, 342, 2439, 129, 496]
        weights = class_weights_from_counts(counts, power=0.5)

        assert abs(weights.mean().item() - 1.0) < 1e-6

    def test_class_weights_ordering(self):
        """Class weights ordering matches expected pattern."""
        counts = [3355, 342, 2439, 129, 496]
        weights = class_weights_from_counts(counts, power=0.5)

        assert weights[3] > weights[1]
        assert weights[1] > weights[4]
        assert weights[4] > weights[2]
        assert weights[2] > weights[0]

    def test_class_weights_power_zero(self):
        """power=0 gives all ones."""
        counts = [10, 20, 30, 40, 50]
        weights = class_weights_from_counts(counts, power=0.0)

        assert torch.allclose(weights, torch.ones(5))

    def test_class_weights_all_same_count(self):
        """Equal counts give equal weights."""
        counts = [100, 100, 100, 100, 100]
        weights = class_weights_from_counts(counts, power=0.5)

        assert torch.allclose(weights, torch.ones(5))


class TestBuildCriterion:
    """Tests for criterion factory."""

    def test_build_criterion_corn(self):
        """build_criterion('corn') creates CornLoss."""
        from drstage.models.losses import CornLoss

        criterion = build_criterion("corn", class_weights=None, num_classes=5, label_smoothing=0.0)

        assert isinstance(criterion, CornLoss)

    def test_build_criterion_ce(self):
        """build_criterion('ce') creates WeightedCrossEntropy."""
        from drstage.models.losses import WeightedCrossEntropy

        criterion = build_criterion("ce", class_weights=None, num_classes=5, label_smoothing=0.1)

        assert isinstance(criterion, WeightedCrossEntropy)

    def test_build_criterion_with_weights(self):
        """build_criterion with class weights works."""
        weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5])

        criterion = build_criterion(
            "corn", class_weights=weights, num_classes=5, label_smoothing=0.0
        )

        assert criterion.class_weights is not None


class TestHeadInterfaceParity:
    """Tests for interface parity between CORN and CE heads."""

    def test_both_heads_predict_returns_three_tensors(self):
        """Both heads predict() returns (grade, prob_dr, class_probs)."""
        corn_head = CornHead(in_features=256, num_classes=5, dropout=0.0)
        ce_head = CeHead(in_features=256, num_classes=5, dropout=0.0)

        corn_logits = torch.randn(10, 4)
        ce_logits = torch.randn(10, 5)

        corn_grade, corn_dr, corn_probs = corn_head.predict(corn_logits)
        ce_grade, ce_dr, ce_probs = ce_head.predict(ce_logits)

        assert corn_grade.shape == ce_grade.shape == (10,)
        assert corn_dr.shape == ce_dr.shape == (10,)
        assert corn_probs.shape == ce_probs.shape == (10, 5)

    def test_both_heads_prob_dr_in_valid_range(self):
        """Both heads produce prob_dr in [0, 1]."""
        corn_head = CornHead(in_features=256, num_classes=5, dropout=0.0)
        ce_head = CeHead(in_features=256, num_classes=5, dropout=0.0)

        corn_logits = torch.randn(20, 4)
        ce_logits = torch.randn(20, 5)

        _, corn_dr, _ = corn_head.predict(corn_logits)
        _, ce_dr, _ = ce_head.predict(ce_logits)

        assert (corn_dr >= 0).all() and (corn_dr <= 1).all()
        assert (ce_dr >= 0).all() and (ce_dr <= 1).all()

    def test_both_heads_grades_in_valid_range(self):
        """Both heads produce grades in [0, 4]."""
        corn_head = CornHead(in_features=256, num_classes=5, dropout=0.0)
        ce_head = CeHead(in_features=256, num_classes=5, dropout=0.0)

        corn_logits = torch.randn(20, 4)
        ce_logits = torch.randn(20, 5)

        corn_grade, _, _ = corn_head.predict(corn_logits)
        ce_grade, _, _ = ce_head.predict(ce_logits)

        assert (corn_grade >= 0).all() and (corn_grade <= 4).all()
        assert (ce_grade >= 0).all() and (ce_grade <= 4).all()


class TestFactoryAndBackbone:
    """Tests for model factory and backbone."""

    def test_cam_target_layer_resnet18(self):
        """cam_target_layer on resnet18 returns layer4[-1]."""
        import timm

        from drstage.models.backbone import cam_target_layer

        backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        target = cam_target_layer(backbone, "resnet18")

        assert target is backbone.layer4[-1]

    def test_build_model_corn_forward_shape(self):
        """build_model with CORN head forward returns [N, K-1] logits."""
        from drstage.models.factory import build_model

        cfg = {
            "backbone": "resnet18",
            "head": "corn",
            "num_classes": 5,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
        }

        model = build_model(cfg, pretrained=False)
        x = torch.randn(4, 3, 64, 64)
        logits = model(x)

        assert logits.shape == (4, 4)

    def test_build_model_ce_forward_shape(self):
        """build_model with CE head forward returns [N, K] logits."""
        from drstage.models.factory import build_model

        cfg = {
            "backbone": "resnet18",
            "head": "ce",
            "num_classes": 5,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
        }

        model = build_model(cfg, pretrained=False)
        x = torch.randn(4, 3, 64, 64)
        logits = model(x)

        assert logits.shape == (4, 5)
