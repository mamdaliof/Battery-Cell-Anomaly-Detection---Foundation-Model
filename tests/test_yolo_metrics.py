import unittest
from pathlib import Path
import numpy as np
import torch

from bcadfm.training.yolo_trainer import CustomDetectionValidator


class MockMetrics:
    """Mock class mimicking DetMetrics class in ultralytics."""
    def __init__(self, names):
        self.names = names
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.ap_class_index = np.array(list(names.keys()))
        self.box = self
        self.p = np.array([0.9, 0.8, 0.7])
        self.r = np.array([0.85, 0.75, 0.65])
        self.f1 = np.array([0.87, 0.77, 0.67])
        self.all_ap = np.array([[0.88, 0.80], [0.78, 0.70], [0.68, 0.60]])

    def process(self, *args, **kwargs):
        return {k: np.concatenate(v, 0) for k, v in self.stats.items()}

    def update_stats(self, stat):
        pass

    def clear_stats(self):
        pass

    @property
    def results_dict(self):
        return {}

class TestCustomDetectionValidator(unittest.TestCase):
    """Verifies that the CustomDetectionValidator correctly matches boxes,

    converts classification labels, and logs the metrics dict structure.
    """

    def setUp(self):
        from ultralytics.cfg import get_cfg
        args = get_cfg(overrides=dict(plots=False, save_json=False, save_txt=False, conf=0.25))

        self.names = {0: "abnormal", 1: "cell", 2: "text"}
        self.validator = CustomDetectionValidator(args=args)
        self.validator.seen = 0

        self.validator.names = self.names
        self.validator.metrics = MockMetrics(self.names)
        self.validator.save_dir = Path(".")


    def test_coordinate_matching_and_classification(self):
        # 1. Mock predictions [xmin, ymin, xmax, ymax, conf, class_idx]
        # Image 1 has abnormal (index 0) and cell (index 1) predictions
        pred1 = torch.tensor([
            [10.0, 10.0, 50.0, 50.0, 0.90, 0.0],
            [100.0, 100.0, 200.0, 200.0, 0.85, 1.0]
        ])
        
        # Image 2 has text (index 2) prediction
        pred2 = torch.tensor([
            [50.0, 50.0, 80.0, 80.0, 0.80, 2.0]
        ])

        # 2. Mock batch ground truths
        # batch['bboxes'] shape (N, 4) in absolute xyxy format
        # batch['cls'] contains class indices
        # batch['batch_idx'] is image index in batch
        batch = {
            "bboxes": torch.tensor([
                [10.0, 10.0, 50.0, 50.0],    # Image 0 - abnormal
                [100.0, 100.0, 200.0, 200.0], # Image 0 - cell
                [50.0, 50.0, 80.0, 80.0]      # Image 1 - text
            ]),
            "cls": torch.tensor([
                [0.0],
                [1.0],
                [2.0]
            ]),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "im_file": ["img0.png", "img1.png"],
            "ori_shape": [(640, 640), (640, 640)]
        }

        # Override validator prep methods to return our local formats
        self.validator._prepare_batch = lambda si, b: {
            "cls": b["cls"][b["batch_idx"] == si].squeeze(-1),
            "bboxes": b["bboxes"][b["batch_idx"] == si],
            "im_file": b["im_file"][si],
            "ori_shape": b["ori_shape"][si]
        }
        self.validator._prepare_pred = lambda pred: {
            "cls": pred[:, 5],
            "conf": pred[:, 4],
            "bboxes": pred[:, :4]
        }

        # Run update
        self.validator.update_metrics([pred1, pred2], batch)

        # Assert correct classification ground truths
        # Image 0 has abnormal (1), text (0)
        # Image 1 has abnormal (0), text (1)
        self.assertEqual(self.validator.cls_gt_abnormal, [1, 0])
        self.assertEqual(self.validator.cls_gt_text, [0, 1])

        # Assert correct classification predictions
        self.assertEqual(self.validator.cls_pred_abnormal, [1, 0])
        self.assertEqual(self.validator.cls_pred_text, [0, 1])

        # Verify bbox IoUs and Dice counts (3 matched pairs)
        self.assertEqual(len(self.validator.custom_iou_dice_stats), 3)
        for class_idx, iou, dice in self.validator.custom_iou_dice_stats:
            self.assertIn(class_idx, [0, 1, 2])
            iou_val = iou.item() if isinstance(iou, torch.Tensor) else iou
            dice_val = dice.item() if isinstance(dice, torch.Tensor) else dice
            self.assertAlmostEqual(iou_val, 1.0)
            self.assertAlmostEqual(dice_val, 1.0)

        # Assert cell (class 1) metrics
        cell_idx = 1
        self.assertEqual([int(x.item() if isinstance(x, torch.Tensor) else x) for x in self.validator.cls_gt[cell_idx]], [1, 0])
        self.assertEqual([int(x.item() if isinstance(x, torch.Tensor) else x) for x in self.validator.cls_pred[cell_idx]], [1, 0])
        probs = [float(x.item() if isinstance(x, torch.Tensor) else x) for x in self.validator.cls_prob[cell_idx]]
        self.assertAlmostEqual(probs[0], 0.85, places=4)
        self.assertAlmostEqual(probs[1], 0.0, places=4)

    def test_get_stats_formatting(self):
        # Seed lists to ensure validation calculations can run
        self.validator.cls_gt = {0: [], 1: [], 2: []}
        self.validator.cls_pred = {0: [], 1: [], 2: []}
        self.validator.cls_prob = {0: [], 1: [], 2: []}

        self.validator.cls_gt_abnormal = [1, 0, 1, 0]
        self.validator.cls_pred_abnormal = [1, 0, 0, 1]
        self.validator.cls_prob_abnormal = [0.9, 0.1, 0.2, 0.85]

        self.validator.cls_gt_text = [0, 1, 0, 1]
        self.validator.cls_pred_text = [0, 1, 1, 0]
        self.validator.cls_prob_text = [0.1, 0.8, 0.7, 0.2]

        self.validator.custom_iou_dice_stats = [(0, 0.8, 0.88), (2, 0.9, 0.94)]

        # Mock standard get_stats to return empty dict
        self.validator.metrics.stats = dict(
            tp=[np.array([[True]])],
            conf=[np.array([0.9])],
            pred_cls=[np.array([0])],
            target_cls=[np.array([0])],
            target_img=[np.array([0])]
        )

        stats = self.validator.get_stats()

        # Check for matching custom bbox IoUs
        self.assertIn("metrics/custom_mean_bbox_IoU", stats)
        self.assertAlmostEqual(stats["metrics/custom_mean_bbox_IoU"], 0.85)
        self.assertAlmostEqual(stats["metrics/custom_mean_bbox_Dice"], 0.91)

        # Check for image classification metrics
        self.assertIn("metrics/custom_cls_accuracy/abnormal", stats)
        self.assertIn("metrics/custom_cls_f1/text", stats)
        self.assertEqual(stats["metrics/custom_cls_accuracy/abnormal"], 0.50)


if __name__ == "__main__":
    unittest.main()
