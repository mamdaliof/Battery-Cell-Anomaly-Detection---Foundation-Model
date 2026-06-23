import sys
import unittest
import tempfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# Add project scripts/ and src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from scripts.convert_kfold_to_classification import main as convert_cls_main
from scripts.convert_kfold_to_detection import main as convert_det_main


class TestKFoldConversion(unittest.TestCase):
    """Unit tests for convert_kfold_to_classification.py and convert_kfold_to_detection.py."""

    def setUp(self):
        # Create temp workspace
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source_root = self.temp_dir / "source"
        self.target_cls_root = self.temp_dir / "target_cls"
        self.target_det_root = self.temp_dir / "target_det"
        
        self.source_root.mkdir(parents=True)

        # Create mock 2-fold dataset structure
        # fold_0 and fold_1
        for f_idx in (0, 1):
            fold_dir = self.source_root / f"fold_{f_idx}"
            for split in ("train", "val"):
                split_dir = fold_dir / split
                (split_dir / "normal").mkdir(parents=True)
                (split_dir / "abnormal").mkdir(parents=True)

                # Create 1 normal sample
                img_normal = Image.new("RGB", (640, 640), color="white")
                img_normal.save(split_dir / "normal" / "img_norm.png")
                self.create_mock_xml(
                    split_dir / "normal" / "img_norm.xml",
                    objects=[("cell", 100, 100, 500, 500), ("text", 200, 200, 300, 250)]
                )

                # Create 1 abnormal sample
                img_abnormal = Image.new("RGB", (640, 640), color="black")
                img_abnormal.save(split_dir / "abnormal" / "img_abn.png")
                self.create_mock_xml(
                    split_dir / "abnormal" / "img_abn.xml",
                    objects=[("abnormal", 150, 150, 250, 250), ("cell", 50, 50, 600, 600)]
                )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_mock_xml(self, xml_path: Path, objects: list):
        """Create a mock Pascal VOC XML file with specified objects."""
        annotation = ET.Element("annotation")
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = "640"
        height = ET.SubElement(size, "height")
        height.text = "640"
        
        for name, xmin, ymin, xmax, ymax in objects:
            obj = ET.SubElement(annotation, "object")
            name_el = ET.SubElement(obj, "name")
            name_el.text = name
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin_el = ET.SubElement(bndbox, "xmin")
            xmin_el.text = str(xmin)
            ymin_el = ET.SubElement(bndbox, "ymin")
            ymin_el.text = str(ymin)
            xmax_el = ET.SubElement(bndbox, "xmax")
            xmax_el.text = str(xmax)
            ymax_el = ET.SubElement(bndbox, "ymax")
            ymax_el.text = str(ymax)

        tree = ET.ElementTree(annotation)
        tree.write(xml_path)

    def test_classification_conversion_kfold(self):
        """Verify convert_kfold_to_classification.py processes fold subfolders correctly."""
        import sys
        
        # Mock sys.argv
        sys.argv = [
            "convert_kfold_to_classification.py",
            "--source-root", str(self.source_root),
            "--target-root", str(self.target_cls_root),
            "--abnormal-labels", "abnormal",
            "--kfold"
        ]
        
        convert_cls_main()

        # Check target structure for both folds
        for f_idx in (0, 1):
            fold_dir = self.target_cls_root / f"fold_{f_idx}"
            self.assertTrue(fold_dir.exists())
            
            for split in ("train", "val"):
                self.assertTrue((fold_dir / split / "normal" / "img_norm.png").exists())
                self.assertTrue((fold_dir / split / "abnormal" / "img_abn.png").exists())
                # Should not have XML files copied in target classification dataset
                self.assertFalse((fold_dir / split / "normal" / "img_norm.xml").exists())

    def test_detection_conversion_kfold(self):
        """Verify convert_kfold_to_detection.py processes variants and fold folders correctly."""
        import sys
        
        # Mock sys.argv
        sys.argv = [
            "convert_kfold_to_detection.py",
            "--source-root", str(self.source_root),
            "--target-root", str(self.target_det_root),
            "--kfold"
        ]
        
        convert_det_main()

        # Check target variants
        variants = ["all", "no_cell", "abnormal_only"]
        for var in variants:
            var_dir = self.target_det_root / f"battery_detection_{var}"
            self.assertTrue(var_dir.exists())
            self.assertTrue((self.target_det_root / f"battery_detection_{var}.yaml").exists())

            # Check folds inside variant
            for f_idx in (0, 1):
                fold_dir = var_dir / f"fold_{f_idx}"
                self.assertTrue(fold_dir.exists())
                
                for split in ("train", "val"):
                    images_dir = fold_dir / "images" / split
                    labels_dir = fold_dir / "labels" / split
                    self.assertTrue(images_dir.exists())
                    self.assertTrue(labels_dir.exists())
                    
                    self.assertTrue((images_dir / "img_norm.png").exists())
                    self.assertTrue((images_dir / "img_abn.png").exists())
                    self.assertTrue((labels_dir / "img_norm.txt").exists())
                    self.assertTrue((labels_dir / "img_abn.txt").exists())


if __name__ == "__main__":
    unittest.main()
