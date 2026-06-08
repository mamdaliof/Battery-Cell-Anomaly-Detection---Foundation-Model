import sys
import os
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

class TestPipelines(unittest.TestCase):
    """
    Unit tests and mock execution wrappers to verify end-to-end processing pipelines
    and ablation configuration generators.

    Why We Have It:
    These tests ensure that command-line utilities (dataset conversion scripts, config grid
    sweeps generators) process files correctly, mapping XML bounding boxes and templates
    without failing.
    """

    def setUp(self):
        # Create temp workspace directory for pipeline file reads/writes
        self.workspace_dir = Path(tempfile.mkdtemp())
        
        # Bounding box labels indicate abnormality
        self.abnormal_label = "burnt"

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_dataset_conversion_pipeline(self):
        """
        Verify that convert_split_base_to_classification.py processes detection XML annotations
        and populates classification folders (normal/abnormal) correctly.

        How It Should Behave:
        A split containing normal files (no matching annotations) should end up in 'normal/',
        and files with matching bounding boxes (e.g. 'burnt') should end up in 'abnormal/'.
        """
        # Create mock detection source layout:
        # split_base/
        #   train/
        #     img1.png
        #     img1.xml (contains 'burnt' box)
        #     img2.png
        #     img2.xml (contains only 'ok' box)
        #   val/
        #     img1.png
        #     img1.xml (contains 'burnt' box)
        src_root = self.workspace_dir / "split_base"
        
        for split in ("train", "val"):
            split_dir = src_root / split
            split_dir.mkdir(parents=True)
            
            # 1. Abnormal Sample
            img1_path = split_dir / "img1.png"
            img1_path.touch()
            xml1_content = f"""<annotation>
                <object>
                    <name>{self.abnormal_label}</name>
                    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
                </object>
            </annotation>"""
            with open(split_dir / "img1.xml", "w") as f:
                f.write(xml1_content)

            # 2. Normal Sample
            img2_path = split_dir / "img2.png"
            img2_path.touch()
            xml2_content = """<annotation>
                <object>
                    <name>ok</name>
                    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
                </object>
            </annotation>"""
            with open(split_dir / "img2.xml", "w") as f:
                f.write(xml2_content)

        # Output target directory
        target_root = self.workspace_dir / "classification_data"

        # Import the script's main logic directly or execute as subprocess
        # We execute via subprocess to verify CLI parameters parser
        script_path = str(project_root / "scripts" / "convert_split_base_to_classification.py")
        
        cmd = [
            sys.executable,
            script_path,
            "--source-root", str(src_root),
            "--target-root", str(target_root),
            "--abnormal-labels", self.abnormal_label
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Script failed with output: {res.stderr}")

        # Check output folders structure
        self.assertTrue((target_root / "train" / "normal" / "img2.png").exists())
        self.assertTrue((target_root / "train" / "abnormal" / "img1.png").exists())
        self.assertTrue((target_root / "val" / "normal" / "img2.png").exists())
        self.assertTrue((target_root / "val" / "abnormal" / "img1.png").exists())
        
        # Verify counts
        self.assertEqual(len(list((target_root / "train" / "normal").glob("*.png"))), 1)
        self.assertEqual(len(list((target_root / "train" / "abnormal").glob("*.png"))), 1)

    def test_ablation_grid_generation_and_validation(self):
        """
        Verify that generate_ablation_grid.py generates unique, valid YAML configurations,
        and validate_ablation_configs.py parses them without errors.
        """
        # Create a configs/ folder in the temp workspace and copy template
        (self.workspace_dir / "configs").mkdir()
        shutil.copy(
            str(project_root / "configs" / "benchmark_baseline.yaml"),
            str(self.workspace_dir / "configs" / "benchmark_baseline.yaml")
        )
        
        # 1. Run grid generator CLI script with Cwd set to the temp workspace
        gen_script = str(project_root / "scripts" / "generate_ablation_grid.py")
        cmd_gen = [sys.executable, gen_script]
        
        res_gen = subprocess.run(cmd_gen, cwd=str(self.workspace_dir), capture_output=True, text=True)
        self.assertEqual(res_gen.returncode, 0, f"Grid generation failed: {res_gen.stderr}")
        
        # Verify that multiple YAML files were generated under configs/ablations/
        grid_out_dir = self.workspace_dir / "configs" / "ablations"
        yaml_files = list(grid_out_dir.glob("*.yaml"))
        self.assertTrue(len(yaml_files) > 0)

        # 2. Run validator script on the generated folder
        val_script = str(project_root / "scripts" / "validate_ablation_configs.py")
        cmd_val = [
            sys.executable,
            val_script,
            "--config-dir", str(grid_out_dir)
        ]
        
        res_val = subprocess.run(cmd_val, capture_output=True, text=True)
        self.assertEqual(res_val.returncode, 0, f"Ablation validation failed: {res_val.stderr}")

if __name__ == "__main__":
    unittest.main()
