import sys
import unittest
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

class TestEnvironment(unittest.TestCase):
    """
    Diagnostic test suite to verify the Python environment, CUDA configuration,
    and library dependencies required to run the Battery Cell Anomaly Detection pipeline.
    """

    def test_pytorch_installation(self):
        """
        Verify that PyTorch is installed and accessible.
        """
        try:
            import torch
            print(f"\n🔥 PyTorch version: {torch.__version__}")
            self.assertTrue(hasattr(torch, "__version__"))
        except ImportError:
            self.fail("PyTorch is not installed in the current environment.")

    def test_cuda_availability(self):
        """
        Check CUDA driver status and print GPU counts/names if visible.
        """
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n🖥️ CUDA Available: {cuda_available}")
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"    - GPU Count: {device_count}")
            for i in range(device_count):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("    - CUDA is NOT visible. Running in CPU-only mode.")

    def test_key_dependencies(self):
        """
        Verify that all mandatory third-party packages are installed.
        """
        dependencies = [
            ("transformers", "transformers"),
            ("peft", "peft"),
            ("accelerate", "accelerate"),
            ("ultralytics", "ultralytics"),
            ("sklearn", "scikit-learn"),
            ("yaml", "PyYAML"),
        ]
        
        print("\n📦 Checking dependencies:")
        for module_name, package_name in dependencies:
            with self.subTest(dependency=package_name):
                try:
                    module = __import__(module_name)
                    version = getattr(module, "__version__", "unknown")
                    print(f"    - [OK] {package_name} (Version: {version})")
                except ImportError:
                    self.fail(f"Required package '{package_name}' is not installed.")

    def test_local_module_imports(self):
        """
        Verify that the bcadfm library modules can be imported correctly.
        """
        modules_to_test = [
            "bcadfm.utils.config",
            "bcadfm.utils.yolo_utils",
            "bcadfm.data.dataset",
            "bcadfm.models.dinov3_classifier",
            "bcadfm.models.yolo_dino",
            "bcadfm.training.losses",
            "bcadfm.training.trainer",
            "bcadfm.training.yolo_trainer",
            "bcadfm.metrics.cls_metrics",
            "bcadfm.metrics.cls_callbacks",
        ]

        print("\n🔌 Checking bcadfm local modules imports:")
        for mod in modules_to_test:
            with self.subTest(module=mod):
                try:
                    __import__(mod)
                    print(f"    - [OK] Imported: {mod}")
                except ImportError as e:
                    self.fail(f"Failed to import local module '{mod}'. Error: {e}")

if __name__ == "__main__":
    unittest.main()
