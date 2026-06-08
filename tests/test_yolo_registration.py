#!/usr/bin/env python3
"""
test_yolo_registration.py

Verifies that custom DINOv3 + SFP layers and the wrapped parse_model wrapper
are successfully registered within the Ultralytics namespace.
"""

import sys
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

def test_registration():
    print("🔬 Testing YOLO dynamic registration...")
    
    # Import registration helper
    from bcadfm.utils.yolo_utils import register_yolo_dino, custom_parse_model
    
    # Verify pre-registration state
    import ultralytics.nn.tasks
    pre_registered = hasattr(ultralytics.nn.tasks, "DinoV3Backbone")
    print(f"  - Before registration: Has DinoV3Backbone? {pre_registered} (Expected: False)")
    
    # Run registration
    register_yolo_dino()
    
    # Verify post-registration state
    post_registered = hasattr(ultralytics.nn.tasks, "DinoV3Backbone")
    print(f"  - After registration: Has DinoV3Backbone? {post_registered} (Expected: True)")
    
    has_sfp3 = hasattr(ultralytics.nn.tasks, "DinoV3SFP_P3")
    has_sfp4 = hasattr(ultralytics.nn.tasks, "DinoV3SFP_P4")
    has_sfp5 = hasattr(ultralytics.nn.tasks, "DinoV3SFP_P5")
    print(f"  - SFP components registered? P3: {has_sfp3}, P4: {has_sfp4}, P5: {has_sfp5} (Expected: True, True, True)")
    
    is_patched = ultralytics.nn.tasks.parse_model == custom_parse_model
    print(f"  - parse_model function patched globally? {is_patched} (Expected: True)")
    
    if post_registered and has_sfp3 and has_sfp4 and has_sfp5 and is_patched:
        print("\n✅ Dynamic Ultralytics registration test passed successfully!")
        return 0
    else:
        print("\n❌ Dynamic Ultralytics registration test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(test_registration())
