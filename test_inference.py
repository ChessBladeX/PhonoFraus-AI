import os
import sys

# Ensure photoFrausAI package directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
photofraus_dir = os.path.join(root_dir, "photoFrausAI")
if photofraus_dir not in sys.path:
    sys.path.insert(0, photofraus_dir)

from test_inference import test_inference

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_inference(image_path=img_arg)
