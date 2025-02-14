import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(sys.path[0].removesuffix('tests'), 'src'))

import exttorch