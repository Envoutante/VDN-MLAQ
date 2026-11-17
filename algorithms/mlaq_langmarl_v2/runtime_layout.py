"""Fixed algorithm identity and shared output/config locations."""
from pathlib import Path
import os
import sys
ALGORITHM = 'mlaq_langmarl_v2'
VALUE_MODE = 'mlaq'
ALGORITHM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ALGORITHM_ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / 'data'
def bootstrap():
    # Local packages must precede other checkouts; shared tools follow them.
    for p in (str(PROJECT_ROOT), str(ALGORITHM_ROOT)):
        if p in sys.path: sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(ALGORITHM_ROOT))
def validate_epoch(epoch):
    parts = epoch.replace('\\', '/').split('/')
    if len(parts) != 2 or parts[0] != ALGORITHM or parts[1] in ('', '.', '..') or ':' in parts[1]:
        raise ValueError('epoch must be ' + ALGORITHM + '/<batch>; other versions cannot be overwritten')
    return '/'.join(parts)
