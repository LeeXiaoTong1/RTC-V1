"""Run the original baseline trainer with the w2v-BERT 2.0 frontend."""
import sys

from model import model_w2vbert

# main_train imports Model lazily inside main(); redirect that import to the
# w2v-BERT wrapper while leaving the original V2 files untouched.
sys.modules["model.model"] = model_w2vbert

import main_train as _base


if __name__ == "__main__":
    _base.main()
