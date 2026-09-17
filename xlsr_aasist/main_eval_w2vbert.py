"""Run the original evaluator with w2v-BERT 2.0 + AASIST."""
import sys

from model import model_w2vbert

sys.modules["model.model"] = model_w2vbert

import main_eval as _base


if __name__ == "__main__":
    _base.main()
