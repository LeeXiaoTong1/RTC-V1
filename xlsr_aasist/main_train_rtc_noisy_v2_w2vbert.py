"""Run the exact V2 noisy-pair trainer with w2v-BERT 2.0 + AASIST."""
import sys

from model import model_w2vbert

sys.modules["model.model"] = model_w2vbert

import main_train_rtc_noisy_v2 as _base


if __name__ == "__main__":
    _base.main()
