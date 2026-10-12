"""Reuse the pinned R2 interface_smoke implementation without modifying its producer."""
from w2v_v3183r2.interface_smoke import *

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    main(parser.parse_args().device)
