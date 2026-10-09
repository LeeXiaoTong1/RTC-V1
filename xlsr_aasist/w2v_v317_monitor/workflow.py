"""Original V3.17 training, with a full Train evaluation after each saved epoch."""
from .hooks import attach


def main():
    from w2v_v317 import workflow
    from .archive import export_report
    original=workflow.export_report;workflow.export_report=export_report
    try:
        with attach():workflow.main()
    finally:workflow.export_report=original


if __name__=='__main__':main()
