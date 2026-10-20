"""Distinguish optional report-host failures from training/artifact failures."""
from urllib.parse import urlparse


class ReportUploadError(RuntimeError):
    """A local report exists, but its optional remote delivery failed."""


def upload_report(archive, uploader):
    try:
        url = uploader(archive)
        parsed = urlparse(url)
        if (parsed.scheme != 'https' or parsed.netloc != 'temp.sh'
                or parsed.path in ('', '/', '/upload')):
            raise ValueError('Uploader returned an unexpected temporary download URL')
        return url
    except Exception as exc:
        raise ReportUploadError(str(exc)) from exc
