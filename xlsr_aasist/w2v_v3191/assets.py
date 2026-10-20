"""Pinned public speech-denoising weight; no speech examples are downloaded."""
from pathlib import Path
import os
import urllib.request
from .common import digest, atomic_json

REVISION = "3eee541a283fd3b8f81b85b1748e3b9ccbefa04d"
SHA256 = "70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0"
URL = ("https://raw.githubusercontent.com/GregorR/rnnoise-models/" + REVISION +
       "/somnolent-hogwash-2018-09-01/sh.rnnn")


def prepare(cache, supplied=None, opener=urllib.request.urlopen):
    cache = Path(cache).expanduser().resolve()
    path = Path(supplied).expanduser().resolve() if supplied else cache/"sh.rnnn"
    if not path.is_file():
        if supplied:
            raise FileNotFoundError("Missing --rnnoise-model: " + str(path))
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache/f".sh.rnnn.{os.getpid()}.part"
        errors = []
        for attempt in range(3):
            print(f"[Assets] pinned RNNoise download {attempt+1}/3", flush=True)
            try:
                request = urllib.request.Request(URL, headers={"User-Agent": "RTC-SDD-V3191"})
                with opener(request, timeout=30) as stream:
                    data = stream.read(1024*1024)
                temporary.write_bytes(data)
                if digest(temporary) != SHA256:
                    raise ValueError("Pinned RNNoise download hash mismatch")
                temporary.replace(path)
                break
            except Exception as exc:
                errors.append(str(exc))
            finally:
                temporary.unlink(missing_ok=True)
        if not path.is_file():
            raise RuntimeError("RNNoise preparation failed; keep existing checkpoints. "
                               "Download the pinned URL and supply --rnnoise-model. " +
                               URL + "; " + "; ".join(errors))
    if digest(path) != SHA256:
        raise ValueError("RNNoise file does not match the registered public weight: " + str(path))
    provenance = dict(url=URL, revision=REVISION, sha256=SHA256,
                      purpose="Frozen speech denoising, never a spoof-trained teacher",
                      source="GregorR/rnnoise-models; somnolent-hogwash",
                      inference_applied=False)
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache/"provenance.json", provenance)
    return dict(rnnoise_model=str(path), rnnoise_sha256=SHA256,
                robust_asset_files={str(path): SHA256}, robust_asset_provenance=provenance)
