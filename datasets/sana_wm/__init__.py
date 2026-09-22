"""SANA-WM data contracts and strict local reader.

Reader symbols are loaded lazily so manifest-only tooling does not import
OpenCV on headless hosts.
"""

from datasets.sana_wm.manifest import Manifest, ManifestRecord, load_manifest

__all__ = [
    "CameraData",
    "Manifest",
    "ManifestRecord",
    "SANAReader",
    "SANASample",
    "SampleReadError",
    "load_manifest",
]


def __getattr__(name: str):
    if name in {"CameraData", "SANAReader", "SANASample", "SampleReadError"}:
        from datasets.sana_wm import reader

        return getattr(reader, name)
    raise AttributeError(name)
