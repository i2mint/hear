"""
Some useful tools for audio manipulation
"""
from typing import Callable, Any
from functools import lru_cache
from dataclasses import dataclass

import soundfile as sf
from py2store.utils.affine_conversion import AffineConverter

from hear.util import Number, Waveform

AudioSource = Any


@lru_cache(maxsize=1)
def src_to_wf_and_indexer(
    src: AudioSource, index_to_seconds_scale, index_to_seconds_offset, src_to_wfsr,
):
    wf, sr = src_to_wfsr(src)
    indexer = AffineConverter(
        scale=sr * index_to_seconds_scale, offset=index_to_seconds_offset
    )
    return wf, indexer


@dataclass
class AudioSegments:
    """A class to access waveform segments through a obj[audio_src, bt:tt] interface
    - flexibility of audio source: specify how to get from src specification to a (wf, sr) pair
    - caches: if the same audio_src is asked multiple times in a row, it will keep in memory for faster access
    - specify what affine space to use to index segments (default would be samples but can specify seconds, utc, etc.)

    ```
    import soundfile as sf
    from hear.tools import AudioSegments

    ass = AudioSegments(src_to_wfsr=sf.read, index_to_seconds_scale=1 / 1e6)  # 1 / 1e6 to index wf in micro-seconds
    wf = ass[wav_filepath, 1056189.407:2112556.8]

    ```
    """

    src_to_wfsr: Callable[[AudioSource], Waveform] = sf.read
    index_to_seconds_scale: Number = 1
    index_to_seconds_offset: Number = 0

    def __getitem__(self, k):
        src, _slice = k
        wf, indexer = src_to_wf_and_indexer(
            # note: cached so repeated retrieval of same src will be efficient (already in memory)
            src,
            index_to_seconds_scale=self.index_to_seconds_scale,
            index_to_seconds_offset=self.index_to_seconds_offset,
            src_to_wfsr=self.src_to_wfsr,
        )
        bt, tt = _slice.start, _slice.stop
        if bt is not None:
            bt = int(indexer(bt))
        if tt is not None:
            tt = int(indexer(tt))
        return wf[bt:tt]
