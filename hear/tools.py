"""
Some useful tools for audio manipulation
"""
from typing import Callable, Any
from functools import lru_cache, wraps, partial
from dataclasses import dataclass
from io import BytesIO

import soundfile as sf

from hear.util import Number, Waveform

AudioSource = Any

# TODO: alternate indexing specified as indexing function is more open-closed
#   Note that offset has little use since usually key-dependent (here it is fixed)
#   indexing func could include (optinally) key to cover key-dependent offsets
@dataclass
class AudioSegments:
    """A class to access waveform segments through a obj[audio_src, bt:tt] interface
    - flexibility of audio source: specify how to get from src specification to a
        (wf, sr) pair
    - caches: if the same audio_src is asked multiple times in a row, it will keep in
        memory for faster access
    - specify what affine space to use to index segments (default would be samples but
        can specify seconds, utc, etc.)

    ```
    import soundfile as sf
    from hear.tools import AudioSegments

    segs = AudioSegments(src_to_wfsr=sf.read, index_to_seconds_scale=1 / 1e6)
    # Note: 1 / 1e6 to index wf in micro-seconds
    wf = segs[wav_filepath, 1056189.407, 2112556.8]
    # equivalently, use slices
    wf2 = segs[wav_filepath, 1056189.407:2112556.8]
    assert all(wf == wf2)
    ```

    Note: Keys can be [wf_key], [wf_key, bt:tt], or [wf_key, bt, tt].
    Warning: If your wf_key is a tuple, you cannot use [wf_key] since it will be
    confused with the other forms. In this case, do [wf_key,] or [tuple([wf_key])]

    Note: The default index unit is seconds. If you want to index by samples (which
    have less problems due to floating paint arithmetic), you CAN do
    ```
    segs = AudioSegments(..., index_to_seconds_scale=1 / sample_rate)
    ```
    but you should avoid this (unnecessary overhead of still float point arithmetic).
    Instead, you should just retrieve the whole waveform and index it directly:
    ```
    segs = AudioSegments(...)
    segs[wav_filepath][1000:1100]
    ```
    """

    src_to_wfsr: Callable[[AudioSource], Waveform] = sf.read
    index_to_seconds_scale: Number = 1
    index_to_seconds_offset: Number = 0

    def __getitem__(self, k):
        src, bt, tt = self._key_to_src_bt_tt(k)
        wf, indexer = src_to_wf_and_indexer(
            # note: cached so repeated retrieval of same src will be efficient
            # (already in memory)
            src,
            index_to_seconds_scale=self.index_to_seconds_scale,
            index_to_seconds_offset=self.index_to_seconds_offset,
            src_to_wfsr=self.src_to_wfsr,
        )
        if bt is not None:
            bt = int(indexer(bt))
        if tt is not None:
            tt = int(indexer(tt))
        return wf[bt:tt]

    @staticmethod
    def _key_to_src_bt_tt(k):
        """extract (src, bt, tt) from key (for use in __getitem__)"""
        if isinstance(k, tuple):
            if len(k) == 3:  # assume k is (wf_key, bt, tt)
                return k
            elif len(k) == 2:  # assume k is (wf_key, slice(bt, tt))
                wf_key, _slice = k
                return wf_key, _slice.start, _slice.stop
            elif len(k) == 1:
                return k[0], None, None
        else:
            return k, None, None


@lru_cache(maxsize=1)
def src_to_wf_and_indexer(
    src: AudioSource, index_to_seconds_scale, index_to_seconds_offset, src_to_wfsr,
):
    wf, sr = src_to_wfsr(src)
    indexer = AffineConverter(
        scale=sr * index_to_seconds_scale, offset=index_to_seconds_offset
    )
    return wf, indexer


def wf_func_to_wfsr_func(wf_func=None, *, sr):
    """Transform a wf returning function so it returns (wf, sr) pairs.

    Sometimes you don't have a function providing (wf, sr) in hand, but instead,
    a function providing wfs (and you know what the sr is).

    >>> wf_func = lambda n: list(range(n))
    >>> wf_func(4)
    [0, 1, 2, 3]

    To use AudioSegment, you'll need to provide a wfsr function though.
    To do so easily from your wf function, we give you wf_func_to_wfsr_func

    >>> wfsr_func = wf_func_to_wfsr_func(wf_func, sr=44100)
    >>> wfsr_func(4)
    ([0, 1, 2, 3], 44100)
    """
    if wf_func is None:
        return partial(wf_func_to_wfsr_func, sr=sr)

    @wraps(wf_func)
    def wfsr_func(*args, **kwargs):
        return wf_func(*args, **kwargs), sr

    return wfsr_func


class AffineConverter(object):
    """
    Getting a callable that will perform an affine conversion.
    Note, it does it as
        (val - offset) * scale
    (Note slope-intercept style (though there is the .from_slope_and_intercept constructor method for that)

    Inverse is available through the inv method, performing:
        val / scale + offset

    >>> convert = AffineConverter(scale=0.5, offset=1)
    >>> convert(0)
    -0.5
    >>> convert(10)
    4.5
    >>> convert.inv(4)
    9.0
    >>> convert.inv(4.5)
    10.0
    """

    def __init__(self, scale=1.0, offset=0.0):
        self.scale = scale
        self.offset = offset

    @classmethod
    def from_slope_and_intercept(cls, slope=1.0, intercept=0.0):
        cls(offset=-intercept / slope, scale=slope)

    def __call__(self, x):
        return (x - self.offset) * self.scale

    def inv(self, x):
        return x / self.scale + self.offset

    def map(self, seq):
        return (self(x) for x in seq)

    def invmap(self, seq):
        return (self.inv(x) for x in seq)
