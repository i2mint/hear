"""
hear utilities
"""
from typing import Sequence, Union, Any, Iterable, Callable
import numpy as np

Number = Union[int, float, np.number]
Sample = Number
Waveform = Sequence[Sample]

DFLT_DTYPE = 'int16'
DFLT_FORMAT = 'WAV'
DFLT_N_CHANNELS = 1

# TODO: Do some validation and smart defaults with these
dtype_from_sample_width = {
    1: 'int16',
    2: 'int16',
    3: 'int32',
    4: 'int32',
    8: 'float64',
}

sample_width_for_soundfile_subtype = {
    'DOUBLE': 8,
    'FLOAT': 4,
    'G721_32': 4,
    'PCM_16': 2,
    'PCM_24': 3,
    'PCM_32': 4,
    'PCM_U8': 1,
}

# soundfile_signature not used yet, but intended for a future version of this module, that will use minting
# and signature injection instead of long copy pastes of
soundfile_signature = dict(
    dtype=DFLT_DTYPE, format=DFLT_FORMAT, subtype=None, endian=None
)


class SampleRateAssertionError(ValueError):
    ...


soundfile_signature = dict(
    dtype=DFLT_DTYPE, format=DFLT_FORMAT, subtype=None, endian=None
)
