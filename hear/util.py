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

dflt_subtype_for_sample_width = {2: 'PCM_16', 3: 'PCM_24', 4: 'PCM_32', 8: 'DOUBLE'}

# soundfile_signature not used yet, but intended for a future version of this module, that will use minting
# and signature injection instead of long copy pastes of
soundfile_signature = dict(
    dtype=DFLT_DTYPE, format=DFLT_FORMAT, subtype=None, endian=None
)


class SampleRateError(ValueError):
    """Error raised when something is wrong regarding the sample rate (sr)."""


class SampleRateAssertionError(SampleRateError):
    """Error raised when a sample rate's value is not the one that is expected/enforced."""


class SampleRateMissing(SampleRateError):
    """Error raised when a sample rate is missing (and might be required)."""


num_type_synonyms = [
    {
        'dtype': 'int16',
        'soundfile': 'PCM_16',
        'pyaudio': 'paInt16',
        'n_bits': 16,
        'n_bytes': 2,
        'numpy': np.int16,
        'struct': 'h',
    },
    {
        'dtype': 'int8',
        'soundfile': 'PCM_S8',
        'pyaudio': 'paInt8',
        'n_bits': 8,
        'n_bytes': 1,
        'numpy': np.int8,
        'struct': 'b',
    },
    {
        'dtype': 'int24',
        'soundfile': 'PCM_24',
        'pyaudio': 'paInt24',
        'n_bits': 24,
        'n_bytes': 3,
        'numpy': None,
        'struct': None,
    },
    {
        'dtype': 'int32',
        'soundfile': 'PCM_32',
        'pyaudio': 'paInt32',
        'n_bits': 32,
        'n_bytes': 4,
        'numpy': np.int32,
        'struct': 'i',
    },
    {
        'dtype': 'uint8',
        'soundfile': 'PCM_U8',
        'pyaudio': 'paUInt8',
        'n_bits': 8,
        'n_bytes': 1,
        'numpy': np.uint8,
        'struct': 'B',
    },
    {
        'dtype': 'float32',
        'soundfile': 'FLOAT',
        'pyaudio': 'paFloat32',
        'n_bits': 32,
        'n_bytes': 4,
        'numpy': np.float32,
        'struct': 'f',
    },
    {
        'dtype': 'float64',
        'soundfile': 'DOUBLE',
        'pyaudio': None,
        'n_bits': 64,
        'n_bytes': 8,
        'numpy': np.float64,
        'struct': 'd',
    },
]


def num_type_for(num, num_sys='n_bits', target_num_sys='soundfile'):
    """Translate from one (sample width) number type to another.

    :param num:
    :param num_sys:
    :param target_num_sys:
    :return:

    >>> num_type_for(16, "n_bits", "soundfile")
    'PCM_16'
    >>> num_type_for(np.array([1.0, 2.0, 3.0]).dtype, "numpy", "soundfile")
    'PCM_24'

    Tip: Use with `functools.partial` when you have some fix translation endpoints.

    >>> from functools import partial
    >>> get_dtype_from_n_bytes = partial(
    ...     num_type_for, num_sys="n_bytes", target_num_sys="dtype"
    ... )
    >>> get_dtype_from_n_bytes(8)
    'float64'
    """
    for d in num_type_synonyms:
        if num == d[num_sys]:
            if target_num_sys in d:
                return d[target_num_sys]
            else:
                raise ValueError(
                    f'Did not find any {target_num_sys} entry for {num_sys}={num}'
                )
    raise ValueError(f'Did not find any entry for {num_sys}={num}')
