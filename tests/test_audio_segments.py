"""Real tests for `hear.tools.AudioSegments`.

Provenance: `hear/tests/test_tools.py` has always contained a function named
`audio_segment_demo_test`. pytest collects `test_*` (prefix), not `*_test`
(suffix), so that function has never run. When invoked by hand it raises
`TypeError`, because three of its assertions are malformed -- they call
`np.all(wf, [3, 4, 5])`, and the second positional argument of `np.all` is
`axis`, not a comparison target.

The behaviour it was *describing* is correct: every claim below was verified
against the library before being written down here. This module makes those
claims executable. The original file is deliberately left untouched -- see the
PR body -- so the decision to fix or retire it stays with the maintainer.

The fixture wav (`hear/tests/data/0123456789.wav`) holds
`np.arange(10).astype('int16')` at a sample rate of 100, i.e. one sample every
0.01s.
"""

from functools import partial

import numpy as np
import pytest
import soundfile

from hear.tests.test_util import file_0123456789_wav, wf_0123456789
from hear.tools import AudioSegments

SR = 100


@pytest.fixture
def read_int16():
    """`soundfile.read` curried to return the original int16 waveform."""
    return partial(soundfile.read, dtype="int16")


@pytest.fixture
def segs(read_int16):
    """AudioSegments yielding int16, indexed in seconds."""
    return AudioSegments(src_to_wfsr=read_int16)


def test_default_read_is_float():
    """The default `src_to_wfsr` is `soundfile.read`, which yields float64."""
    segs = AudioSegments()
    wf = segs[file_0123456789_wav]
    assert wf.dtype == np.float64
    # int16 samples are scaled by 1/2**15 on the way to float
    assert np.allclose(wf, np.arange(10) / 32768.0)


def test_int16_roundtrip(segs):
    """Currying `src_to_wfsr` gets the original int16 waveform back."""
    wf = segs[file_0123456789_wav]
    assert np.all(wf == wf_0123456789)
    assert np.all(wf == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_segment_by_triple_and_by_slice_agree(segs):
    """`[src, lo, hi]` and `[src, lo:hi]` are two spellings of one thing."""
    by_triple = segs[file_0123456789_wav, 0.03, 0.07]
    by_slice = segs[file_0123456789_wav, 0.03:0.07]
    assert np.all(by_triple == [3, 4, 5, 6])
    assert np.all(by_slice == [3, 4, 5, 6])
    assert np.all(by_triple == by_slice)


@pytest.mark.parametrize(
    "top_time, expected",
    [
        (0.07, [3, 4, 5, 6]),
        (0.07999999, [3, 4, 5, 6]),  # not quite 0.08 -> sample 7 not included
        (0.06999999, [3, 4, 5]),  # not quite 0.07 -> sample 6 not included
    ],
)
def test_top_index_truncates(segs, top_time, expected):
    """Top index truncates: just-shy-of-the-next-sample does not reach it."""
    assert np.all(segs[file_0123456789_wav, 0.03:top_time] == expected)


@pytest.mark.parametrize(
    "bottom_time, expected",
    [
        (0.03, [3, 4, 5, 6]),
        # Just shy of 0.03 lands at sample index 2.9999. Truncation keeps
        # sample 2; rounding would drop it. This case is what distinguishes
        # `int()` from `round()` in the bottom-time indexer -- without it the
        # two are indistinguishable, since 0.03 * sr is exactly 3.0.
        (0.029999, [2, 3, 4, 5, 6]),
        (0.025, [2, 3, 4, 5, 6]),
    ],
)
def test_bottom_index_truncates(segs, bottom_time, expected):
    """Bottom index truncates too -- it does not round to nearest."""
    assert np.all(segs[file_0123456789_wav, bottom_time:0.07] == expected)


def test_index_unit_can_be_samples(read_int16):
    """`index_to_seconds_scale=1/sr` makes the index unit samples."""
    samples_segs = AudioSegments(src_to_wfsr=read_int16, index_to_seconds_scale=1 / SR)
    assert np.all(samples_segs[file_0123456789_wav, 3:7] == [3, 4, 5, 6])
    # ... though indexing the whole waveform directly is the efficient way
    assert np.all(samples_segs[file_0123456789_wav][3:7] == [3, 4, 5, 6])


def test_offset_and_scale_shift_the_index_origin(read_int16):
    """A scale + offset lets the index be e.g. days since the Unix epoch.

    This also pins the float-precision caveat the original demo was written to
    document: the "natural" top time loses the last sample, and a couple of
    picoseconds more recovers it.
    """
    unix_segs = AudioSegments(
        src_to_wfsr=read_int16,
        index_to_seconds_scale=24 * 60 * 60,  # days -> seconds
        index_to_seconds_offset=18123.45,
    )
    lo, hi = 18123.450000347224, 18123.450000810186
    assert np.all(unix_segs[file_0123456789_wav, lo, hi] == [3, 4, 5])
    assert np.all(unix_segs[file_0123456789_wav, lo, hi + 2e-12] == [3, 4, 5, 6])
