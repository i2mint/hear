"""Test tools.py"""
from functools import partial
import soundfile as sf
import numpy as np

from hear.tools import AudioSegments
from hear.tests.test_util import data_files, wf_0123456789, file_0123456789_wav


def audio_segment_demo_test():
    # Note to reader: wf_0123456789 = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int16)
    #   and file_0123456789_wav is a path to that waveform saved as a wav file
    #   with samplerate (sr) of 100

    # The file_0123456789_wav wav file was obtained by serializing the
    # waveform wf_0123456789 = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int16)
    # with samplerate (sr) of 100.
    # If we ask AudioSegment to fetch it, it will give us an array of floats.
    # This is because the default `src_to_wfsr` function is soundfile.read,
    # which provides waveforms in float64 by default.
    from hear.tools import AudioSegments

    segs = AudioSegments()
    float_version_of_wf_0123456789 = np.array(
        [
            0.00000000e00,
            3.05175781e-05,
            6.10351562e-05,
            9.15527344e-05,
            1.22070312e-04,
            1.52587891e-04,
            1.83105469e-04,
            2.13623047e-04,
            2.44140625e-04,
            2.74658203e-04,
        ]
    )
    assert np.allclose(segs[file_0123456789_wav], float_version_of_wf_0123456789)

    # If you want to get your waveform in int16 (as the original), you'll need
    # to specify your own `src_to_wfsr` function.
    # One way to do this is to "curry" soundfile.read to your needs.
    read_int16_wfsr = partial(sf.read, dtype='int16')
    segs = AudioSegments(src_to_wfsr=read_int16_wfsr)
    # See that now you get the original waveform back!
    np.all(segs[file_0123456789_wav] == wf_0123456789)
    np.all(segs[file_0123456789_wav] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # The raise d'etre of AudioSegment is to provide segments of the waveform.
    # You can do so with the [src, bottom_time, top_time] format
    assert np.all(segs[file_0123456789_wav, 0.03, 0.07] == [3, 4, 5, 6])
    # ... but also the [src, bottom_time:top_time] format
    assert np.all(segs[file_0123456789_wav, 0.03:0.07] == [3, 4, 5, 6])
    # Note that by default the indexing unit is in seconds.
    # Since the sample rate is 100, this means that there's a sample every 0.01s.
    # Note that the indices are rounded down, so if you're slightly before the next 0.01
    # you're not there yet!
    assert np.all(segs[file_0123456789_wav, 0.03:0.07999999] == [3, 4, 5, 6])
    assert np.all(segs[file_0123456789_wav, 0.03:0.06999999] == [3, 4, 5])

    # Again, the default indexing unit is seconds, but you can change that if you want.
    # If, for instance, your need to refer to your sound in days since 1970 (Unix epoch),
    # and you know your file's first byte is timestamped 18123.45.
    # You would specify
    #     - index_to_seconds_scale = 1 / 24 * 60 * 60  # to convert from days to seconds
    #     - index_to_seconds_offset = 18123.45  # offset of file
    unix_segs = AudioSegments(
        src_to_wfsr=read_int16_wfsr,
        index_to_seconds_scale=24 * 60 * 60,  # to convert from days to seconds
        index_to_seconds_offset=18123.45,  # offset of file
    )

    seconds_to_relative_unix_days = lambda x: 18123.45 + x / (60 * 60 * 24)
    assert (
        seconds_to_relative_unix_days(0.03),
        seconds_to_relative_unix_days(0.07) == (18123.450000347224, 18123.450000810186),
    )

    wf = unix_segs[file_0123456789_wav, 18123.450000347224, 18123.450000810186]
    assert np.all(wf, [3, 4, 5])
    # Ah! you were expecting [3, 4, 5, 6], and you are right to.
    # But computers, and with their floats, fail to represent real numbers.
    # A bit more on the top time and we get it:
    wf = unix_segs[
        file_0123456789_wav, 18123.450000347224, (18123.450000810186 + 2e-12)
    ]
    assert np.all(wf, [3, 4, 5, 6])

    # So beware when using floats for indexing, if it matters whether you loose a
    # sample or not.

    # What if you wanted to index by sample offset (to avoid the float problems
    # mentioned above)? You could do this:
    sr = 100
    samples_segs = AudioSegments(
        src_to_wfsr=read_int16_wfsr,
        index_to_seconds_scale=1 / sr,  # to convert to samples unit
    )
    assert np.all(samples_segs[file_0123456789_wav, 3:7] == [3, 4, 5, 6])
    # But this would be inefficient, and could still lead to float problems
    # Instead, you should just retrieve the whole waveform and index it directly
    wf = samples_segs[file_0123456789_wav][3:7]
    assert np.all(wf, [3, 4, 5, 6])
