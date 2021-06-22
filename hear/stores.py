"""
Useful stores (dols; Data Object Layers) for audio storage management.
"""
from io import BytesIO
from py2store.stores.local_store import LocalBinaryStore
from dol.trans import add_wrapper_method
from dol.signatures import Sig, call_forgivingly


import soundfile as sf

from hear.util import (
    DFLT_DTYPE,
    DFLT_FORMAT,
    DFLT_N_CHANNELS,
    SampleRateAssertionError,
    SampleRateMissing,
)


class ReadAudioFileMixin:
    read_kwargs = {}

    def _obj_of_data(self, data):
        return sf.read(BytesIO(data), **self.read_kwargs)


class PcmSerializationTrans:
    def __init__(
        self,
        sr,
        channels=DFLT_N_CHANNELS,
        dtype=DFLT_DTYPE,
        format='RAW',
        subtype='PCM_16',
        endian=None,
    ):
        assert isinstance(sr, int), 'sr must be an int'
        self.sr = sr
        self._rw_kwargs = dict(
            samplerate=sr,
            channels=channels,
            dtype=dtype,
            format=format,
            subtype=subtype,
            endian=endian,
        )

    def _obj_of_data(self, data):
        return sf.read(BytesIO(data), **self._rw_kwargs)[0]


@add_wrapper_method
class WfSrSerializationTrans:
    """An audio serialization/deserialization transformer object whose external objects are (
    waveform, sample_rate)
    pairs, and internal data some bytes-encoding of the latter.

    See WavSerializationTrans for explanations and doctest examples.
    """

    # _read_format = DFLT_FORMAT
    # _rw_kwargs = dict(dtype=DFLT_DTYPE, subtype=None, endian=None)

    def __init__(self, dtype=DFLT_DTYPE, format=DFLT_FORMAT, subtype=None, endian=None):
        self._r_kwargs = dict(dtype=dtype, subtype=subtype, endian=endian)
        self._w_kwargs = dict(format=format, subtype=subtype, endian=endian)

    def _obj_of_data(self, data):
        return sf.read(BytesIO(data), **self._r_kwargs)

    def _data_of_obj(self, obj):
        wf, sr = obj
        b = BytesIO()
        sf.write(b, wf, samplerate=sr, **self._w_kwargs)
        b.seek(0)
        return b.read()


from typing import Optional


@add_wrapper_method
class WfsrToWfWithSrAssertionTrans:
    """Asserts a fixed specified sr (if given) and returns only the wf part of the (wf, sr) data"""

    def __init__(self, assert_sr: Optional[int] = None):
        assert assert_sr is None or isinstance(assert_sr, int), (
            f'assert_sr must be None or an integer. Was ' f'{assert_sr}'
        )
        self.assert_sr = assert_sr

    def wfsr_to_wf_with_sr_assertion(self, wfsr):
        wf, sr = wfsr
        if self.assert_sr != sr:
            if (
                self.assert_sr is not None
            ):  # Putting None check here because less common, so more efficient on avg
                raise SampleRateAssertionError(
                    f'sr was {sr}, should be {self.assert_sr}'
                )
        return wf

    def _obj_of_data(self, data):
        return self.wfsr_to_wf_with_sr_assertion(data)

    def _data_of_obj(self, obj):
        if self.assert_sr is None:
            raise SampleRateMissing(
                f'To write data you need to specify an assert_sr sample rate'
            )
        return obj, self.assert_sr


@add_wrapper_method
class WfSerializationTrans(WfSrSerializationTrans):
    """An audio serialization object like WfSrSerializationTrans, but working with waveforms
    only (sample rate fixed).

    See WavSerializationTrans for explanations and doctest examples.
    """

    def __init__(
        self,
        assert_sr=None,
        dtype=DFLT_DTYPE,
        format=DFLT_FORMAT,
        subtype=None,
        endian=None,
    ):
        super().__init__(dtype, format, subtype, endian)
        self.assert_sr = assert_sr

    def _obj_of_data(self, data):
        wf, sr = super()._obj_of_data(data)
        if self.assert_sr is not None and sr != self.assert_sr:
            raise SampleRateAssertionError(
                f'{self.assert_sr} expected but I encountered {sr}'
            )
        return wf

    def _data_of_obj(self, obj):
        if self.assert_sr is None:
            raise SampleRateMissing(
                f'To write data you need to specify an assert_sr sample rate'
            )
        return super()._data_of_obj((obj, self.assert_sr))


# TODO: Make this with above elements:
#  For example, with @WfsrToWfWithSrAssertionTrans.wrapper(assert_sr=None)
@add_wrapper_method
class WavSerializationTrans(WfSrSerializationTrans, WfsrToWfWithSrAssertionTrans):
    r"""A wav serialization/deserialization transformer.

    First let's make a very short waveform.

    >>> from numpy import sin, arange, pi
    >>> n_samples = 5; sr = 44100;
    >>> wf = sin(arange(n_samples) * 2 * pi * 440 / sr)
    >>> wf
    array([0.        , 0.06264832, 0.12505052, 0.18696144, 0.24813785])

    An instance of ``WavSerializationTrans`` will allow you to

    >>> trans = WavSerializationTrans(assert_sr=sr)

    If you want to write data you NEED to specify assert_sr

    >>> wav_bytes = trans._data_of_obj(wf)
    >>> wav_bytes[:44]  # the header bytes
    b'RIFF.\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\n\x00\x00\x00'

    >>> wav_bytes[44:]  # the data bytes (5 * 2 = 10 bytes)
    b'\x00\x00\x04\x08\x01\x10\xee\x17\xc2\x1f'

    >>> wf_read_from_bytes = trans._obj_of_data(wav_bytes)
    >>> wf_read_from_bytes
    array([   0, 2052, 4097, 6126, 8130], dtype=int16)

    Note that we've serialized floats, but they were deserialized as int16.
    This is the default behavior, but is cusomizable through dtype, subtype, etc.
    With this default dtype=int16 setting though, if you serialize int16 arrays, you'll recover
    them exactly.

    >>> assert all(trans._obj_of_data(trans._data_of_obj(wf_read_from_bytes)) == wf_read_from_bytes)

    The most common use of WavSerializationTrans through, is to make a class decorator for a
    store that
    provides wav bytes.

    >>> @WavSerializationTrans.wrapper(assert_sr=sr)
    ... class MyWavStore(dict):
    ...     pass
    >>> my_wav_store = MyWavStore(just_one=wav_bytes)
    >>> my_wav_store['just_one']
    array([   0, 2052, 4097, 6126, 8130], dtype=int16)

    """

    def __init__(
        self, assert_sr=None, dtype=DFLT_DTYPE, format='WAV', subtype=None, endian=None,
    ):
        WfsrToWfWithSrAssertionTrans.__init__(self, assert_sr=assert_sr)
        WfSrSerializationTrans.__init__(
            self, dtype=dtype, format=format, subtype=subtype, endian=endian
        )

    def _obj_of_data(self, data):
        wf, sr = super()._obj_of_data(data)
        return WfsrToWfWithSrAssertionTrans._obj_of_data(self, (wf, sr))

    def _data_of_obj(self, obj):
        wf, sr = WfsrToWfWithSrAssertionTrans._data_of_obj(self, obj)
        return super()._data_of_obj((wf, sr))


PcmSerializationMixin = PcmSerializationTrans  # alias for back-compatibility
WfSrSerializationMixin = WfSrSerializationTrans  # alias for back-compatibility
WavSerializationMixin = WavSerializationTrans  # alias for back-compatibility
WfSerializationMixin = WfSerializationTrans  # alias for back-compatibility


class WavLocalFileStore(WavSerializationTrans, LocalBinaryStore):
    def __init__(
        self,
        path_format,
        assert_sr=None,
        max_levels=None,
        dtype=DFLT_DTYPE,
        format='WAV',
        subtype=None,
        endian=None,
    ):
        LocalBinaryStore.__init__(self, path_format, max_levels)
        WavSerializationTrans.__init__(
            self,
            assert_sr=assert_sr,
            dtype=dtype,
            format=format,
            subtype=subtype,
            endian=endian,
        )


WavLocalFileStore2 = WavLocalFileStore  # back-compatibility alias

from py2store.slib.s_zipfile import FilesOfZip, ZipStore
from dol import filt_iter


def has_wav_extension(string):
    return string.endswith('.wav') and not string.startswith('__MACOSX/')


@filt_iter(filt=has_wav_extension)
class ZipOfWavs(FilesOfZip):
    """Only lists .wav files and provides values as bytes.

    To get numerical waveform, use `WavSerializationTrans.wrapper`

    >>> @WavSerializationTrans.wrapper(assert_sr=44100)
    ... class WfOfZipOfWavs(ZipOfWavs):
    ...     pass
    """


@WfSrSerializationTrans.wrapper()
class WfSrOfZipOfWavs(ZipOfWavs):
    """A KvReader that provides the (wf, sr) pairs of the .wav files in a zip file"""


@WfSrSerializationTrans.wrapper()
class WfsrZipStore(ZipStore):
    """Read and write (waveform, sample_rate) pairs in a zip file's files.

    >>> from tempfile import gettempdir
    >>> import os
    >>>
    >>> rootdir = gettempdir()
    >>>
    >>> # preparation
    >>> test_zipfile = os.path.join(rootdir, 'zipstore_test_file.zip')
    >>> if os.path.isfile(test_zipfile):
    ...     os.remove(test_zipfile)
    >>> assert not os.path.isfile(test_zipfile)

    Okay, test_zipfile doesn't exist (but will soon...)

    >>> z = WfsrZipStore(test_zipfile)

    See that the file still doesn't exist (it will only be created when we start writing)

    >>> assert not os.path.isfile(test_zipfile)
    >>> list(z)  # z "is" empty (which makes sense?)
    []
    >>> import numpy as np
    >>> z['foo.wav'] = (np.array([0, 1, -2, 3], np.int16), 44100)
    >>> list(z)
    ['foo.wav']
    >>> z['foo.wav']
    (array([ 0,  1, -2,  3], dtype=int16), 44100)
    """


# TODO: Need to give init access to assert_sr (and more?)
@WfSerializationTrans.wrapper()
class WfZipStore(ZipStore):
    """Read and write waveforms in a zip file's files.

    # >>> from tempfile import gettempdir
    # >>> import os
    # >>>
    # >>> rootdir = gettempdir()
    # >>>
    # >>> # preparation
    # >>> test_zipfile = os.path.join(rootdir, 'zipstore_test_file.zip')
    # >>> if os.path.isfile(test_zipfile):
    # ...     os.remove(test_zipfile)
    # >>> assert not os.path.isfile(test_zipfile)
    #
    # Okay, test_zipfile doesn't exist (but will soon...)
    #
    # >>> z = WfZipStore(test_zipfile, assert_sr=44100)
    #
    # See that the file still doesn't exist (it will only be created when we start writing)
    #
    # >>> assert not os.path.isfile(test_zipfile)
    # >>> list(z)  # z "is" empty (which makes sense?)
    # []
    # >>> import numpy as np
    # >>> z['foo.wav'] = np.array([0, 1, -2, 3], np.int16)
    # >>> list(z)
    # ['foo.wav']
    # >>> z['foo.wav']
    # array([ 0,  1, -2,  3], dtype=int16)
    #
    # Notice that the sample rate was neither written, no read.
    # But if you played `foo.wav` it would reflect the 44100 sample rate you specified when
    # constructing the store.
    #
    # You can also see this sample rate if you use a `Wfsr...` store to read the zip file:
    #
    # >>> zz = WfsrZipStore(test_zipfile)
    # >>> z['foo.wav']
    # (array([ 0,  1, -2,  3], dtype=int16), 44100)
    """

    # @Sig.from_objs(
    #         # lambda assert_sr=None: None,
    #         ZipStore,
    #         WfSerializationTrans)
    # def __init__(self, *args, **kwargs):
    #     # call_forgivingly(super().__init__, self, *args, **kwargs)
    #     call_forgivingly(ZipStore.__init__, self, *args, **kwargs)
    #     call_forgivingly(WfSerializationTrans.__init__, self, *args, **kwargs)


def _length_and_sr_of_wavs(z):
    """A dataframe containing information about the wavfiles of the files in the wfsr store
    Note: Don't depend on this yet -- it's in motion.
    """
    import pandas as pd

    def gen():
        for k, (wf, sr) in z.items():
            yield {'file': k, 'n_samples': len(wf), 'sr': sr}

    df = pd.DataFrame(list(gen()))
    df = df.set_index('file')
    df['duration_s'] = df['n_samples'] / df['sr']
    return df


from py2store.stores.local_store import MakeMissingDirsStoreMixin
import os


class PcmSourceSessionBlockStore(MakeMissingDirsStoreMixin, LocalBinaryStore):
    sep = os.path.sep
    path_depth = 3

    def _id_of_key(self, k):
        raise DeprecationWarning('Deprecated')
        assert len(k) == self.path_depth
        return super()._id_of_key(self.sep.join(self.path_depth * ['{}']).format(*k))

    def _key_of_id(self, _id):
        raise DeprecationWarning('Deprecated')
        key = super()._key_of_id(_id)
        return tuple(key.split(self.sep))
