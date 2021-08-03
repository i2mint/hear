"""hear test util objects"""

from importlib_resources import files
import numpy as np
import soundfile as sf


data_files = files('hear') / 'tests' / 'data'

file_0123456789_wav = data_files / '0123456789.wav'
wf_0123456789 = np.arange(10).astype('int16')


def mk_test_files():
    file_obj = file_0123456789_wav
    n = 10
    dtype = 'int16'
    written_wf = np.arange(n).astype(dtype)
    written_sr = 100
    sf.write(file_obj, written_wf, written_sr)

    read_wf, read_sr = sf.read(file_obj, dtype=dtype)
    assert read_sr == written_sr
    assert np.all(read_wf == written_wf)
    assert np.all(read_wf == wf_0123456789)
