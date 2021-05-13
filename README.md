# hear
https://github.com/otosense/hear


To install:	```pip install hear```


# Examples


A wav serialization/deserialization transformer.

First let's make a very short waveform.

```pydocstring
>>> from hear import WavSerializationTrans
>>> from numpy import sin, arange, pi
>>> n_samples = 5; sr = 44100;
>>> wf = sin(arange(n_samples) * 2 * pi * 440 / sr)
>>> wf
array([0.        , 0.06264832, 0.12505052, 0.18696144, 0.24813785])
```

An instance of ``WavSerializationTrans`` will allow you to

```pydocstring
>>> trans = WavSerializationTrans(assert_sr=sr)  # if you want to write data you NEED to specify assert_sr
>>> wav_bytes = trans._data_of_obj(wf)
>>> wav_bytes[:44]  # the header bytes
b'RIFF.\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\n\x00\x00\x00'
>>> wav_bytes[44:]  # the data bytes (5 * 2 = 10 bytes)
b'\x00\x00\x04\x08\x01\x10\xee\x17\xc2\x1f'

>>> wf_read_from_bytes = trans._obj_of_data(wav_bytes)
>>> wf_read_from_bytes
array([   0, 2052, 4097, 6126, 8130], dtype=int16)
```


Note that we've serialized floats, but they were deserialized as int16.
This is the default behavior, but is cusomizable through dtype, subtype, etc.
With this default dtype=int16 setting though, if you serialize int16 arrays, you'll recover them exactly.

```pydocstring
>>> assert all(trans._obj_of_data(trans._data_of_obj(wf_read_from_bytes)) == wf_read_from_bytes)
```

The most common use of WavSerializationTrans through, is to make a class decorator for a store that
provides wav bytes.

```pydocstring
>>> @WavSerializationTrans.wrapper(assert_sr=sr)
... class MyWavStore(dict):
...     pass
>>> my_wav_store = MyWavStore(just_one=wav_bytes)
>>> my_wav_store['just_one']
array([   0, 2052, 4097, 6126, 8130], dtype=int16)
```