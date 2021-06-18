"""
Defining codecs
"""
from dataclasses import dataclass
from typing import Iterable, Callable, Sequence, Union, Any, Protocol
from struct import pack, unpack
import struct
from operator import itemgetter

Chunk = Sequence[bytes]
Chunks = Iterable[Chunk]
ByteChunker = Callable[[bytes], Chunks]
Sample = Any  # but usually a number
Frame = Union[Sample, Sequence[Sample]]
Frames = Iterable[Frame]

Encoder = Callable[[Frames], bytes]
Decoder = Callable[[bytes], Iterable[Frame]]

ChunkToFrame = Callable[[Chunk], Frame]
FrameToChunk = Callable[[Frame], Chunk]

# class Codec(Protocol):
#     encode: Encoder
#     decoder: Decoder


@dataclass
class ChunkedEncoder(Encoder):
    frame_to_chk: FrameToChunk

    def __call__(self, frames: Frames):
        return b"".join(map(self.frame_to_chk, frames))


first_element = itemgetter(0)  # "equivalent" to lambda x: x[0]


@dataclass
class ChunkedDecoder(Decoder):
    chk_to_frame: ChunkToFrame
    n_channels: int = None
    chk_size_bytes: int = None

    # ByteChunker
    def chunker(self, b: bytes) -> Chunks:
        # TODO: Check efficiency against other byte-specific chunkers
        return map(bytes, zip(*([iter(b)] * self.chk_size_bytes)))

    def __call__(self, b: bytes):
        frames = map(self.chk_to_frame, self.chunker(b))
        if self.n_channels == 1:
            return map(first_element, frames)
        return frames


ChkFormat = str  # a str recognized as a struct format


def _split_chk_format(chk_format):
    if chk_format[0] in "@=<>!":
        return chk_format[0], chk_format[1:]
    return "", chk_format


def _format_chars_part_of_chk_format(chk_format):
    byte_order, format_chars = _split_chk_format(chk_format)
    return format_chars


def _chk_format_to_n_channels(chk_format):
    """
    >>> _chk_format_to_n_channels('hq')
    2
    >>> _chk_format_to_n_channels('@hq')
    2
    """
    return len(_format_chars_part_of_chk_format(chk_format))


def _chk_format_is_for_single_channel(chk_format):
    """Checks if a chk_format string is meant for a single channel codec"""
    return _chk_format_to_n_channels(chk_format) == 1


@dataclass
class StructCodecSpecs:
    r"""Enable the definition of codec specs based on format characters of the
    python struct module
    (https://docs.python.org/3/library/struct.html#format-characters)

    :param chk_format: The format of a chunk, as specified by the struct module
        See https://docs.python.org/3/library/struct.html#format-characters
    :param n_channels: Only n_channels = 1 serves a purpose; to indicate that

    >>> specs = StructCodecSpecs(
    ...     chk_format='h',
    ... )
    >>> print(specs)
    StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)
    >>>
    >>> encoder = ChunkedEncoder(
    ...     frame_to_chk=specs.frame_to_chk
    ... )
    >>> decoder = ChunkedDecoder(
    ...     chk_size_bytes=specs.chk_size_bytes,
    ...     chk_to_frame=specs.chk_to_frame,
    ...     n_channels=specs.n_channels
    ... )
    >>>
    >>> frames = [1, 2, 3]
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x02\x00\x03\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames


    >>> specs = StructCodecSpecs(
    ...     chk_format='@h',
    ...     n_channels = 2
    ... )
    >>> print(specs)
    StructCodecSpecs(chk_format='@hh', n_channels=2, chk_size_bytes=4)
    >>>
    >>>
    >>> encoder = ChunkedEncoder(
    ...     frame_to_chk=specs.frame_to_chk
    ... )
    >>> decoder = ChunkedDecoder(
    ...     chk_size_bytes=specs.chk_size_bytes,
    ...     chk_to_frame=specs.chk_to_frame,
    ...     n_channels=specs.n_channels
    ... )
    >>>
    >>> frames = [(1, 2), (3, 4), (5, 6)]
    >>>
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames
    """

    chk_format: str
    n_channels: int = None
    chk_size_bytes: int = None

    def __post_init__(self):
        inferred_n_channels = _chk_format_to_n_channels(self.chk_format)
        if self.n_channels is None:
            self.n_channels = inferred_n_channels
        else:
            assert (
                inferred_n_channels == 1
            ), "if n_channels is given, chk_format needs to be for a single channel"
            byte_order, format_chars = _split_chk_format(self.chk_format)
            self.chk_format = byte_order + format_chars * self.n_channels

        chk_size_bytes = struct.calcsize(self.chk_format)
        if self.chk_size_bytes is None:
            self.chk_size_bytes = chk_size_bytes
        else:
            assert self.chk_size_bytes == chk_size_bytes, (
                f"The given chk_size_bytes {self.chk_size_bytes} did not match the "
                f"inferred (from chk_format) {chk_size_bytes}"
            )

    def frame_to_chk(self, frame):
        if self.n_channels == 1:
            return pack(self.chk_format, frame)
        else:
            return pack(self.chk_format, *frame)

    def chk_to_frame(self, chk):
        return unpack(self.chk_format, chk)
