"""
Defining codecs
"""
import struct
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
        return b''.join(map(self.frame_to_chk, frames))


first_element = itemgetter(0)


@dataclass
class ChunkedDecoder(Decoder):
    chk_size_bytes: int  # positive integer in fact TODO: builtin type for that?
    chk_to_frame: ChunkToFrame
    single_channel: bool = False

    # ByteChunker
    def chunker(self, b: bytes) -> Chunks:
        # TODO: Check efficiency against other byte-specific chunkers
        return map(bytes, zip(*([iter(b)] * self.chk_size_bytes)))

    def __call__(self, b: bytes):
        frames = map(self.chk_to_frame, self.chunker(b))
        if self.single_channel:
            return map(first_element, frames)
        return frames


ChkFormat = str  # a str recognized as a struct format


@dataclass
class StructCodecSpecs:
    """
    :param chk_format: The format of a chunk, as specified by the struct module
        See https://docs.python.org/3/library/struct.html#format-characters
    :param n_channels: Only n_channels = 1 serves a purpose; to indicate that
    """

    chk_format: str
    single_channel: bool = True

    def __post_init__(self):
        self.chk_size_bytes = struct.calcsize(self.chk_format)

    def frame_to_chk(self, frame):
        return pack(self.chk_format, frame)

    def chk_to_frame(self, chk):
        return unpack(self.chk_format, chk)


#
# class FrameCodec:
#     def __init__(self, frame_code, meta=None):
#         self.frame_code = frame_code
#         self.n_channels = len(self.frame_code)
#         self.meta = meta or {}

# def encode(self, frames):
#     for frame in frames:
#         pass
#
# def decode(self, b: bytes):
#     for chk in chunker(b):
#         pass
