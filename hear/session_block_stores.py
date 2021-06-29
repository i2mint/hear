"""
Stores that specialize in block storage;
That is, where continuous signal is stored over multiple blocks of data.
"""
from sortedcontainers import SortedList
from collections.abc import Iterable, Mapping
from functools import lru_cache, cached_property
import soundfile as sf
from io import BytesIO
import json
import os
from math import ceil

# from ocore.utils import utime

from py2store.stores.local_store import RelativePathFormatStore
from py2store.mixins import ReadOnlyMixin, IterBasedSizedContainerMixin
from py2store.parse_format import match_re_for_fstring

DFLT_TIME_UNITS_PER_SEC = int(1e6)


class LocalPathStore(ReadOnlyMixin, RelativePathFormatStore):
    """A RelativePathFormatStore, but with write/delete functionality disabled"""

    def head(self):
        return next(iter(self.items()))


def wf_of_block_bytes(block_bytes, sr):
    return sf.read(
        BytesIO(block_bytes),
        dtype='int16',
        channels=1,
        samplerate=sr,
        format='RAW',
        subtype='PCM_16',
    )[0]


def int_if_not_none(x):
    if x is not None:
        return int(x)
    else:
        return None


class SessionBlockStore(LocalPathStore):
    def __init__(self, channel_data_dir, sr=None, **kwargs):
        # figuring out the sample rate
        channel_store = LocalPathStore(channel_data_dir)
        if 'config.json' in channel_store:
            config = json.loads(channel_store['config.json'])
            self.sr = int(config.get('sr', sr))
        else:
            self.sr = sr
        assert (
            self.sr is not None
        ), "I couldn't figure out the sample rate. Should be in a config.json file or given explicitly as argument"

        # Here we make our path inclusion condition more specific, in order to avoid including any files that
        # are not "block" files that are compliant with the folder structure.
        # Namely, we'll only take filepath show structure is CHANNEL_ROOT/s/SESSION/b/BLOCK
        # and further, constrain valid SESSION and BLOCK names to be exactly 16 digits.
        # By tagging the session and block segments in the path_format, we're also enabling the construction of
        # a regular expression that can be used to extract information from our keys.
        # See the get_session_and_block for how it can be used.
        path_format = os.path.join(channel_data_dir, 's/{session:16d}/b/{block:16d}')
        self._path_match_re = match_re_for_fstring(path_format)

        super().__init__(path_format, **kwargs)

    def get_session_and_block(self, k):
        """Extract (session, block) tuple"""
        m = self._path_match_re.search(self._id_of_key(k))
        if m:
            d = m.groupdict()
            return (
                int_if_not_none(d.get('session', None)),
                int_if_not_none(d.get('block', None)),
            )
        else:
            return None, None


class WfStore(SessionBlockStore):
    def __init__(self, channel_data_dir, sr=None):
        super().__init__(channel_data_dir=channel_data_dir, sr=sr, mode='b')

    def __getitem__(self, k):
        block_bytes = super().__getitem__(k)  # the raw bytes
        return wf_of_block_bytes(block_bytes, sr=self.sr)


class TxyzStore(SessionBlockStore):
    @staticmethod
    def df_from_json_lines(data):
        import pandas as pd

        return pd.DataFrame(
            list(map(json.loads, filter(None, data.split('\n'))))
        ).set_index('timestamp')

    def __init__(self, channel_data_dir, sr=None):
        super().__init__(channel_data_dir=channel_data_dir, sr=sr)

    def __getitem__(self, k):
        block_bytes = super().__getitem__(k)  # the raw bytes
        return self.df_from_json_lines(block_bytes)


class DictKeyMap:
    def __init__(self, id_of_key):
        self.__id_of_key = id_of_key.get
        self.__key_of_id = {_id: k for k, _id in id_of_key.items()}.get

    def _id_of_key(self, k):
        return super()._id_of_key(self.__id_of_key(k))

    def _key_of_id(self, _id):
        return self.__key_of_id(super()._key_of_id(_id))


inf = float('infinity')


# TODO: The class below isn't general at all. It assumes, for instance microseconds blocks.
# TODO: Need to factor out the indexing concern (including the assumption of _key_of_block property existence)
class BlocksSearchResult:
    def __init__(
        self, _block_store, bt=-inf, tt=inf, _inclusive=(True, False), _sort_key=None,
    ):
        """
        Object that contains and implements an interval query of a block store.
        :param _block_store: A store. Must be iterable, have a __getitem__, and have sr and time_units_per_sec attrs
        :param bt: the bottom time of the time interval query
        :param tt: the top time of the time interval query
        :param _inclusive: Whether to include or exclude the bt and/or tt
        :param _sort_key: What to sort on.
        >>> from collections import UserDict
        >>> _block_store = UserDict({1000: [1,2,3,4], 1008: [5,6,7,8,9]})
        >>> _block_store.sr = 44100
        >>> _block_store.time_units_per_sec = 88200
        >>>
        >>> r = BlocksSearchResult(_block_store, bt=1004, tt=1014)
        >>> list(r.values())
        [[3, 4], [5, 6, 7]]
        >>> assert list(BlocksSearchResult(_block_store).values()) == [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
        >>> assert list(BlocksSearchResult(_block_store, bt=1002, tt=20000).values()) == [[2, 3, 4], [5, 6, 7, 8, 9]]
        >>> assert list(BlocksSearchResult(_block_store, bt=900, tt=1000).values()) == []
        >>> assert list(BlocksSearchResult(_block_store, bt=900, tt=1004).values()) == [[1, 2]]
        >>> assert list(BlocksSearchResult(_block_store, bt=1004, tt=1008).values()) == [[3, 4]]
        >>> assert list(BlocksSearchResult(_block_store, bt=1004, tt=1009).values()) == [[3, 4], [5]]
        >>> assert list(BlocksSearchResult(_block_store, bt=1004, tt=1010).values()) == [[3, 4], [5]]
        >>> assert list(BlocksSearchResult(_block_store, bt=1004, tt=1011).values()) == [[3, 4], [5, 6]]

        """
        self._block_store = _block_store
        self.bt = bt
        self.tt = tt
        self.sr = self._block_store.sr  # sample rate of the block data
        self.time_units_per_sec = (
            self._block_store.time_units_per_sec
        )  # sample rate of the query time unit
        self._inclusive = _inclusive
        # _sort_key is not used here, but can be used to tell SortedList how to sort keys outputed by __iter__
        self._sort_key = _sort_key

    @cached_property
    def _key_of_block(self):
        if hasattr(self._block_store, '_key_of_block'):
            return self._block_store._key_of_block
        else:
            return SortedList(self._block_store.__iter__(), self._sort_key)

    @cached_property
    def intersecting_blocks(self):
        """ Iterator of blocks that CONTAIN the [bt, tt) range """
        # TODO: Figure out the most efficient way to do this
        bottom_idx = self._key_of_block.bisect_right(self.bt)
        if bottom_idx > 0:
            bottom_idx -= 1
        min_block = self._key_of_block[bottom_idx]
        return list(
            self._key_of_block.irange(
                minimum=min_block, maximum=self.tt, inclusive=self._inclusive
            )
        )

    def __iter__(
        self,
    ):  # TODO: Change to be the bts of the values returned by items, not block bts.
        return self.items()

    def blocks_intersecting_bt_tt(self):
        """ Iterator of blocks that CONTAIN the [bt, tt) range """
        # TODO: Figure out the most efficient way to do this
        bottom_idx = self._key_of_block.bisect_right(self.bt)
        if bottom_idx > 0:
            bottom_idx -= 1
        min_block = self._key_of_block[bottom_idx]
        return self._key_of_block.irange(
            minimum=min_block, maximum=self.tt, inclusive=self._inclusive
        )

    def items(self):
        samples_per_ts_unit = self.sr / self.time_units_per_sec
        assert self.bt < self.tt, 'you entered a bt >= tt'

        ts_gen = iter(self.intersecting_blocks)
        # getting the first block
        try:
            current_block_ts = next(ts_gen)
            while current_block_ts < self.tt:
                wf = self._block_store[current_block_ts]
                wf_len_ts_unit = len(wf) / samples_per_ts_unit
                # how much to remove from the left and right from the wf in the unit of the block timestamps
                remove_left_ts_unit = max(0, self.bt - current_block_ts)
                remove_right_ts_unit = max(
                    0, current_block_ts + wf_len_ts_unit - self.tt
                )
                # converting into number of samples:
                # is int best here for rounding? Does not matter in our case at hand since no decimal
                remove_left_sample_unit = int(remove_left_ts_unit * samples_per_ts_unit)
                remove_right_sample_unit = int(
                    remove_right_ts_unit * samples_per_ts_unit
                )

                # sadly array[:-0] is empty (of course) instead of everything so I have to take that case apart
                # there must be a better way but running out of time
                if remove_right_ts_unit:
                    wf = wf[remove_left_sample_unit:-remove_right_sample_unit]
                else:
                    wf = wf[remove_left_sample_unit:]
                if len(wf) > 0:
                    wf_bt = current_block_ts + remove_left_ts_unit
                    wf_tt = current_block_ts + wf_len_ts_unit - remove_right_ts_unit
                    yield (wf_bt, wf_tt), wf
                current_block_ts = next(ts_gen)
        except StopIteration:
            pass

    def keys(self):
        for k, _ in self.items():
            yield k

    def values(self):
        for _, v in self.items():
            yield v

    @lru_cache(maxsize=1)
    def __len__(self):
        return super().__len__()

    @lru_cache(maxsize=1)
    def __contains__(self, k):
        return super().__contains__(k)


class BlockWfStore(DictKeyMap, WfStore):
    def __init__(
        self,
        channel_data_dir,
        sr=None,
        time_units_per_sec=DFLT_TIME_UNITS_PER_SEC,
        _sort_key=None,
    ):
        # TODO: Here I construct a WfStore twice. There must be a better way!
        s = WfStore(channel_data_dir, sr)
        key_of_block = {s.get_session_and_block(key)[1]: key for key in s.__iter__()}
        # assert list(key_of_block.keys()) == sorted(key_of_block.keys()), "the blocks are not sorted"
        WfStore.__init__(self, channel_data_dir, sr)
        DictKeyMap.__init__(self, id_of_key=key_of_block)
        self.time_units_per_sec = time_units_per_sec

        self._sort_key = _sort_key

    @cached_property
    def _key_of_block(self):
        return SortedList(self.__iter__(), self._sort_key)

    def block_search(self, bt=-inf, tt=inf):
        return BlocksSearchResult(
            _block_store=self,
            bt=bt,
            tt=tt,
            _inclusive=(True, False),
            _sort_key=self._sort_key,
        )


class SessionStore(LocalPathStore):
    def __init__(
        self,
        session_dir,
        time_units_per_sec=DFLT_TIME_UNITS_PER_SEC,
        csv_timestamp_time_units_per_sec=int(1e3),
        rel_path_format='{session:13d}/t/tags{csv_timestamp:13d}.csv',
        sr=None,
        **kwargs
    ):
        """
        A store for sessions folders.
        :param session_dir: The sessions directory, which contains sessions subfolders.
        :param sr:
        :param time_units_per_sec: The timestamp unit to use for bt and tt, in num of units per second.
            For example, if milliseconds, time_units_per_sec=1000, if microseconds, time_units_per_sec=1000000.
        :param csv_timestamp_time_units_per_sec: "time sample rate" The timestamp unit the csv filename uses.
        :param rel_path_format: The pathformat under the sessions_dir. Is used to filter the files as
            well as match information encoded in filename (such as csv_timestamp)
        :param kwargs: Passed on to the super class __init__
        """
        path_format = os.path.join(session_dir, rel_path_format)

        super().__init__(path_format, **kwargs)

        self._path_match_re = match_re_for_fstring(path_format)
        self.sr = sr
        self.time_units_per_sec = time_units_per_sec
        self.csv_timestamp_time_units_per_sec = csv_timestamp_time_units_per_sec

    def key_info(self, k):
        """Extract dict of info from parsing file"""
        m = self._path_match_re.search(self._id_of_key(k))
        if m:
            return {field: int(val) for field, val in m.groupdict().items()}
        else:
            return {}


class ScoopAnnotationsStore(SessionStore):
    @staticmethod
    def df_of_csv(csv_str):
        from io import StringIO
        import pandas as pd

        return pd.read_csv(StringIO(csv_str))

    def __getitem__(self, k):
        v = super().__getitem__(k)
        df = self.df_of_csv(v)
        csv_timestamp = self.key_info(k).get('csv_timestamp', None)
        if csv_timestamp is None:
            raise ValueError("Couldn't parse out the csv's timestamp.")
        csv_timestamp = int(
            csv_timestamp
            * self.time_units_per_sec
            / self.csv_timestamp_time_units_per_sec
        )
        df['bt'] = csv_timestamp + df['bt'] * self.time_units_per_sec
        df['tt'] = csv_timestamp + df['tt'] * self.time_units_per_sec
        df = df.sort_values(['bt', 'tt'])
        return df


class ScoopAnnotations:
    def __init__(self, annotations_dir):
        import pandas as pd

        store = ScoopAnnotationsStore(annotations_dir)
        self._store = store
        self.annots_df = pd.concat(list(store.values())).sort_values(['bt', 'tt'])
        self.annots_df = self.annots_df.reset_index(drop=True)

    def __getitem__(self, k):
        if not isinstance(k, slice):
            k = slice(k)
        df = self.annots_df
        return df[(df['bt'] >= k.start) & (df['tt'] < k.stop)].to_dict(orient='rows')


# if __name__ == '__main__':
#     from otolite.stores import BlocksSearchResult
#     from collections import UserDict
#
#     _block_store = UserDict({1000: [1, 2, 3, 4], 1008: [5, 6, 7, 8, 9]})
#     _block_store.sr = 44100
#     _block_store.time_units_per_sec = 88200
#
#     r = BlocksSearchResult(_block_store, bt=1001, tt=1014)
#     print(list(r.items()))
#
#     assert list(BlocksSearchResult(_block_store, bt=900, tt=1009).values()) == [[1, 2, 3, 4]]
