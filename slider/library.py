from __future__ import annotations

import os
import sys
import logging
import pathlib
import sqlite3
from functools import lru_cache
from hashlib import md5
from typing import (
    ContextManager,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast
)

import requests  # type: ignore[import]

from .beatmap import Beatmap
from .cli import maybe_show_progress

if sys.platform.startswith("win"):

    def sanitize_filename(name: str) -> str:
        for invalid_character in r':*?"\/|<>':
            name = name.replace(invalid_character, "")
        return name

else:

    def sanitize_filename(name: str) -> str:
        return name.replace("/", "")


sanitize_filename.__doc__ = """\
Sanitize a filename so that it is safe to write to the filesystem.

Parameters
----------
name : str
    The name of the file without the directory.

Returns
-------
sanitized_name : str
    The name with invalid characters stripped out.
"""

class _CachedBeatmapReader(Protocol):
    def __call__(
        self,
        library: "Library",
        *,
        beatmap_id: Optional[Union[int, str]] = ...,
        beatmap_md5: Optional[str] = ...,
    ) -> Beatmap:
        ...

    def cache_clear(self) -> None:
        ...


class Library:
    """A library of beatmaps backed by a local directory.

    Parameters
    ----------
    path : path-like
        The path to a local library directory.
    cache : int, optional
        The amount of beatmaps to cache in memory. This uses
        :func:`functools.lru_cache`, and if set to None will cache everything.
    download_url : str, optional
        The default location to download beatmaps from.
    """

    DEFAULT_DOWNLOAD_URL = "https://osu.ppy.sh/osu"
    DEFAULT_CACHE_SIZE = 2048

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        cache: Optional[int] = DEFAULT_CACHE_SIZE,
        download_url: str = DEFAULT_DOWNLOAD_URL,
    ) -> None:
        self.path = path = pathlib.Path(path)

        self._cache_size = cache
        cached_reader = lru_cache(cache)(self._raw_read_beatmap)
        self._read_beatmap = cast(_CachedBeatmapReader, cached_reader)
        self._db = db = sqlite3.connect(str(path / ".slider.db"))
        with db:
            db.execute(
                """\
                CREATE TABLE IF NOT EXISTS beatmaps (
                    md5 BLOB PRIMARY KEY,
                    id INT,
                    path TEXT UNIQUE NOT NULL
                )
                """,
            )
        self._download_url = download_url

    def copy(self) -> "Library":
        """Create a copy suitable for use in a new thread.

        Returns
        -------
        Library
            The new copy.
        """
        return type(self)(
            self.path,
            cache=self._cache_size,
            download_url=self._download_url,
        )

    def close(self) -> None:
        """Close any resources used by this library."""
        self._read_beatmap.cache_clear()
        self._db.close()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _osu_files(
        path: Union[str, os.PathLike[str]],
        recurse: bool,
    ) -> Iterator[pathlib.Path]:
        """An iterator of ``.osu`` filepaths in a directory.

        Parameters
        ----------
        path : path-like
            The directory to search in.
        recurse : bool
            Recursively search ``path``?

        Yields
        ------
        path : str
            The path to a ``.osu`` file.
        """
        if recurse:
            for directory, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(".osu"):
                        yield pathlib.Path(os.path.join(directory, filename))
        else:
            for entry in os.scandir(path):
                entry_path = entry.path
                if entry_path.endswith(".osu"):
                    yield pathlib.Path(entry_path)

    @classmethod
    def create_db(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        recurse: bool = True,
        cache: Optional[int] = DEFAULT_CACHE_SIZE,
        download_url: str = DEFAULT_DOWNLOAD_URL,
        show_progress: bool = False,
        skip_exceptions: bool = False,
    ) -> "Library":
        """Create a Library from a directory of ``.osu`` files.

        Parameters
        ----------
        path : path-like
            The path to the directory to read.
        recurse : bool, optional
            Recursively search for beatmaps?
        cache : int, optional
            The amount of beatmaps to cache in memory. This uses
            :func:`functools.lru_cache`, and if set to None will cache
            everything.
        download_url : str, optional
            The default location to download beatmaps from.
        show_progress : bool, optional
            Display a progress bar?

        Notes
        -----
        Moving the underlying ``.osu`` files invalidates the library. If this
        happens, just re-run ``create_db`` again.
        """
        path = pathlib.Path(path)
        db_path = path / ".slider.db"
        try:
            # ensure the db is cleared
            os.remove(db_path)
        except FileNotFoundError:
            pass

        self = cls(path, cache=cache, download_url=download_url)
        write_to_db = self._write_to_db

        progress = cast(
            ContextManager[Iterable[pathlib.Path]],
            maybe_show_progress(
                self._osu_files(path, recurse=recurse),
                show_progress,
                label="Processing beatmaps: ",
                item_show_func=lambda p: "Done!" if p is None else str(p.stem),
            ),
        )
        with self._db, progress as it:
            for file_path in it:
                with open(file_path, "rb") as f:
                    data = f.read()

                try:
                    beatmap = Beatmap.parse(data.decode("utf-8-sig"))
                except Exception as e:
                    if skip_exceptions:
                        logging.exception(f'Failed to parse "{file_path}"')
                        continue
                    raise ValueError(
                        f'Failed to parse "{file_path}". '
                        "Use --skip-exceptions to skip this file and continue."
                    ) from e

                write_to_db(beatmap, data, file_path)

        return self

    def beatmap_cached(
        self,
        *,
        beatmap_id: Optional[int] = None,
        beatmap_md5: Optional[str] = None,
    ) -> bool:
        """Whether we have the given beatmap cached.

        Parameters
        ----------
        beatmap_id : int
            The id of the beatmap to look for.
        beatmap_md5 : str
            The md5 hash of the beatmap to look for.

        Returns
        -------
        bool
            Whether we have the given beatmap cached.
        """
        with self._db:
            if beatmap_id is not None:
                path_query = self._db.execute(
                    "SELECT 1 FROM beatmaps WHERE id = ? LIMIT 1",
                    (beatmap_id,),
                )
            else:
                if beatmap_md5 is None:
                    raise ValueError("beatmap_md5 must be provided if beatmap_id is None")
                path_query = self._db.execute(
                    "SELECT 1 FROM beatmaps WHERE md5 = ? LIMIT 1",
                    (beatmap_md5,),
                )

        path = path_query.fetchone()
        return bool(path)

    @staticmethod
    def _raw_read_beatmap(
        self: "Library",
        *,
        beatmap_id: Optional[Union[int, str]] = None,
        beatmap_md5: Optional[str] = None,
    ) -> Beatmap:
        """Function for opening beatmaps from disk.

        This handles both cases to only require a single lru cache.

        Notes
        -----
        This is a ``staticmethod`` to avoid a cycle from self to the lru_cache
        back to self.
        """
        with self._db:
            if beatmap_id is not None:
                key = beatmap_id
                path_query = self._db.execute(
                    "SELECT path FROM beatmaps WHERE id = ?",
                    (beatmap_id,),
                )
            else:
                if beatmap_md5 is None:
                    raise ValueError("beatmap_md5 must be provided if beatmap_id is None")
                key = beatmap_md5
                path_query = self._db.execute(
                    "SELECT path FROM beatmaps WHERE md5 = ?",
                    (beatmap_md5,),
                )

        path = path_query.fetchone()
        if path is None:
            raise KeyError(key)

        (path,) = path
        # Make path relative to the root path. We save paths relative to
        # ``self.path`` so a library can be relocated without requiring a
        # rebuild
        return Beatmap.from_path(self.path / path)

    def lookup_by_id(
        self,
        beatmap_id: Union[int, str],
        *,
        download: bool = False,
        save: bool = False,
    ) -> Beatmap:
        """Retrieve a beatmap by its beatmap id.

        Parameters
        ----------
        beatmap_id : int or str
            The id of the beatmap to lookup.

        Returns
        -------
        beatmap : Beatmap
            The beatmap with the given id.
        download : bool. optional
            Download the map if it doesn't exist.
        save : bool, optional
            If the lookup falls back to a download, should the result be saved?

        Raises
        ------
        KeyError
            Raised when the given id is not in the library.
        """
        try:
            return self._read_beatmap(self, beatmap_id=beatmap_id)
        except KeyError:
            if not download:
                raise
        return self.download(beatmap_id, save=save)

    def lookup_by_md5(self, beatmap_md5: str) -> Beatmap:
        """Retrieve a beatmap by its md5 hash.

        Parameters
        ----------
        beatmap_md5 : bytes
            The md5 hash of the beatmap to lookup.

        Returns
        -------
        beatmap : Beatmap
            The beatmap with the given md5 hash.

        Raises
        ------
        KeyError
            Raised when the given md5 hash is not in the library.
        """
        return self._read_beatmap(self, beatmap_md5=beatmap_md5)

    def beatmap_from_path(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        copy: bool = False,
    ) -> Beatmap:
        """Returns a beatmap from a file on disk.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the file to create the beatmap from.
        copy : bool
            Should the file be copied to the library's beatmap directory?

        Returns
        -------
        beatmap : Beatmap
            The beatmap represented by the given file.
        """
        with open(path, "rb") as f:
            data_bytes = f.read()
        data = data_bytes.decode("utf-8-sig")
        beatmap = Beatmap.parse(data)

        if copy:
            self.save(data_bytes, beatmap=beatmap)

        return beatmap

    def save(
        self,
        data: bytes,
        *,
        beatmap: Optional[Beatmap] = None,
    ) -> Beatmap:
        """Save raw data for a beatmap at a given location.

        Parameters
        ----------
        data : bytes
            The unparsed beatmap data.
        beatmap : Beatmap, optional
            The parsed beatmap. If not provided, the raw data will be parsed.

        Returns
        -------
        beatmap : Beatmap
            The parsed beatmap.
        """
        if beatmap is None:
            beatmap = Beatmap.parse(data.decode("utf-8-sig"))

        path = self.path / sanitize_filename(
            f"{beatmap.artist} - "
            f"{beatmap.title} "
            f"({beatmap.creator})"
            f"[{beatmap.version}]"
            f".osu"
        )
        with open(path, "wb") as f:
            f.write(data)

        with self._db:
            self._write_to_db(beatmap, data, path)
        return beatmap

    def delete(
        self,
        beatmap: Beatmap,
        *,
        remove_file: bool = True,
    ) -> None:
        """Remove a beatmap from the library.

        Parameters
        ----------
        beatmap : Beatmap
            The beatmap to delete.
        remove_file : bool, optional
            Remove the .osu file from disk.
        """
        with self._db:
            if remove_file:
                paths = self._db.execute(
                    "SELECT path FROM beatmaps WHERE id = ?",
                    (beatmap.beatmap_id,),
                )
                for (path,) in paths:
                    os.unlink(path)

            self._db.execute(
                "DELETE FROM beatmaps WHERE id = ?",
                (beatmap.beatmap_id,),
            )

    def _write_to_db(
        self,
        beatmap: Beatmap,
        data: bytes,
        path: pathlib.Path,
    ) -> None:
        """Write data to the database.

        Parameters
        ----------
        beatmap : Beatmap
            The beatmap being stored.
        data : bytes
            The raw data for the beatmap
        path : str
            The path to save
        """
        # save paths relative to ``self.path`` so a library can be relocated
        # without requiring a rebuild
        path = path.relative_to(self.path)
        beatmap_md5 = md5(data).hexdigest()
        beatmap_id = beatmap.beatmap_id

        try:
            self._db.execute(
                "INSERT INTO beatmaps VALUES (?,?,?)",
                (beatmap_md5, beatmap_id, str(path)),
            )
        except sqlite3.IntegrityError:
            # ignore duplicate beatmaps
            pass

    def download(
        self,
        beatmap_id: Union[int, str],
        *,
        save: bool = False,
    ) -> Beatmap:
        """Download a beatmap.

        Parameters
        ----------
        beatmap_id : int or str
            The id of the beatmap to download.
        save : bool, optional
            Save the beatmap to disk?

        Returns
        -------
        beatmap : Beatmap
            The downloaded beatmap.
        """
        beatmap_response = requests.get(f"{self._download_url}/{beatmap_id}")
        beatmap_response.raise_for_status()

        data = beatmap_response.content
        beatmap = Beatmap.parse(data.decode("utf-8-sig"))

        if save:
            self.save(data, beatmap=beatmap)

        return beatmap

    @property
    def md5s(self) -> Tuple[str, ...]:
        """All of the beatmap hashes that this has downloaded."""
        return tuple(md5 for md5, in self._db.execute("SELECT md5 FROM beatmaps"))

    @property
    def ids(self) -> Tuple[int, ...]:
        """All of the beatmap ids that this has downloaded."""
        return tuple(
            int(id_)
            for id_, in self._db.execute("SELECT id FROM beatmaps")
            if id_ is not None
        )
