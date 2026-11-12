"""Tests for the identifiers of the revisions."""

import uuid
from concurrent.futures import ThreadPoolExecutor

from ebaml.edit.revision_ids import new_revision_id


def test_a_revision_id_is_a_uuid_of_version_7() -> None:
    revision_id = new_revision_id()
    assert isinstance(revision_id, uuid.UUID)
    assert revision_id.version == 7


def _make_id(_: int) -> uuid.UUID:
    return new_revision_id()


def test_32_threads_get_32_different_ids() -> None:
    barrier_size = 32
    with ThreadPoolExecutor(max_workers=barrier_size) as pool:
        ids = list(pool.map(_make_id, range(barrier_size)))
    assert len(set(ids)) == barrier_size
