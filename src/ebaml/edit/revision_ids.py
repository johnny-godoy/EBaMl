"""The identifiers of the revisions of an edit session.

An id is a UUID of version 7 from ``uuid_utils.compat.uuid7``, which returns a real ``uuid.UUID``. Python 3.13 has no
``uuid.uuid7``. When 3.14 is the lowest version that the package supports, use ``uuid.uuid7`` and drop the dependency.

An id names a revision. It does not give the order of the revisions: the position in the history does. The generators
share the last timestamp between threads, and the order of the ids of two threads is not a promise. The 74 random bits
keep the ids unique in every case. A lock serializes the calls anyway, because the locking of the native code was not
checked.
"""

import threading
import uuid

from uuid_utils.compat import uuid7

_ID_LOCK = threading.Lock()


def new_revision_id() -> uuid.UUID:
    """Make the id of a new revision.

    Returns
    -------
    uuid.UUID
        A UUID of version 7, different from every other id that this function made.

    """
    with _ID_LOCK:
        return uuid7()
