"""Retry short SQLite write contention only outside a rolled-back transaction."""
import sqlite3
import time
from functools import wraps

from django.db import OperationalError, close_old_connections, connection


def retry_sqlite_write(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        for attempt in range(4):
            try:
                return operation(*args, **kwargs)
            except OperationalError as exc:
                cause = exc.__cause__
                code = getattr(cause, 'sqlite_errorcode', None)
                busy = (code is not None and (code & 0xff) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
                busy = busy or str(exc).lower().startswith(('database is locked', 'database table is locked', 'database schema is locked'))
                # An enclosing atomic block must unwind first. The caller that
                # owns the complete transaction retries it, never a broken save.
                if connection.vendor != 'sqlite' or not busy or connection.in_atomic_block or attempt == 3:
                    raise
                close_old_connections()
                time.sleep(.15 * (2 ** attempt))
    return wrapped
