"""SQLite-backed observation store with a single visibility gate.

The load-bearing invariant (DESIGN.md P1): every read path goes through
``Store._visible()``. There is no other code path that returns observation
data. Retrieval, views, exports, error messages — all of them call the same
gate. The leak-path test suite (tests/test_leak_paths.py) enumerates every
public surface and proves a hidden observation cannot escape through it.

Identity is caller-supplied at the Store API layer but is expected to come
from the RUNTIME (session objects, framework auth), never from a model (P2).
Framework adapters are responsible for that resolution; the model-facing
tool surface only ever sees resolved tokens.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from .model import (
    KithError,
    Observation,
    Scope,
    _validate_context,
    _validate_principal_id,
    validate_payload,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id       TEXT PRIMARY KEY,
    observer TEXT NOT NULL,
    subject  TEXT NOT NULL,
    kind     TEXT NOT NULL,
    context  TEXT,
    payload  TEXT NOT NULL,
    scope    TEXT NOT NULL,
    ts       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_observer_subject
    ON observations (observer, subject, ts);
CREATE INDEX IF NOT EXISTS idx_obs_subject ON observations (subject, ts);

CREATE TABLE IF NOT EXISTS grants (
    id        TEXT PRIMARY KEY,
    obs_id    TEXT NOT NULL REFERENCES observations(id),
    grantor   TEXT NOT NULL,
    grantee   TEXT NOT NULL,
    contexts  TEXT NOT NULL,     -- JSON list; empty = all contexts of the obs
    reason    TEXT,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grants_obs ON grants (obs_id);
CREATE INDEX IF NOT EXISTS idx_grants_grantee ON grants (grantee);
"""


class Store:
    """Append-only observation store. One instance per trust domain."""

    def __init__(self, path: str = ":memory:"):
        # Accept "sqlite:///x.db" (docs style) or a bare path.
        if path.startswith("sqlite:///"):
            path = path[len("sqlite:///"):] or ":memory:"
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    # -- write path ---------------------------------------------------------

    def observe(
        self,
        observer: str,
        subject: str,
        kind: str,
        payload: Dict[str, Any],
        context: Optional[str] = None,
        scope: Optional[Scope] = None,
    ) -> Observation:
        """Record one observation. Scope defaults to observer-private."""
        validate_payload(kind, payload)
        obs = Observation(
            observer=observer, subject=subject, kind=kind,
            payload=payload, context=context, scope=scope or Scope(),
        )
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO observations "
                "(id, observer, subject, kind, context, payload, scope, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (obs.id, obs.observer, obs.subject, obs.kind, obs.context,
                 json.dumps(obs.payload, ensure_ascii=False),
                 obs.scope.to_json(), obs.ts),
            )
        return obs

    def grant(
        self,
        grantor: str,
        obs_id: str,
        to: str,
        contexts: Iterable[str] = (),
        reason: Optional[str] = None,
    ) -> str:
        """Explicitly share one observation with another principal.

        Only the OBSERVER of an observation may grant it (the observer owns
        the memory). Grants are append-only audit records; revocation is a
        future concern (would be another record type, never a delete).
        """
        _validate_principal_id(grantor)
        _validate_principal_id(to)
        ctxs = [_validate_context(c) for c in contexts]
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT observer FROM observations WHERE id = ?", (obs_id,)
            ).fetchone()
            # Deliberate: identical error for "doesn't exist" and "not yours".
            # A distinguishable error would let a caller probe for the
            # existence of other principals' observations (P1).
            if row is None or row["observer"] != grantor:
                raise KithError("No grantable observation with that id.")
            import uuid as _uuid
            gid = _uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO grants (id, obs_id, grantor, grantee, contexts, reason, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (gid, obs_id, grantor, to,
                 json.dumps(ctxs, ensure_ascii=False), reason, time.time()),
            )
        return gid

    # -- THE gate -----------------------------------------------------------

    def _visible(
        self,
        row: sqlite3.Row,
        reader: str,
        reader_context: Optional[str],
        grants_by_obs: Dict[str, List[sqlite3.Row]],
    ) -> bool:
        """The single visibility rule. Every read path funnels through here.

        Visible iff ANY of:
          a) reader is the observer, OR
          b) reader is in the observation's scope.holders AND (scope has no
             context restriction OR reader_context is one of them), OR
          c) a grant exists for (this obs, this reader) AND (grant has no
             context restriction OR reader_context is in it).
        """
        if row["observer"] == reader:
            return True
        scope = Scope.from_json(row["scope"])
        if reader in scope.holders:
            if not scope.contexts or (reader_context in scope.contexts):
                return True
        for g in grants_by_obs.get(row["id"], ()):  # explicit shares
            if g["grantee"] != reader:
                continue
            g_ctxs = json.loads(g["contexts"] or "[]")
            if not g_ctxs or (reader_context in g_ctxs):
                return True
        return False

    def _read(
        self,
        reader: str,
        reader_context: Optional[str] = None,
        subject: Optional[str] = None,
        observer: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Internal scoped read. ALL public read surfaces call this."""
        _validate_principal_id(reader)
        if reader_context is not None:
            _validate_context(reader_context)
        q = "SELECT * FROM observations WHERE 1=1"
        args: List[Any] = []
        if subject is not None:
            q += " AND subject = ?"
            args.append(subject)
        if observer is not None:
            q += " AND observer = ?"
            args.append(observer)
        if kind is not None:
            q += " AND kind = ?"
            args.append(kind)
        q += " ORDER BY ts ASC"
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
            grant_rows = self._db.execute(
                "SELECT * FROM grants WHERE grantee = ?", (reader,)
            ).fetchall()
        grants_by_obs: Dict[str, List[sqlite3.Row]] = {}
        for g in grant_rows:
            grants_by_obs.setdefault(g["obs_id"], []).append(g)

        out: List[Observation] = []
        for row in rows:
            if not self._visible(row, reader, reader_context, grants_by_obs):
                continue
            out.append(Observation(
                id=row["id"], observer=row["observer"], subject=row["subject"],
                kind=row["kind"], context=row["context"],
                payload=json.loads(row["payload"]),
                scope=Scope.from_json(row["scope"]), ts=row["ts"],
            ))
        if limit is not None:
            out = out[-limit:]
        return out

    # -- public read surfaces (ALL of them delegate to _read) ---------------

    def observations(
        self,
        reader: str,
        subject: Optional[str] = None,
        reader_context: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Observations visible to ``reader``, optionally filtered."""
        return self._read(reader, reader_context, subject=subject,
                          kind=kind, limit=limit)

    def export(self, reader: str, reader_context: Optional[str] = None) -> str:
        """JSON export of everything ``reader`` may see — and nothing else.

        Export is a classic leak surface ("admin dump"); it gets no special
        privileges (P1: admin = a principal with granted scopes).
        """
        obs = self._read(reader, reader_context)
        return json.dumps(
            [{
                "id": o.id, "observer": o.observer, "subject": o.subject,
                "kind": o.kind, "context": o.context, "payload": o.payload,
                "ts": o.ts,
            } for o in obs],
            ensure_ascii=False, indent=2,
        )

    def principal(self, pid: str) -> "BoundPrincipal":
        """Ergonomic handle: a principal-bound facade over this store."""
        _validate_principal_id(pid)
        return BoundPrincipal(self, pid)


class BoundPrincipal:
    """Store facade bound to one principal's identity (the README surface).

    The binding is done ONCE, by whoever holds the Store — i.e. the runtime,
    not the model (P2). Everything called through this handle reads/writes
    as that principal.
    """

    def __init__(self, store: Store, pid: str):
        self._store = store
        self.id = pid

    def observe(self, subject: str, kind: str, payload: Dict[str, Any],
                context: Optional[str] = None,
                scope: Optional[Scope] = None) -> Observation:
        return self._store.observe(self.id, subject, kind, payload,
                                   context=context, scope=scope)

    def view(self, subject: str, context: Optional[str] = None):
        from .view import RelationshipView
        return RelationshipView.build(self._store, self.id, subject, context)

    def observations(self, subject: Optional[str] = None,
                     context: Optional[str] = None,
                     kind: Optional[str] = None,
                     limit: Optional[int] = None):
        return self._store.observations(self.id, subject=subject,
                                        reader_context=context, kind=kind,
                                        limit=limit)

    def grant(self, obs_id: str, to: str, contexts: Iterable[str] = (),
              reason: Optional[str] = None) -> str:
        return self._store.grant(self.id, obs_id, to,
                                 contexts=contexts, reason=reason)

    def export(self, context: Optional[str] = None) -> str:
        return self._store.export(self.id, reader_context=context)
