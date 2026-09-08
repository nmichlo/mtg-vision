from __future__ import annotations

# A JSON document. Used for qdrant payloads and lightning checkpoints, which are
# both arbitrary-but-serialisable nested data rather than a fixed shape.
type Json = str | int | float | bool | None | list[Json] | dict[str, Json]
