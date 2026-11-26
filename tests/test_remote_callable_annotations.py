"""create_remote_callable with annotations that cannot be evaluated (PEP 649)."""

from typing import TYPE_CHECKING

from exxec.remote_callable import create_remote_callable


if TYPE_CHECKING:
    from decimal import Decimal  # only imported for type checkers


def _unresolvable(value: Decimal) -> Decimal:
    return value


def _typed(value: int) -> list[int]:
    return [value]


def test_unresolvable_annotations_do_not_raise() -> None:
    wrapped = create_remote_callable(_unresolvable, object)  # type: ignore[arg-type]
    assert callable(wrapped)


def test_resolvable_annotations_still_work() -> None:
    wrapped = create_remote_callable(_typed, object)  # type: ignore[arg-type]
    assert callable(wrapped)
