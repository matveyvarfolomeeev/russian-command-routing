"""A transparent text filter, also usable as the first experimental baseline."""

from .objects import CatalogObject


def normalize(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def filter_catalog(
    items: list[CatalogObject], query: str, *, active_only: bool = True
) -> list[CatalogObject]:
    needle = normalize(query)
    results = []
    for item in items:
        if active_only and not (item.enabled and item.available):
            continue
        fields = [item.name, item.description, *item.aliases]
        if not needle or any(needle in normalize(field) for field in fields):
            results.append(item)
    return sorted(
        results,
        key=lambda item: (
            normalize(item.name) != needle if needle else False,
            not item.favorite,
            item.name.casefold(),
            item.object_id,
        ),
    )
