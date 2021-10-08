from functools import lru_cache


# @lru_cache
def get_dependencies():
    from pip._internal.operations import freeze  # lazy import to reduce startup

    return list(freeze.freeze())
