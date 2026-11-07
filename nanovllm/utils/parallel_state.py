_rank = 0
_world_size = 1


def set_parallel_state(rank: int, world_size: int):
    global _rank, _world_size
    _rank = rank
    _world_size = world_size


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size
