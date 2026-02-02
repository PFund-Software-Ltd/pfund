from pfund.enums import RunMode


# FIXME: remove it
def derive_run_mode(ray_kwargs: dict | None=None) -> RunMode:
    if ray_kwargs:
        # NOTE: if `num_cpus` is not set, Ray will only use 1 CPU for scheduling, and 0 CPU for running
        assert 'num_cpus' in ray_kwargs, '`num_cpus` must be set for a Ray actor'
        assert ray_kwargs['num_cpus'] > 0, '`num_cpus` must be greater than 0'
        run_mode = RunMode.REMOTE
    else:
        run_mode = RunMode.LOCAL
    return run_mode
