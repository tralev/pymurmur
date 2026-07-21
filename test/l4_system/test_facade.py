"""P10.5 — Simulation facade unit tests.

Tests the pymurmur.Simulation class: construction, run, metrics, benchmark.
"""



class TestSimulationFacade:
    """P10.5: pymurmur.Simulation — headless entry point and benchmark API."""

    def test_construct_defaults(self):
        """Simulation() with no args creates a usable simulation."""
        from pymurmur import Simulation
        sim = Simulation()
        assert sim.config.num_boids == 150
        assert sim.config.mode == "projection"

    def test_construct_with_kwargs(self):
        """Simulation(num_boids=500, mode='spatial', seed=42) routes to config."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=500, mode="spatial", seed=42)
        assert sim.config.num_boids == 500
        assert sim.config.mode == "spatial"
        assert sim.config.seed == 42

    def test_construct_v0_kwarg(self):
        """Simulation(v0=6.0) sets cruise speed."""
        from pymurmur import Simulation
        sim = Simulation(v0=6.0)
        assert sim.config.v0 == 6.0

    def test_run_completes(self):
        """Simulation.run(steps=10) runs without error."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)
        sim.run(steps=10)

    def test_run_zero_steps(self):
        """Simulation.run(steps=0) is a no-op."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50)
        sim.run(steps=0)

    def test_metrics_history_after_run(self):
        """After run(steps=10), metrics_history has entries."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)
        sim.run(steps=10)
        assert len(sim.metrics_history) > 0, (
            f"Expected metrics after run, got {len(sim.metrics_history)}"
        )

    def test_metrics_history_empty_before_run(self):
        """Before run(), metrics_history is empty."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50)
        assert sim.metrics_history == []

    def test_benchmark_returns_list_of_floats(self):
        """benchmark() returns a list of float timings."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=100, seed=1)
        timings = sim.benchmark(flock_size=100, num_steps=20)
        assert isinstance(timings, list), "benchmark must return list"
        assert len(timings) == 20, (
            f"Expected 20 timings, got {len(timings)}"
        )
        for t in timings:
            assert isinstance(t, float), f"Timing {t!r} should be float"

    def test_benchmark_count_matches_num_steps(self):
        """benchmark(num_steps=N) returns exactly N values."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)
        for n in (5, 10, 20):
            timings = sim.benchmark(flock_size=100, num_steps=n)
            assert len(timings) == n, (
                f"benchmark(num_steps={n}) returned {len(timings)} values"
            )

    def test_benchmark_timings_are_positive(self):
        """All benchmark timings are > 0 (real wall-clock time)."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)
        timings = sim.benchmark(flock_size=100, num_steps=20)
        for i, t in enumerate(timings):
            assert t > 0.0, (
                f"Timing {i} = {t:.6f} should be positive"
            )

    def test_benchmark_different_flock_sizes(self):
        """benchmark() works with different flock sizes."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)
        for size in (50, 200, 500):
            timings = sim.benchmark(flock_size=size, num_steps=10)
            assert len(timings) == 10
            assert all(t > 0 for t in timings)

    def test_facade_importable_from_public_api(self):
        """Simulation class is importable from pymurmur (public facade)."""
        from pymurmur import Simulation
        assert isinstance(Simulation, type), (
            "pymurmur.Simulation must be a class, got {type(Simulation)}"
        )

    def test_run_with_callback(self):
        """Simulation.run() passes callback to engine.run_headless."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)

        frames = []
        def cb(engine):
            frames.append(engine.frame)

        sim.run(steps=10, callback=cb)
        assert len(frames) == 10, (
            f"Callback should be called 10 times, got {len(frames)}"
        )

    def test_run_callback_receives_increasing_frames(self):
        """Each callback invocation receives an engine with advancing frame count."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=50, seed=1)

        frame_numbers = []
        def cb(engine):
            frame_numbers.append(engine.frame)

        sim.run(steps=5, callback=cb)
        # Frame numbers should be strictly increasing
        for i in range(1, len(frame_numbers)):
            assert frame_numbers[i] > frame_numbers[i - 1], (
                f"Frames should increase: got {frame_numbers}"
            )

    def test_benchmark_default_params(self):
        """benchmark() with no args uses defaults (2000, 50)."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=100, seed=1)
        timings = sim.benchmark()
        assert len(timings) == 50, "Default benchmark should return 50 timings"

    def test_benchmark_does_not_affect_main_config(self):
        """benchmark() copies config so the facade's config is unchanged."""
        from pymurmur import Simulation
        sim = Simulation(num_boids=100, seed=1)
        old_n = sim.config.num_boids

        sim.benchmark(flock_size=500, num_steps=10)
        assert sim.config.num_boids == old_n, (
            f"benchmark() should not mutate facade config: "
            f"{old_n} → {sim.config.num_boids}"
        )


class TestSimulationPublicFacade:
    """P10.5: Simulation identity check — public facade exports real class."""

    def test_public_facade_simulation_is_same_class(self):
        """pymurmur.Simulation is the class defined in __init__.py."""
        # Simulation is defined directly in __init__.py, so re-importing
        # the same module should yield the same class object.
        from pymurmur import Simulation
        from pymurmur import Simulation as PublicSim
        assert PublicSim is Simulation, (
            "pymurmur.Simulation must be the same object on re-import"
        )
