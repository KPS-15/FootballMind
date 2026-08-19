import pytest
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.simulation.simulator import WhatIfSimulator
from src.core.types import WhatIfRequest


def test_what_if_simulator():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=10)
    simulator = WhatIfSimulator()

    req = WhatIfRequest(
        match_id="test",
        frame_index=0,
        modified_player_id=3,
        new_x=22.0,
        new_y=26.0
    )

    res = simulator.simulate_scenario(frames[0], req)
    assert 0.0 <= res.baseline_danger <= 1.0
    assert 0.0 <= res.scenario_danger <= 1.0
    assert isinstance(res.summary, str)
