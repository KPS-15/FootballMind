from fastapi import APIRouter
from src.core.types import WhatIfRequest, WhatIfResponse
from src.simulation.simulator import WhatIfSimulator
from src.data.synthetic_generator import SyntheticMatchGenerator

router = APIRouter(tags=["What-If Simulation"])
simulator = WhatIfSimulator()
generator = SyntheticMatchGenerator()
match_frames = generator.generate_sequence(num_frames=120)


@router.post("/simulation", response_model=WhatIfResponse)
def run_simulation(req: WhatIfRequest):
    idx = max(0, min(len(match_frames) - 1, req.frame_index))
    frame = match_frames[idx]
    
    response = simulator.simulate_scenario(frame, req)
    return response
