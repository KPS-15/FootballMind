from typing import List
from src.core.types import WhatIfRequest


class TacticalScenarioManager:
    """
    Manages preset tactical scenarios for testing (e.g. defensive shift, inward fullback, high press).
    """

    @staticmethod
    def get_preset_scenarios() -> List[WhatIfRequest]:
        return [
            WhatIfRequest(
                match_id="demo_match_01",
                frame_index=45,
                modified_player_id=3,  # Home LB
                new_x=22.0,            # Shift inward by 3.2m
                new_y=26.0
            ),
            WhatIfRequest(
                match_id="demo_match_01",
                frame_index=45,
                modified_player_id=14, # Away CB
                new_x=72.0,            # Step up high line press
                new_y=26.0
            )
        ]
