import copy
from typing import Tuple
from src.core.types import FrameTacticalState, WhatIfRequest, WhatIfResponse
from src.tactical.defensive_analysis import DefensiveAnalyzer
from src.tactical.goalkeeper import GoalkeeperAnalyzer
from src.core.state_encoder import FootballStateEncoder


class WhatIfSimulator:
    """
    Counterfactual Tactical Simulator evaluating tactical modifications by recalculating
    defensive danger, passing space, and xG deltas from identical tactical feature engines.
    """

    def __init__(self):
        self.defensive_analyzer = DefensiveAnalyzer()
        self.gk_analyzer = GoalkeeperAnalyzer()
        self.state_encoder = FootballStateEncoder()

    def simulate_scenario(self, frame: FrameTacticalState, req: WhatIfRequest) -> WhatIfResponse:
        # 1. Evaluate baseline state
        base_def_index = self.defensive_analyzer.analyze_defensive_structure(frame)
        base_xg_prob = self.gk_analyzer.calculate_xg(frame, frame.ball.possession_player_id or 10)
        base_encoded = self.state_encoder.encode_frame(frame)

        # Average space across teammates
        base_space = sum(f["available_space"] for f in base_encoded["player_features"].values()) / max(1, len(base_encoded["player_features"]))

        # 2. Construct modified scenario state deep copy
        scenario_frame = copy.deepcopy(frame)
        target_player = next((p for p in scenario_frame.players if p.id == req.modified_player_id), None)

        if target_player:
            target_player.x = req.new_x
            target_player.y = req.new_y

        # 3. Recalculate scenario metrics
        scen_def_index = self.defensive_analyzer.analyze_defensive_structure(scenario_frame)
        scen_xg_prob = self.gk_analyzer.calculate_xg(scenario_frame, scenario_frame.ball.possession_player_id or 10)
        scen_encoded = self.state_encoder.encode_frame(scenario_frame)
        scen_space = sum(f["available_space"] for f in scen_encoded["player_features"].values()) / max(1, len(scen_encoded["player_features"]))

        # 4. Compute Deltas
        danger_delta = round(scen_def_index.overall_danger - base_def_index.overall_danger, 2)
        xg_delta = round(scen_xg_prob.goal_probability - base_xg_prob.goal_probability, 3)
        space_delta = round(scen_space - base_space, 2)

        summary_text = (
            f"Modifying Player #{req.modified_player_id} position to [{req.new_x:.1f}m, {req.new_y:.1f}m]: "
            f"Defensive Danger changed by {danger_delta * 100:+.1f} percentage points "
            f"(from {base_def_index.overall_danger * 100:.0f}% to {scen_def_index.overall_danger * 100:.0f}%)."
        )

        return WhatIfResponse(
            baseline_danger=round(base_def_index.overall_danger, 2),
            scenario_danger=round(scen_def_index.overall_danger, 2),
            danger_delta=danger_delta,
            baseline_xg=round(base_xg_prob.goal_probability, 3),
            scenario_xg=round(scen_xg_prob.goal_probability, 3),
            xg_delta=xg_delta,
            baseline_space=round(base_space, 2),
            scenario_space=round(scen_space, 2),
            space_delta=space_delta,
            summary=summary_text
        )
