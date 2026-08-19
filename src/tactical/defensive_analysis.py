import math
from typing import List
from src.core.types import DefensiveCollapseIndex, FrameTacticalState, PlayerState


class DefensiveAnalyzer:
    """
    Calculates continuously updated Defensive Collapse Index and component risk factors
    using pitch-direction aware spatial structural geometry.
    """

    def analyze_defensive_structure(
        self,
        frame: FrameTacticalState,
        defending_team: str = "away"
    ) -> DefensiveCollapseIndex:
        defenders = [p for p in frame.players if p.team == defending_team]
        attackers = [p for p in frame.players if p.team != defending_team]

        if not defenders:
            return DefensiveCollapseIndex(
                overall_danger=0.50,
                cb_lb_gap_risk=0.50,
                cb_rb_gap_risk=0.50,
                midfield_exposure=0.50,
                passing_lane_exposure=0.50,
                unmarked_attacker_risk=0.50,
                space_occupation_risk=0.50,
                methodology="Direction-Aware Spatial Structural Geometry Index"
            )

        # Defending direction: Away defends x=105 goal, Home defends x=0 goal
        is_defending_right = (defending_team == "away")
        defending_goal_x = 105.0 if is_defending_right else 0.0

        # Sort defenders laterally by Y coordinate (LB -> CB -> CB -> RB)
        sorted_defenders = sorted(defenders, key=lambda d: d.y)

        # 1. Inter-defender lateral gaps (ideal gap 8m - 12m)
        gaps = [
            math.hypot(sorted_defenders[i].x - sorted_defenders[i+1].x, sorted_defenders[i].y - sorted_defenders[i+1].y)
            for i in range(len(sorted_defenders) - 1)
        ]

        left_gaps = gaps[:len(gaps)//2] if gaps else [10.0]
        right_gaps = gaps[len(gaps)//2:] if gaps else [10.0]

        max_left_gap = max(left_gaps, default=10.0)
        max_right_gap = max(right_gaps, default=10.0)

        cb_lb_gap_risk = min(1.0, max(0.0, (max_left_gap - 12.0) / 14.0))
        cb_rb_gap_risk = min(1.0, max(0.0, (max_right_gap - 12.0) / 14.0))

        # 2. Midfield Exposure (distance between defensive line & midfield/ball)
        def_line_x = sum(d.x for d in defenders) / len(defenders)
        ball_def_dist = abs(frame.ball.x - def_line_x)
        midfield_exposure = min(1.0, max(0.0, ball_def_dist / 28.0))

        # 3. Passing Lane Exposure into Zone 14 (central attacking region 18m out)
        zone14_x = 87.0 if is_defending_right else 18.0
        zone14_unblocked = 0.0
        for att in attackers:
            att_in_zone14 = (att.x > 70.0 and 20.0 < att.y < 48.0) if is_defending_right else (att.x < 35.0 and 20.0 < att.y < 48.0)
            if att_in_zone14:
                # Check if passing lane from ball to att is blocked by any defender
                blocked = False
                for d in defenders:
                    l2 = (att.x - frame.ball.x)**2 + (att.y - frame.ball.y)**2
                    if l2 > 0:
                        t = max(0.0, min(1.0, ((d.x - frame.ball.x)*(att.x - frame.ball.x) + (d.y - frame.ball.y)*(att.y - frame.ball.y)) / l2))
                        proj_x = frame.ball.x + t * (att.x - frame.ball.x)
                        proj_y = frame.ball.y + t * (att.y - frame.ball.y)
                        if math.hypot(d.x - proj_x, d.y - proj_y) < 2.5:
                            blocked = True
                            break
                if not blocked:
                    zone14_unblocked += 0.35

        passing_lane_exposure = min(1.0, max(0.05, zone14_unblocked))

        # 4. Unmarked Attacker Risk in Attacking Third
        unmarked_count = 0
        for att in attackers:
            is_in_att_third = (att.x > 65.0) if is_defending_right else (att.x < 40.0)
            if is_in_att_third:
                nearest_def = min([math.hypot(att.x - d.x, att.y - d.y) for d in defenders], default=99.0)
                if nearest_def > 5.5:
                    unmarked_count += 1
        unmarked_risk = min(1.0, unmarked_count * 0.30)

        # 5. Space Occupation Risk in Penalty Box
        in_box_count = sum(
            1 for a in attackers
            if ((a.x > 88.0 and 18.0 < a.y < 50.0) if is_defending_right else (a.x < 17.0 and 18.0 < a.y < 50.0))
        )
        space_occupation_risk = min(1.0, in_box_count * 0.35)

        # Overall Danger Score calculation (weighted sum)
        overall = round(
            0.25 * max(cb_lb_gap_risk, cb_rb_gap_risk) +
            0.25 * passing_lane_exposure +
            0.20 * unmarked_risk +
            0.15 * midfield_exposure +
            0.15 * space_occupation_risk,
            3
        )

        return DefensiveCollapseIndex(
            overall_danger=round(overall, 3),
            cb_lb_gap_risk=round(cb_lb_gap_risk, 3),
            cb_rb_gap_risk=round(cb_rb_gap_risk, 3),
            midfield_exposure=round(midfield_exposure, 3),
            passing_lane_exposure=round(passing_lane_exposure, 3),
            unmarked_attacker_risk=round(unmarked_risk, 3),
            space_occupation_risk=round(space_occupation_risk, 3),
            methodology="Direction-Aware Spatial Structural Geometry Index"
        )

