import math
import random
from typing import List
from src.core.types import PlayerState, BallState, FrameTacticalState


class SyntheticMatchGenerator:
    """
    Generates realistic 2D tactical tracking frames for FootballMind benchmark,
    demo mode, unit testing, and model baseline training.
    """

    def __init__(self, fps: int = 15, pitch_length: float = 105.0, pitch_width: float = 68.0, seed: int = 42):
        self.fps = fps
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        random.seed(seed)

    def generate_sequence(self, num_frames: int = 300) -> List[FrameTacticalState]:
        frames: List[FrameTacticalState] = []
        dt = 1.0 / self.fps

        # Initial formations (4-3-3 Home vs 4-4-2 Away)
        home_initials = [
            (5.0, 34.0),   # GK (#1)
            (25.0, 10.0),  # LB (#3)
            (22.0, 26.0),  # CB (#4)
            (22.0, 42.0),  # CB (#5)
            (25.0, 58.0),  # RB (#2)
            (45.0, 22.0),  # CM (#6)
            (50.0, 34.0),  # CM (#8)
            (45.0, 46.0),  # CM (#10)
            (70.0, 12.0),  # LW (#11)
            (72.0, 34.0),  # ST (#9)
            (70.0, 56.0),  # RW (#7)
        ]

        away_initials = [
            (100.0, 34.0), # GK (#12)
            (80.0, 12.0),  # RB (#13)
            (78.0, 26.0),  # CB (#14)
            (78.0, 42.0),  # CB (#15)
            (80.0, 54.0),  # LB (#16)
            (60.0, 12.0),  # RM (#17)
            (58.0, 26.0),  # CM (#18)
            (58.0, 42.0),  # CM (#19)
            (60.0, 54.0),  # LM (#20)
            (48.0, 28.0),  # ST (#21)
            (48.0, 40.0),  # ST (#22)
        ]

        # Current positions
        home_pos = [list(pt) for pt in home_initials]
        away_pos = [list(pt) for pt in away_initials]
        ball_pos = [52.0, 34.0]

        # Ball possession state
        ball_carrier_id = 8  # Home CM #8 starts with ball
        ball_target_pos = None
        pass_progress = 0.0

        for frame_idx in range(num_frames):
            timestamp = frame_idx * dt
            players: List[PlayerState] = []

            # Determine possession & target
            if frame_idx % 45 == 0 and frame_idx > 0:
                # Initiate a pass to teammate every ~3s
                possessor_idx = (frame_idx // 45) % len(home_initials)
                ball_carrier_id = possessor_idx + 1

            # Update Home players
            for i, pos in enumerate(home_pos):
                pid = i + 1
                target = list(home_initials[i])
                
                # Attacking movement towards opponent goal
                target[0] += math.sin(frame_idx * 0.05 + i) * 3.5 + 5.0
                target[1] += math.cos(frame_idx * 0.04 + i) * 2.5

                # Move towards target
                vx = (target[0] - pos[0]) * 0.1
                vy = (target[1] - pos[1]) * 0.1
                pos[0] = max(1.0, min(self.pitch_length - 1.0, pos[0] + vx * dt * 5.0))
                pos[1] = max(1.0, min(self.pitch_width - 1.0, pos[1] + vy * dt * 5.0))
                speed = math.hypot(vx, vy)
                direction = math.degrees(math.atan2(vy, vx)) % 360

                ball_dist = math.hypot(pos[0] - ball_pos[0], pos[1] - ball_pos[1])
                possession_prob = max(0.0, 1.0 - ball_dist / 5.0) if ball_dist < 5.0 else 0.0

                players.append(PlayerState(
                    id=pid,
                    team="home",
                    x=round(pos[0], 2),
                    y=round(pos[1], 2),
                    velocity_x=round(vx, 2),
                    velocity_y=round(vy, 2),
                    speed=round(speed, 2),
                    direction=round(direction, 1),
                    acceleration=round(random.uniform(-0.5, 0.5), 2),
                    body_orientation=round(direction, 1),
                    ball_distance=round(ball_dist, 2),
                    possession_probability=round(possession_prob, 2)
                ))

            # Update Away players (defensive shift)
            for i, pos in enumerate(away_pos):
                pid = i + 12
                target = list(away_initials[i])
                # Shift towards ball
                target[0] += (ball_pos[0] - target[0]) * 0.2
                target[1] += (ball_pos[1] - target[1]) * 0.15

                vx = (target[0] - pos[0]) * 0.1
                vy = (target[1] - pos[1]) * 0.1
                pos[0] = max(1.0, min(self.pitch_length - 1.0, pos[0] + vx * dt * 5.0))
                pos[1] = max(1.0, min(self.pitch_width - 1.0, pos[1] + vy * dt * 5.0))
                speed = math.hypot(vx, vy)
                direction = math.degrees(math.atan2(vy, vx)) % 360

                ball_dist = math.hypot(pos[0] - ball_pos[0], pos[1] - ball_pos[1])
                possession_prob = max(0.0, 0.9 - ball_dist / 5.0) if ball_dist < 5.0 else 0.0

                players.append(PlayerState(
                    id=pid,
                    team="away",
                    x=round(pos[0], 2),
                    y=round(pos[1], 2),
                    velocity_x=round(vx, 2),
                    velocity_y=round(vy, 2),
                    speed=round(speed, 2),
                    direction=round(direction, 1),
                    acceleration=round(random.uniform(-0.5, 0.5), 2),
                    body_orientation=round(direction, 1),
                    ball_distance=round(ball_dist, 2),
                    possession_probability=round(possession_prob, 2)
                ))

            # Update Ball position near ball carrier
            carrier_player = next((p for p in players if p.id == ball_carrier_id), players[6])
            ball_pos = [carrier_player.x + 0.5, carrier_player.y + 0.2]

            ball_state = BallState(
                x=round(ball_pos[0], 2),
                y=round(ball_pos[1], 2),
                velocity_x=round(carrier_player.velocity_x, 2),
                velocity_y=round(carrier_player.velocity_y, 2),
                speed=round(carrier_player.speed, 2),
                possession_player_id=carrier_player.id,
                possession_team=carrier_player.team
            )

            frames.append(FrameTacticalState(
                frame_index=frame_idx,
                timestamp=round(timestamp, 2),
                players=players,
                ball=ball_state,
                attacking_team="home",
                defensive_team="away"
            ))

        return frames

    def get_ground_truth_action(self, frame: FrameTacticalState, player_id: int) -> int:
        """
        Determines ground truth action label (0..7) based on tactical state.
        0: PASS, 1: DRIBBLE, 2: SHOT, 3: CROSS, 4: TACKLE, 5: CARRY, 6: HOLD, 7: CLEARANCE
        """
        p = next((p for p in frame.players if p.id == player_id), None)
        if not p:
            return 6  # HOLD

        target_goal_x = 105.0 if p.team == "home" else 0.0
        dist_goal = math.hypot(p.x - target_goal_x, p.y - 34.0)

        opponents = [o for o in frame.players if o.team != p.team]
        opp_dist = min([math.hypot(p.x - o.x, p.y - o.y) for o in opponents], default=99.0)

        has_ball = (frame.ball.possession_player_id == p.id)

        if has_ball:
            if dist_goal < 22.0 and abs(p.y - 34.0) < 18.0:
                return 2  # SHOT
            if p.x > 68.0 and (p.y < 16.0 or p.y > 52.0):
                return 3  # CROSS
            if p.x < 25.0 and opp_dist < 4.0:
                return 7  # CLEARANCE
            if opp_dist < 5.0:
                return 0  # PASS
            if p.speed > 2.2:
                return 1  # DRIBBLE
            if p.speed > 1.0:
                return 5  # CARRY
            return 6      # HOLD
        else:
            if p.ball_distance < 2.5 and p.team != frame.ball.possession_team:
                return 4  # TACKLE
            return 6      # HOLD

