import pytest
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.tactical.defensive_analysis import DefensiveAnalyzer
from src.tactical.pass_recommender import PassRecommender
from src.tactical.goalkeeper import GoalkeeperAnalyzer


def test_defensive_collapse_index():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=10)
    analyzer = DefensiveAnalyzer()

    def_index = analyzer.analyze_defensive_structure(frames[0])
    assert 0.0 <= def_index.overall_danger <= 1.0
    assert 0.0 <= def_index.cb_lb_gap_risk <= 1.0
    assert 0.0 <= def_index.passing_lane_exposure <= 1.0


def test_pass_recommender():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=10)
    recommender = PassRecommender()

    recs = recommender.recommend_passes(frames[0], passer_id=7)
    assert len(recs) > 0
    assert recs[0].score >= recs[-1].score
    assert recs[0].receiver_id != 7


def test_goalkeeper_xg():
    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=10)
    gk_analyzer = GoalkeeperAnalyzer()

    xg_prob = gk_analyzer.calculate_xg(frames[0], shooter_id=7)
    assert 0.0 <= xg_prob.goal_probability <= 1.0

    gk_rec = gk_analyzer.recommend_gk_position(frames[0], shooter_id=7)
    assert len(gk_rec.recommended_position) == 2
