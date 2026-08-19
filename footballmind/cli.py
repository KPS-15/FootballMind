import sys
import argparse
from src.data.synthetic_generator import SyntheticMatchGenerator
from src.tactical.defensive_analysis import DefensiveAnalyzer
from src.models.action_predictor import ActionPredictor
from src.tactical.pass_recommender import PassRecommender
from src.simulation.simulator import WhatIfSimulator
from src.core.types import WhatIfRequest
from training.train_action import train_action_model
from training.evaluate import evaluate_all_models


def run_demo():
    print("=" * 60)
    print(" FOOTBALLMIND: DEMO & BENCHMARK MODE")
    print("=" * 60)

    gen = SyntheticMatchGenerator()
    frames = gen.generate_sequence(num_frames=60)
    sample_frame = frames[30]

    action_predictor = ActionPredictor()
    pass_recommender = PassRecommender()
    defensive_analyzer = DefensiveAnalyzer()
    simulator = WhatIfSimulator()

    # 1. Action prediction
    pred_action = action_predictor.predict_action(sample_frame, player_id=7)
    print(f"\n[1] Next Action Prediction for Player #7:")
    print(f"    Action: {pred_action.action} (Confidence: {pred_action.confidence * 100:.0f}%)")

    # 2. Best pass recommendations
    passes = pass_recommender.recommend_passes(sample_frame, passer_id=7)
    print(f"\n[2] Best Pass Recommendations:")
    for idx, p in enumerate(passes[:3]):
        print(f"    {idx+1}. Receiver #{p.receiver_id} - Score: {p.score} (Success: {p.success_probability * 100:.0f}%)")

    # 3. Defensive danger index
    def_danger = defensive_analyzer.analyze_defensive_structure(sample_frame)
    print(f"\n[3] Defensive Collapse Index:")
    print(f"    Overall Danger: {def_danger.overall_danger * 100:.0f}%")
    print(f"    CB-LB Gap Risk: {def_danger.cb_lb_gap_risk * 100:.0f}%")

    # 4. What-If Tactical Simulation
    whatif_req = WhatIfRequest(match_id="demo", frame_index=30, modified_player_id=3, new_x=22.0, new_y=26.0)
    sim_res = simulator.simulate_scenario(sample_frame, whatif_req)
    print(f"\n[4] What-If Tactical Simulation:")
    print(f"    {sim_res.summary}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FootballMind CLI Tools")
    subparsers = parser.add_subparsers(dest="command")

    # demo command
    subparsers.add_parser("demo", help="Run full system synthetic demo")

    # evaluate command
    subparsers.add_parser("evaluate", help="Run model evaluation suite")

    # train-action command
    subparsers.add_parser("train-action", help="Train temporal action predictor model")

    # process-video command
    proc_parser = subparsers.add_parser("process-video", help="Process input video file")
    proc_parser.add_argument("video_path", type=str, help="Path to input video file")

    # api command
    api_parser = subparsers.add_parser("api", help="Start FastAPI backend server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI server")

    args = parser.parse_args()

    if args.command == "demo" or not args.command:
        run_demo()
    elif args.command == "evaluate":
        evaluate_all_models()
    elif args.command == "train-action":
        train_action_model()
    elif args.command == "process-video":
        from src.vision.video_processor import FootballVideoProcessor
        processor = FootballVideoProcessor()
        print(f"Processing video: {args.video_path}...")
        frames = processor.process_video(args.video_path)
        print(f"Successfully processed {len(frames)} frames!")
    elif args.command == "api":
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port, reload=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
