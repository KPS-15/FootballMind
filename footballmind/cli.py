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
from training.train_detector import train_yolo_detector
from training.evaluate_detector import evaluate_detector_on_dataset, benchmark_detector_latency, print_evaluation_summary


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
    subparsers.add_parser("evaluate", help="Run full model evaluation suite")

    # train-action command
    subparsers.add_parser("train-action", help="Train temporal action predictor model")

    # train-detector command
    train_det = subparsers.add_parser("train-detector", help="Train YOLO11m detector on football dataset")
    train_det.add_argument("--model", type=str, default="yolo11m.pt", help="Base model weights")
    train_det.add_argument("--data", type=str, default="datasets/data.yaml", help="Path to data.yaml")
    train_det.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_det.add_argument("--imgsz", type=int, default=1280, help="Training image resolution")
    train_det.add_argument("--batch", type=int, default=16, help="Batch size")
    train_det.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', '0')")
    train_det.add_argument("--dry-run", action="store_true", help="Validate dataset and exit without training")

    # evaluate-detector command
    eval_det = subparsers.add_parser("evaluate-detector", help="Evaluate vision detector accuracy and latency")
    eval_det.add_argument("--model", type=str, default="yolo11m.pt", help="Model to evaluate")
    eval_det.add_argument("--data", type=str, default="datasets/data.yaml", help="Path to data.yaml")
    eval_det.add_argument("--imgsz", type=int, default=1280, help="Image resolution")
    eval_det.add_argument("--benchmark-only", action="store_true", help="Run throughput latency benchmark only")

    # detect-image command
    det_img = subparsers.add_parser("detect-image", help="Run YOLO detection on a single image file")
    det_img.add_argument("image_path", type=str, help="Path to image file")
    det_img.add_argument("--model", type=str, default=None, help="YOLO model path or name")
    det_img.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    det_img.add_argument("--ball-conf", type=float, default=0.20, help="Ball confidence threshold")
    det_img.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")

    # process-video command
    proc_parser = subparsers.add_parser("process-video", help="Process input video file")
    proc_parser.add_argument("video_path", type=str, help="Path to input video file")
    proc_parser.add_argument("--output", type=str, default=None, help="Optional output path for annotated video")
    proc_parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    proc_parser.add_argument("--model", type=str, default=None, help="YOLO model weights path")

    # roboflow-workflow command
    rf_parser = subparsers.add_parser("roboflow-workflow", help="Run Roboflow hosted serverless workflow on an image")
    rf_parser.add_argument("image_path", type=str, help="Path to input image file")
    rf_parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key")
    rf_parser.add_argument("--workspace", type=str, default=None, help="Roboflow Workspace name")
    rf_parser.add_argument("--workflow-id", type=str, default=None, help="Roboflow Workflow ID")
    rf_parser.add_argument("--classes", type=str, default="ball, player, referee, goalkeeper", help="Class names")

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
    elif args.command == "train-detector":
        train_yolo_detector(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            dry_run=args.dry_run
        )
    elif args.command == "evaluate-detector":
        from pathlib import Path
        if args.benchmark_only or not Path(args.data).exists():
            bench = benchmark_detector_latency(model_name=args.model, imgsz=args.imgsz)
            print_evaluation_summary(bench)
        else:
            report = evaluate_detector_on_dataset(model_path=args.model, data_yaml=args.data, imgsz=args.imgsz)
            print_evaluation_summary(report)
    elif args.command == "detect-image":
        from src.vision.detector import FootballDetector
        detector = FootballDetector(
            model_name=args.model,
            conf_thresh=args.conf,
            ball_conf_thresh=args.ball_conf,
            imgsz=args.imgsz
        )
        print(f"Running detection on image: {args.image_path} with model '{detector.model_name}'...")
        dets = detector.detect_image(args.image_path)
        print(f"Detected {len(dets)} objects:")
        for d in dets:
            print(f" - [{d.class_name.upper():<10}] Conf: {d.confidence:.2f} | BBox: {d.bbox} | Center: {d.center}")
    elif args.command == "roboflow-workflow":
        from src.vision.roboflow_client import RoboflowWorkflowClient
        client = RoboflowWorkflowClient(
            api_key=args.api_key,
            workspace_name=args.workspace,
            workflow_id=args.workflow_id,
            classes=args.classes
        )
        print(f"Running Roboflow workflow '{client.workflow_id}' on {args.image_path}...")
        try:
            res = client.run_workflow_on_image(args.image_path)
            print("Roboflow Raw Workflow Output:")
            print(res)
            parsed = client.parse_predictions(res)
            print(f"\nParsed {len(parsed)} DetectedObjects:")
            for p in parsed:
                print(f" - [{p.class_name.upper():<10}] Conf: {p.confidence:.2f} | BBox: {p.bbox} | Center: {p.center}")
        except Exception as e:
            print(f"Error running Roboflow workflow: {e}")
    elif args.command == "process-video":
        from src.vision.video_processor import FootballVideoProcessor
        processor = FootballVideoProcessor(model_name=args.model)
        print(f"Processing video: {args.video_path}...")
        if args.output:
            frames = processor.process_and_annotate_video(args.video_path, args.output, max_frames=args.max_frames)
        else:
            frames = processor.process_video(args.video_path, max_frames=args.max_frames)
        print(f"Successfully processed {len(frames)} frames!")
    elif args.command == "api":
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port, reload=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
