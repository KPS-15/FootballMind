# Limitations & Future Work

## Current Limitations
1. **Camera Occlusion & Broadcast Pan**: Single broadcast camera angles can cause player occlusion when players move off-screen. Multi-camera stitching is recommended for full-pitch tracking.
2. **Synthetic Baseline Calibration**: Synthetic benchmark datasets allow offline testing without expensive dataset downloads, but fine-tuning on official SoccerNet tracking data improves real-world accuracy.
3. **CPU Real-Time Throughput**: On machines without dedicated GPUs, full frame-by-frame YOLO inference is throttled; frame skipping (FPS = 15 -> 5) is utilized.

## Future Enhancements
- Graph Neural Networks (GNNs) for multi-agent relational spatial modeling.
- 3D pose estimation and body orientation vectors.
- Automated tactical commentary text generation using multimodal LLMs.
