import os
import cv2
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from src.core.types import DetectedObject


class RoboflowWorkflowClient:
    """
    Client for running Roboflow Hosted / Serverless Workflow APIs (e.g. general-segmentation-api).
    Parses segmentations and bounding boxes into standard FootballMind DetectedObject structures.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        classes: str = "ball, player, referee, goalkeeper",
        use_cache: bool = True,
    ):
        self.api_url = (
            api_url
            or os.environ.get("ROBOFLOW_API_URL")
            or "https://serverless.roboflow.com"
        )
        self.api_key = (
            api_key
            or os.environ.get("ROBOFLOW_API_KEY")
            or "6iE3b3FVMQzewTLHfWDT"
        )
        self.workspace_name = (
            workspace_name
            or os.environ.get("ROBOFLOW_WORKSPACE")
            or "k-p-shohil"
        )
        self.workflow_id = (
            workflow_id
            or os.environ.get("ROBOFLOW_WORKFLOW_ID")
            or "general-segmentation-api"
        )
        self.classes = classes
        self.use_cache = use_cache
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes InferenceHTTPClient with header-based authorization."""
        if not self.api_key:
            print("[RoboflowWorkflowClient] Notice: No API key provided.")
            return

        try:
            from inference_sdk import InferenceHTTPClient, InferenceConfiguration

            self.client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key,
            ).configure(
                InferenceConfiguration(
                    api_key_transport="header"
                )
            )
            print(f"[RoboflowWorkflowClient] Initialized client for workspace '{self.workspace_name}', workflow '{self.workflow_id}'")
        except Exception as e:
            print(f"[RoboflowWorkflowClient] Warning: Could not initialize inference_sdk client ({e}).")
            self.client = None

    def run_workflow_on_image(
        self, image_or_path: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Executes Roboflow workflow on a given image file or numpy array.
        """
        if self.client is None:
            raise RuntimeError("Roboflow client is not initialized. Check your API key and inference_sdk installation.")

        temp_file_created = False
        img_path = None

        try:
            if isinstance(image_or_path, (str, Path)):
                img_path = str(image_or_path)
            elif isinstance(image_or_path, np.ndarray):
                # Write array to temporary JPG for inference-sdk client
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                img_path = tmp.name
                tmp.close()
                temp_file_created = True
                cv2.imwrite(img_path, image_or_path)
            else:
                raise ValueError(f"Unsupported image type: {type(image_or_path)}")

            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={
                    "image": img_path
                },
                parameters={
                    "classes": self.classes
                },
                use_cache=self.use_cache
            )
            return result
        finally:
            if temp_file_created and img_path and os.path.exists(img_path):
                try:
                    os.unlink(img_path)
                except Exception:
                    pass

    def parse_predictions(self, workflow_result: Any) -> List[DetectedObject]:
        """
        Extracts bounding boxes and class names from Roboflow workflow output format.
        Supports standard object-detection, segmentation, and custom workflow outputs.
        """
        detections: List[DetectedObject] = []
        if not workflow_result:
            return detections

        raw_preds = []
        if isinstance(workflow_result, dict):
            # Try common workflow result output keys
            if "output" in workflow_result:
                raw_preds = workflow_result["output"]
            elif "predictions" in workflow_result:
                raw_preds = workflow_result["predictions"]
            elif "detections" in workflow_result:
                raw_preds = workflow_result["detections"]
            else:
                # Search first list in dict values
                for v in workflow_result.values():
                    if isinstance(v, list):
                        raw_preds = v
                        break
                    elif isinstance(v, dict) and ("predictions" in v or "detections" in v):
                        raw_preds = v.get("predictions") or v.get("detections", [])
                        break
        elif isinstance(workflow_result, list):
            # Workflow might return list of results
            if len(workflow_result) > 0 and isinstance(workflow_result[0], dict):
                first = workflow_result[0]
                raw_preds = first.get("predictions") or first.get("detections") or workflow_result
            else:
                raw_preds = workflow_result

        if not isinstance(raw_preds, list):
            return detections

        for idx, item in enumerate(raw_preds):
            if not isinstance(item, dict):
                continue

            cls_name = str(item.get("class") or item.get("class_name") or item.get("label") or "player").lower().strip()
            conf = float(item.get("confidence") or item.get("score") or 0.85)

            # Normalize class
            if any(k in cls_name for k in ["ball", "football", "soccer", "sports ball"]):
                normalized_cls = "ball"
            elif any(k in cls_name for k in ["goalkeeper", "gk", "keeper"]):
                normalized_cls = "goalkeeper"
            elif any(k in cls_name for k in ["referee", "ref", "judge"]):
                normalized_cls = "referee"
            else:
                normalized_cls = "player"

            # Parse bounding box coordinates
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
            if "x" in item and "y" in item and "width" in item and "height" in item:
                # Center coordinates format
                cx, cy = float(item["x"]), float(item["y"])
                w, h = float(item["width"]), float(item["height"])
                x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
            elif "bbox" in item and isinstance(item["bbox"], (list, tuple)) and len(item["bbox"]) == 4:
                x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
            elif all(k in item for k in ["x_min", "y_min", "x_max", "y_max"]):
                x1, y1, x2, y2 = float(item["x_min"]), float(item["y_min"]), float(item["x_max"]), float(item["y_max"])
            else:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append(
                DetectedObject(
                    track_id=idx + 1,
                    class_name=normalized_cls,
                    confidence=round(conf, 3),
                    bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    center=[round(cx, 1), round(cy, 1)],
                )
            )

        return detections

    def detect_frame(self, frame: np.ndarray) -> List[DetectedObject]:
        """Runs workflow and returns standard DetectedObjects list."""
        if frame is None or frame.size == 0:
            return []
        try:
            res = self.run_workflow_on_image(frame)
            return self.parse_predictions(res)
        except Exception as e:
            print(f"[RoboflowWorkflowClient] Inference error: {e}")
            return []
