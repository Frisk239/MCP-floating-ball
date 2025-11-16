"""
MCP Floating Ball - 图像分析模块

提供图像内容分析、目标检测、图像分类等功能。
"""

import time
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageStat
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from src.core.logging import get_logger
from src.core.exceptions import VisionError


class ImageAnalyzer:
    """图像分析类"""

    def __init__(self):
        """初始化图像分析器"""
        self.logger = get_logger("vision.image_analyzer")

        # 检查依赖
        self.dependencies = {
            "opencv": OPENCV_AVAILABLE,
            "mediapipe": MEDIAPIPE_AVAILABLE
        }

        # 初始化MediaPipe
        self.mp_hands = None
        self.mp_face_mesh = None
        self.mp_pose = None
        self.hands_detector = None
        self.face_mesh_detector = None
        self.pose_detector = None

        if MEDIAPIPE_AVAILABLE:
            self._initialize_mediapipe()

        self.logger.info(f"图像分析器初始化完成，依赖状态: {self.dependencies}")

    def _initialize_mediapipe(self):
        """初始化MediaPipe"""
        try:
            # 初始化手部检测
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # 初始化面部网格检测
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # 初始化姿态检测
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.logger.info("MediaPipe初始化成功")

        except Exception as e:
            self.logger.error(f"MediaPipe初始化失败: {e}")
            # 清理已初始化的组件
            if self.hands_detector:
                self.hands_detector.close()
            if self.face_mesh_detector:
                self.face_mesh_detector.close()
            if self.pose_detector:
                self.pose_detector.close()

    def analyze_image(self, image_source: Union[str, Path, np.ndarray],
                     analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        综合分析图像

        Args:
            image_source: 图片路径或numpy数组
            analysis_types: 分析类型列表 (basic, colors, edges, faces, hands, pose)

        Returns:
            分析结果字典
        """
        start_time = time.time()

        try:
            self.logger.info(f"开始分析图像: {image_source}")

            # 设置默认分析类型
            if analysis_types is None:
                analysis_types = ["basic", "colors"]

            # 加载图片
            image = self._load_image(image_source)
            if image is None:
                raise VisionError("无法加载图片")

            results = {
                "success": True,
                "image_source": str(image_source) if isinstance(image_source, (str, Path)) else "numpy_array",
                "analysis_types": analysis_types,
                "timestamp": datetime.now().isoformat()
            }

            # 执行各类分析
            for analysis_type in analysis_types:
                try:
                    if analysis_type == "basic":
                        results["basic_analysis"] = self._basic_analysis(image)
                    elif analysis_type == "colors":
                        results["color_analysis"] = self._color_analysis(image)
                    elif analysis_type == "edges":
                        results["edge_analysis"] = self._edge_analysis(image)
                    elif analysis_type == "faces":
                        results["face_analysis"] = self._face_analysis(image)
                    elif analysis_type == "hands":
                        results["hand_analysis"] = self._hand_analysis(image)
                    elif analysis_type == "pose":
                        results["pose_analysis"] = self._pose_analysis(image)
                    else:
                        self.logger.warning(f"未知的分析类型: {analysis_type}")

                except Exception as e:
                    self.logger.error(f"{analysis_type}分析失败: {e}")
                    results[f"{analysis_type}_error"] = str(e)

            execution_time = time.time() - start_time
            results["execution_time"] = execution_time

            self.logger.info(f"图像分析完成，耗时: {execution_time:.2f}秒")
            return results

        except Exception as e:
            error_msg = f"图像分析失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "image_source": str(image_source) if isinstance(image_source, (str, Path)) else "numpy_array",
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    def _load_image(self, image_source: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """加载图片"""
        try:
            if isinstance(image_source, (str, Path)):
                # 从文件路径加载
                image = cv2.imread(str(image_source))
                if image is None:
                    raise VisionError(f"无法读取图片文件: {image_source}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_source, np.ndarray):
                # 已经是numpy数组
                return image_source
            elif hasattr(image_source, 'save'):  # PIL Image 对象
                # PIL Image对象
                if image_source.mode != 'RGB':
                    image_source = image_source.convert('RGB')
                return np.array(image_source)
            else:
                self.logger.error(f"不支持的图片格式: {type(image_source)}")
                return None

        except Exception as e:
            self.logger.error(f"加载图片失败: {e}")
            return None

    def _basic_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """基础图像分析"""
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            # 计算图片大小
            size_mb = image.nbytes / (1024 * 1024)

            # 计算亮度
            if len(image.shape) == 3:
                # RGB图像
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            brightness = np.mean(gray)

            # 计算对比度
            contrast = np.std(gray)

            return {
                "dimensions": {
                    "width": width,
                    "height": height,
                    "channels": channels
                },
                "size_mb": round(size_mb, 2),
                "aspect_ratio": round(width / height, 3),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "is_color": channels == 3,
                "is_grayscale": channels == 1
            }

        except Exception as e:
            raise VisionError(f"基础分析失败: {e}")

    def _color_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """颜色分析"""
        try:
            if len(image.shape) != 3:
                # 灰度图像
                return {
                    "dominant_colors": [],
                    "color_histogram": {},
                    "is_grayscale": True
                }

            # 计算主要颜色
            # 将图片重塑为像素列表
            pixels = image.reshape(-1, 3)

            # 使用K-means聚类找到主要颜色
            from sklearn.cluster import KMeans

            # 如果sklearn不可用，使用简化方法
            try:
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)

                dominant_colors = []
                for i, color in enumerate(colors):
                    dominant_colors.append({
                        "color": color.tolist(),
                        "percentage": round(float(np.sum(kmeans.labels_ == i)) / len(pixels) * 100, 2)
                    })

            except ImportError:
                # 简化方法：计算平均颜色
                avg_color = np.mean(pixels, axis=0).astype(int)
                dominant_colors = [{
                    "color": avg_color.tolist(),
                    "percentage": 100.0
                }]

            # 计算颜色分布
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

            return {
                "dominant_colors": dominant_colors,
                "color_distribution": {
                    "red_peaks": self._find_histogram_peaks(hist_r),
                    "green_peaks": self._find_histogram_peaks(hist_g),
                    "blue_peaks": self._find_histogram_peaks(hist_b)
                },
                "is_grayscale": False
            }

        except Exception as e:
            raise VisionError(f"颜色分析失败: {e}")

    def _find_histogram_peaks(self, histogram, num_peaks=3):
        """找到直方图的峰值"""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(histogram.flatten(), height=np.max(histogram) * 0.1, distance=20)
            peaks = peaks[:num_peaks]  # 取前几个峰值
            return peaks.tolist() if len(peaks) > 0 else []
        except ImportError:
            # 如果scipy不可用，使用简化方法
            # 找到最大的几个值
            flat_hist = histogram.flatten()
            top_indices = np.argsort(flat_hist)[-num_peaks:][::-1]
            return top_indices.tolist()

    def _edge_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """边缘分析"""
        try:
            if not OPENCV_AVAILABLE:
                raise VisionError("OpenCV不可用")

            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Canny边缘检测
            edges = cv2.Canny(gray, 100, 200)

            # 计算边缘像素比例
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_ratio = edge_pixels / total_pixels

            # 使用不同的边缘检测方法
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            return {
                "edge_density": round(edge_ratio, 4),
                "edge_count": int(edge_pixels),
                "total_pixels": total_pixels,
                "has_significant_edges": edge_ratio > 0.01
            }

        except Exception as e:
            raise VisionError(f"边缘分析失败: {e}")

    def _face_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """面部分析"""
        try:
            if not MEDIAPIPE_AVAILABLE or not self.face_mesh_detector:
                raise VisionError("MediaPipe不可用")

            results = {
                "faces_detected": 0,
                "face_landmarks": [],
                "face_bounding_boxes": []
            }

            # 执行面部检测
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_results = self.face_mesh_detector.process(rgb_image)

            if mp_results.multi_face_landmarks:
                results["faces_detected"] = len(mp_results.multi_face_landmarks)

                for face_landmarks in mp_results.multi_face_landmarks:
                    # 获取面部关键点
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        landmarks.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })

                    results["face_landmarks"].append(landmarks)

                    # 计算边界框
                    x_coords = [landmark.x for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y for landmark in face_landmarks.landmark]

                    bbox = {
                        "x": min(x_coords),
                        "y": min(y_coords),
                        "width": max(x_coords) - min(x_coords),
                        "height": max(y_coords) - min(y_coords)
                    }
                    results["face_bounding_boxes"].append(bbox)

            return results

        except Exception as e:
            raise VisionError(f"面部分析失败: {e}")

    def _hand_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """手部分析"""
        try:
            if not MEDIAPIPE_AVAILABLE or not self.hands_detector:
                raise VisionError("MediaPipe不可用")

            results = {
                "hands_detected": 0,
                "hand_landmarks": [],
                "hand_gestures": []
            }

            # 执行手部检测
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_results = self.hands_detector.process(rgb_image)

            if mp_results.multi_hand_landmarks:
                results["hands_detected"] = len(mp_results.multi_hand_landmarks)

                for hand_landmarks in mp_results.multi_hand_landmarks:
                    # 获取手部关键点
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })

                    results["hand_landmarks"].append(landmarks)

                    # 简单的手势识别
                    gesture = self._recognize_gesture(landmarks)
                    results["hand_gestures"].append(gesture)

            return results

        except Exception as e:
            raise VisionError(f"手部分析失败: {e}")

    def _pose_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """姿态分析"""
        try:
            if not MEDIAPIPE_AVAILABLE or not self.pose_detector:
                raise VisionError("MediaPipe不可用")

            results = {
                "pose_detected": False,
                "pose_landmarks": [],
                "pose_keypoints": {}
            }

            # 执行姿态检测
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_results = self.pose_detector.process(rgb_image)

            if mp_results.pose_landmarks:
                results["pose_detected"] = True

                # 获取姿态关键点
                landmarks = []
                for landmark in mp_results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

                results["pose_landmarks"] = landmarks

                # 提取关键点位置
                keypoint_names = [
                    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
                    "right_eye_inner", "right_eye", "right_eye_outer",
                    "left_ear", "right_ear", "mouth_left", "mouth_right",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
                    "left_index", "right_index", "left_thumb", "right_thumb",
                    "left_hip", "right_hip", "left_knee", "right_knee",
                    "left_ankle", "right_ankle", "left_heel", "right_heel",
                    "left_foot_index", "right_foot_index"
                ]

                for i, name in enumerate(keypoint_names):
                    if i < len(landmarks):
                        results["pose_keypoints"][name] = landmarks[i]

            return results

        except Exception as e:
            raise VisionError(f"姿态分析失败: {e}")

    def _recognize_gesture(self, landmarks: List[Dict[str, float]]) -> str:
        """简单手势识别"""
        try:
            # 这是一个简化的手势识别
            # 可以根据手部关键点的位置关系来判断手势

            # 检查是否是"OK"手势（拇指和食指接触）
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            distance = ((thumb_tip['x'] - index_tip['x']) ** 2 +
                       (thumb_tip['y'] - index_tip['y']) ** 2) ** 0.5

            if distance < 0.05:
                return "OK"

            # 检查是否是"胜利"手势（食指和中指伸出）
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            # 简单判断：食指和中指高于其他手指
            if (index_tip['y'] < ring_tip['y'] and
                middle_tip['y'] < ring_tip['y'] and
                index_tip['y'] < pinky_tip['y'] and
                middle_tip['y'] < pinky_tip['y']):
                return "Victory"

            return "Unknown"

        except Exception:
            return "Unknown"

    def get_capabilities(self) -> Dict[str, Any]:
        """获取分析能力"""
        capabilities = {
            "dependencies": self.dependencies,
            "available_analyses": ["basic", "colors"],
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        }

        if OPENCV_AVAILABLE:
            capabilities["available_analyses"].extend(["edges"])

        if MEDIAPIPE_AVAILABLE:
            capabilities["available_analyses"].extend(["faces", "hands", "pose"])

        return capabilities

    def cleanup(self):
        """清理资源"""
        try:
            if MEDIAPIPE_AVAILABLE:
                if self.hands_detector:
                    self.hands_detector.close()
                if self.face_mesh_detector:
                    self.face_mesh_detector.close()
                if self.pose_detector:
                    self.pose_detector.close()

            self.logger.info("图像分析器资源清理完成")

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

    def __del__(self):
        """析构函数"""
        self.cleanup()