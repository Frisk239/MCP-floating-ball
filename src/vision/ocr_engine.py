"""
MCP Floating Ball - OCR引擎模块

提供文字识别功能，支持中英文OCR。
"""

import time
import platform
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from datetime import datetime

try:
    import cv2
    import numpy as np
    from PIL import Image
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from src.core.logging import get_logger
from src.core.exceptions import VisionError


class OCREngine:
    """OCR引擎类"""

    def __init__(self, engine: str = "tesseract", lang: str = "chi_sim+eng"):
        """
        初始化OCR引擎

        Args:
            engine: OCR引擎类型 (tesseract, easyocr)
            lang: 语言设置
        """
        self.logger = get_logger("vision.ocr")
        self.engine_type = engine
        self.lang = lang

        # 初始化引擎
        self.ocr_engine = None
        self._initialize_engine()

        # 配置参数
        self.tesseract_config = r'--oem 3 --psm 6'
        self.easyocr_gpu = True

        self.logger.info(f"OCR引擎初始化完成，引擎类型: {engine}, 语言: {lang}")

    def _initialize_engine(self):
        """初始化OCR引擎"""
        try:
            if self.engine_type.lower() == "tesseract":
                if not TESSERACT_AVAILABLE:
                    raise VisionError("Tesseract未安装，请安装: pip install pytesseract")

                # 检查Tesseract是否可用
                try:
                    pytesseract.get_tesseract_version()
                    self.ocr_engine = pytesseract
                    self.logger.info("Tesseract引擎初始化成功")
                except Exception as e:
                    raise VisionError(f"Tesseract不可用: {e}")

            elif self.engine_type.lower() == "easyocr":
                if not EASYOCR_AVAILABLE:
                    raise VisionError("EasyOCR未安装，请安装: pip install easyocr")

                try:
                    self.ocr_engine = easyocr.Reader(['ch_sim', 'en'], gpu=self.easyocr_gpu)
                    self.logger.info("EasyOCR引擎初始化成功")
                except Exception as e:
                    self.logger.warning(f"EasyOCR初始化失败，尝试使用CPU: {e}")
                    self.easyocr_gpu = False
                    self.ocr_engine = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                    self.logger.info("EasyOCR引擎初始化成功（CPU模式）")

            else:
                raise ValueError(f"不支持的OCR引擎: {self.engine_type}")

        except Exception as e:
            self.logger.error(f"OCR引擎初始化失败: {e}")
            raise VisionError(f"OCR引擎初始化失败: {e}")

    def recognize_text(self, image_source: Union[str, Path, np.ndarray],
                      save_result: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        识别图片中的文字

        Args:
            image_source: 图片路径或numpy数组
            save_result: 是否保存结果
            output_dir: 结果输出目录

        Returns:
            OCR识别结果字典
        """
        start_time = time.time()

        try:
            self.logger.info(f"开始OCR识别: {image_source}")

            # 加载图片
            image = self._load_image(image_source)
            if image is None:
                raise VisionError("无法加载图片")

            # 图片预处理
            processed_image = self._preprocess_image(image)

            # 执行OCR
            if self.engine_type.lower() == "tesseract":
                ocr_result = self._tesseract_ocr(processed_image)
            elif self.engine_type.lower() == "easyocr":
                ocr_result = self._easyocr_ocr(processed_image)
            else:
                raise ValueError(f"不支持的OCR引擎: {self.engine_type}")

            # 后处理结果
            result = self._postprocess_result(ocr_result, image_source, save_result, output_dir)

            execution_time = time.time() - start_time
            result["execution_time"] = execution_time

            self.logger.info(f"OCR识别完成，耗时: {execution_time:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"OCR识别失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "engine": self.engine_type,
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    def _load_image(self, image_source: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """加载图片"""
        try:
            if isinstance(image_source, (str, Path)):
                # 从文件路径加载
                image = Image.open(image_source)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.array(image)
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

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图片预处理"""
        try:
            if not OPENCV_AVAILABLE:
                return image

            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 降噪
            denoised = cv2.fastNlMeansDenoising(gray)

            # 二值化
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            self.logger.warning(f"图片预处理失败，使用原图: {e}")
            return image

    def _tesseract_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """使用Tesseract进行OCR"""
        try:
            # 获取详细结果
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.tesseract_config,
                output_type=Output.DICT
            )

            # 提取文本和位置信息
            results = []
            full_text = ""

            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # 过滤低置信度结果
                    text = data['text'][i].strip()
                    if text:
                        results.append({
                            'text': text,
                            'confidence': float(data['conf'][i]),
                            'bbox': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            }
                        })
                        full_text += text + " "

            return {
                'engine': 'tesseract',
                'text': full_text.strip(),
                'words': results,
                'data': data
            }

        except Exception as e:
            raise VisionError(f"Tesseract OCR失败: {e}")

    def _easyocr_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """使用EasyOCR进行OCR"""
        try:
            # 执行OCR
            results = self.ocr_engine.readtext(image)

            # 处理结果
            words = []
            full_text = ""

            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 过滤低置信度结果
                    words.append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': {
                            'x': int(bbox[0][0]),
                            'y': int(bbox[0][1]),
                            'width': int(bbox[2][0] - bbox[0][0]),
                            'height': int(bbox[2][1] - bbox[0][1])
                        }
                    })
                    full_text += text + " "

            return {
                'engine': 'easyocr',
                'text': full_text.strip(),
                'words': words,
                'raw_results': results
            }

        except Exception as e:
            raise VisionError(f"EasyOCR OCR失败: {e}")

    def _postprocess_result(self, ocr_result: Dict[str, Any], image_source: Union[str, Path, np.ndarray],
                           save_result: bool, output_dir: Optional[str]) -> Dict[str, Any]:
        """后处理OCR结果"""
        result = {
            'success': True,
            'engine': ocr_result['engine'],
            'text': ocr_result['text'],
            'words': ocr_result['words'],
            'word_count': len(ocr_result['words']),
            'confidence_avg': self._calculate_avg_confidence(ocr_result['words']),
            'image_source': str(image_source) if isinstance(image_source, (str, Path)) else 'numpy_array',
            'timestamp': datetime.now().isoformat()
        }

        # 保存结果
        if save_result:
            output_path = self._save_result(result, output_dir)
            result['result_file'] = str(output_path)

        return result

    def _calculate_avg_confidence(self, words: List[Dict[str, Any]]) -> float:
        """计算平均置信度"""
        if not words:
            return 0.0

        total_confidence = sum(word['confidence'] for word in words)
        return total_confidence / len(words)

    def _save_result(self, result: Dict[str, Any], output_dir: Optional[str]) -> Path:
        """保存OCR结果"""
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("./ocr_results")

        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_result_{timestamp}.json"
        filepath = output_path / filename

        # 保存为JSON
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self.logger.info(f"OCR结果已保存: {filepath}")
        return filepath

    def extract_text_by_region(self, image_source: Union[str, Path, np.ndarray],
                             region: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        提取指定区域的文字

        Args:
            image_source: 图片源
            region: 区域坐标 (x, y, width, height)

        Returns:
            OCR结果字典
        """
        try:
            self.logger.info(f"提取区域文字: {region}")

            # 加载图片
            image = self._load_image(image_source)
            if image is None:
                raise VisionError("无法加载图片")

            # 裁剪区域
            x, y, w, h = region
            cropped_image = image[y:y+h, x:x+w]

            # 执行OCR
            ocr_result = self.recognize_text(cropped_image, save_result=False)
            ocr_result['region'] = region

            return ocr_result

        except Exception as e:
            error_msg = f"区域文字提取失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "engine": self.engine_type,
                "region": region,
                "error": error_msg
            }

    def get_available_languages(self) -> Dict[str, Any]:
        """获取可用语言列表"""
        try:
            if self.engine_type.lower() == "tesseract":
                try:
                    languages = pytesseract.get_languages(config='')
                    return {
                        "engine": "tesseract",
                        "available_languages": languages,
                        "current_language": self.lang
                    }
                except Exception as e:
                    return {
                        "engine": "tesseract",
                        "available_languages": ["chi_sim", "eng"],
                        "current_language": self.lang,
                        "error": str(e)
                    }

            elif self.engine_type.lower() == "easyocr":
                return {
                    "engine": "easyocr",
                    "available_languages": ["ch_sim", "en"],
                    "current_language": self.lang,
                    "note": "EasyOCR主要支持中文简体和英文"
                }

        except Exception as e:
            self.logger.error(f"获取语言列表失败: {e}")
            return {
                "engine": self.engine_type,
                "error": str(e)
            }

    def get_engine_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        info = {
            "engine_type": self.engine_type,
            "current_language": self.lang,
            "tesseract_config": self.tesseract_config if self.engine_type == "tesseract" else None,
            "easyocr_gpu": self.easyocr_gpu if self.engine_type == "easyocr" else None,
            "available_features": {
                "opencv_available": OPENCV_AVAILABLE,
                "tesseract_available": TESSERACT_AVAILABLE,
                "easyocr_available": EASYOCR_AVAILABLE
            }
        }

        # 获取版本信息
        try:
            if self.engine_type == "tesseract" and TESSERACT_AVAILABLE:
                info["tesseract_version"] = str(pytesseract.get_tesseract_version())
        except Exception:
            pass

        return info

    def switch_engine(self, engine: str, lang: Optional[str] = None):
        """切换OCR引擎"""
        self.logger.info(f"切换OCR引擎: {self.engine_type} -> {engine}")

        self.engine_type = engine
        if lang:
            self.lang = lang

        self._initialize_engine()

    def batch_recognize(self, image_paths: List[Union[str, Path]],
                       output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量OCR识别"""
        results = []

        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"处理第 {i+1}/{len(image_paths)} 张图片: {image_path}")
                result = self.recognize_text(image_path, save_result=True, output_dir=output_dir)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                error_result = {
                    "success": False,
                    "image_path": str(image_path),
                    "error": str(e),
                    "batch_index": i
                }
                results.append(error_result)

        return results