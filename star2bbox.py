import os
import json
import copy
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import warnings

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    warnings.warn("shapely库未安装，无法进行精确的多边形相交计算。请使用 'pip install shapely' 安装。")

class LabelmeDatasetProcessor:
    def __init__(self, img_dir: str, json_dir: str, output_dir: str, 
                 target_width: int = 1920, target_height: int = 1080,
                 overlap: float = 0.1,
                 min_overlap_area: int = 10000,  # 100*100 最小重叠面积阈值
                 min_overlap_ratio_to_window: float = 0.2,  # 重叠面积占window的20%
                 min_overlap_ratio_to_object: float = 0.9):  # 重叠面积占目标的90%
        
        if not SHAPELY_AVAILABLE:
            raise ImportError("需要安装shapely库。请使用 'pip install shapely' 安装。")
        
        self.img_dir = Path(img_dir)
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.target_width = target_width
        self.target_height = target_height
        self.overlap = overlap
        
        # 重叠过滤参数
        self.min_overlap_area = min_overlap_area
        self.min_overlap_ratio_to_window = min_overlap_ratio_to_window
        self.min_overlap_ratio_to_object = min_overlap_ratio_to_object
        
        self.output_img_dir = self.output_dir / "images"
        self.output_json_dir = self.output_dir / "annotations"
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.output_json_dir.mkdir(parents=True, exist_ok=True)

    def get_image_files(self) -> List[Path]:
        """获取所有图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.img_dir.glob(f"*{ext}"))
            image_files.extend(self.img_dir.glob(f"*{ext.upper()}"))
        return sorted(image_files)

    def calculate_sliding_windows(self, img_width: int, img_height: int) -> List[Tuple]:
        """计算滑动窗口，确保覆盖边缘"""
        windows = []
        stride_x = int(self.target_width * (1 - self.overlap))
        stride_y = int(self.target_height * (1 - self.overlap))
        
        # 如果图像比目标尺寸小，直接返回整个图像
        if img_width <= self.target_width and img_height <= self.target_height:
            return [(0, 0, img_width, img_height)]
        
        # 计算滑动窗口
        y_positions = []
        y = 0
        while y < img_height:
            if y + self.target_height > img_height:
                y = max(0, img_height - self.target_height)
            y_positions.append(y)
            if y + self.target_height >= img_height:
                break
            y += stride_y
        
        x_positions = []
        x = 0
        while x < img_width:
            if x + self.target_width > img_width:
                x = max(0, img_width - self.target_width)
            x_positions.append(x)
            if x + self.target_width >= img_width:
                break
            x += stride_x
        
        # 生成所有窗口
        for y in y_positions:
            for x in x_positions:
                # 确保窗口不超出图像边界
                actual_width = min(self.target_width, img_width - x)
                actual_height = min(self.target_height, img_height - y)
                windows.append((x, y, actual_width, actual_height))
        
        return windows

    def crop_and_pad_image(self, img: Image.Image, window: Tuple[int, int, int, int]) -> Image.Image:
        """裁剪图像并进行必要的填充"""
        wx, wy, ww, wh = window
        
        # 裁剪图像
        tile = img.crop((wx, wy, wx + ww, wy + wh))
        
        # 如果裁剪的图像小于目标尺寸，进行填充
        if tile.width < self.target_width or tile.height < self.target_height:
            # 创建新的目标尺寸图像（黑色背景）
            new_tile = Image.new('RGB', (self.target_width, self.target_height), (0, 0, 0))
            # 将裁剪的图像粘贴到新图像上
            new_tile.paste(tile, (0, 0))
            return new_tile
        
        return tile

    def process_annotations_for_window(self, annotations: Dict, 
                                     window: Tuple[int, int, int, int],
                                     image_name: str) -> Dict:
        """核心逻辑：使用shapely进行精确的几何计算"""
        wx, wy, ww, wh = window
        window_area = ww * wh
        
        # 创建窗口的多边形表示
        window_polygon = box(wx, wy, wx + ww, wy + wh)
        
        # 深拷贝原始数据，避免污染
        new_anno = copy.deepcopy(annotations)
        new_anno['imagePath'] = image_name
        new_anno['imageWidth'] = self.target_width  # 使用目标宽度（考虑填充）
        new_anno['imageHeight'] = self.target_height  # 使用目标高度（考虑填充）
        if 'imageData' in new_anno: 
            new_anno['imageData'] = None
        
        new_shapes = []
        for shape in annotations.get('shapes', []):
            shape_type = shape.get('shape_type', '')
            points = shape['points']
            
            if shape_type == 'polygon':
                # 创建旋转矩形的多边形
                # 注意：确保多边形是闭合的（首尾点相同）
                polygon_points = [(float(p[0]), float(p[1])) for p in points]
                # 如果首尾点不相同，添加第一个点使其闭合
                if polygon_points[0] != polygon_points[-1]:
                    polygon_points.append(polygon_points[0])
                
                try:
                    obj_polygon = Polygon(polygon_points)
                    
                    # 检查多边形是否有效
                    if not obj_polygon.is_valid:
                        # 尝试修复无效多边形
                        obj_polygon = obj_polygon.buffer(0)
                    
                    if obj_polygon.is_valid and not obj_polygon.is_empty:
                        # 计算相交部分
                        intersection = obj_polygon.intersection(window_polygon)
                        
                        if not intersection.is_empty:
                            # 计算相交面积
                            intersection_area = intersection.area
                            obj_area = obj_polygon.area
                            
                            # 检查是否满足保留条件
                            overlap_ratio_to_window = intersection_area / window_area if window_area > 0 else 0
                            overlap_ratio_to_object = intersection_area / obj_area if obj_area > 0 else 0
                            
                            if (overlap_ratio_to_window >= self.min_overlap_ratio_to_window or
                                intersection_area >= self.min_overlap_area or
                                overlap_ratio_to_object >= self.min_overlap_ratio_to_object):
                                
                                # 处理相交多边形
                                if intersection.geom_type == 'Polygon':
                                    # 获取多边形坐标（排除最后一个重复的点）
                                    coords = list(intersection.exterior.coords)[:-1]
                                elif intersection.geom_type == 'MultiPolygon':
                                    # 如果是多个多边形，取面积最大的一个
                                    max_poly = max(intersection.geoms, key=lambda p: p.area)
                                    coords = list(max_poly.exterior.coords)[:-1]
                                else:
                                    # 其他几何类型，跳过
                                    continue
                                
                                # 转换到窗口坐标系
                                window_coords = []
                                for x, y in coords:
                                    # 相对于窗口的坐标
                                    rel_x = max(0, min(ww - 1, x - wx))
                                    rel_y = max(0, min(wh - 1, y - wy))
                                    window_coords.append([float(rel_x), float(rel_y)])
                                
                                # 过滤极小框
                                if len(window_coords) >= 3:
                                    x_coords = [p[0] for p in window_coords]
                                    y_coords = [p[1] for p in window_coords]
                                    xmin, xmax = min(x_coords), max(x_coords)
                                    ymin, ymax = min(y_coords), max(y_coords)
                                    
                                    if (xmax - xmin) > 2 and (ymax - ymin) > 2:
                                        new_shape = copy.deepcopy(shape)
                                        # 存储为rectangle，使用两个点表示
                                        new_shape['points'] = [[xmin, ymin], [xmax, ymax]]
                                        # 保持为rectangle类型
                                        new_shape['shape_type'] = 'rectangle'
                                        new_shapes.append(new_shape)
                
                except Exception as e:
                    print(f"处理多边形时出错: {e}")
                    continue
            
            elif shape_type == 'rectangle':
                # 对于普通矩形，转换为多边形处理
                points = shape['points']
                if len(points) == 2:
                    # 两个点的情况，转换为四个角点
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    polygon_points = [
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
                    ]
                else:
                    polygon_points = points
                
                # 使用多边形逻辑处理
                polygon_points = [(float(p[0]), float(p[1])) for p in polygon_points]
                obj_polygon = Polygon(polygon_points)
                
                if obj_polygon.is_valid and not obj_polygon.is_empty:
                    # 计算相交部分
                    intersection = obj_polygon.intersection(window_polygon)
                    
                    if not intersection.is_empty:
                        # 计算相交面积
                        intersection_area = intersection.area
                        obj_area = obj_polygon.area
                        
                        # 检查是否满足保留条件
                        overlap_ratio_to_window = intersection_area / window_area if window_area > 0 else 0
                        overlap_ratio_to_object = intersection_area / obj_area if obj_area > 0 else 0
                        
                        if (overlap_ratio_to_window >= self.min_overlap_ratio_to_window or
                            intersection_area >= self.min_overlap_area or
                            overlap_ratio_to_object >= self.min_overlap_ratio_to_object):
                            
                            # 处理相交多边形
                            if intersection.geom_type == 'Polygon':
                                coords = list(intersection.exterior.coords)[:-1]
                            elif intersection.geom_type == 'MultiPolygon':
                                max_poly = max(intersection.geoms, key=lambda p: p.area)
                                coords = list(max_poly.exterior.coords)[:-1]
                            else:
                                continue
                            
                            # 转换到窗口坐标系
                            window_coords = []
                            for x, y in coords:
                                rel_x = max(0, min(ww - 1, x - wx))
                                rel_y = max(0, min(wh - 1, y - wy))
                                window_coords.append([float(rel_x), float(rel_y)])
                            
                            # 过滤极小框
                            if len(window_coords) >= 3:
                                x_coords = [p[0] for p in window_coords]
                                y_coords = [p[1] for p in window_coords]
                                xmin, xmax = min(x_coords), max(x_coords)
                                ymin, ymax = min(y_coords), max(y_coords)
                                
                                if (xmax - xmin) > 2 and (ymax - ymin) > 2:
                                    new_shape = copy.deepcopy(shape)
                                    # 对于矩形，返回两个点
                                    new_shape['points'] = [
                                        [float(xmin), float(ymin)],
                                        [float(xmax), float(ymax)]
                                    ]
                                    new_shapes.append(new_shape)
            
            else:
                # 对于其他类型的标注（如点、线），保留原始逻辑
                points = np.array(shape['points'])
                rel_points = points - [wx, wy]
                rel_points[:, 0] = np.clip(rel_points[:, 0], 0, self.target_width - 1)
                rel_points[:, 1] = np.clip(rel_points[:, 1], 0, self.target_height - 1)
                
                # 检查点是否在窗口内
                if np.any((rel_points >= 0) & (rel_points < [self.target_width, self.target_height])):
                    new_shape = copy.deepcopy(shape)
                    new_shape['points'] = rel_points.tolist()
                    new_shapes.append(new_shape)
        
        new_anno['shapes'] = new_shapes
        return new_anno

    def process_image(self, image_path: Path):
        """处理单张图像"""
        json_path = self.json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            print(f"警告: 找不到对应的JSON文件: {json_path}")
            return

        try:
            # 使用PIL打开图像
            img = Image.open(str(image_path))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            windows = self.calculate_sliding_windows(img.width, img.height)
            
            for i, window in enumerate(windows):
                wx, wy, ww, wh = window
                
                # 裁剪和填充图像
                tile = self.crop_and_pad_image(img, window)
                
                # 处理标注
                window_img_name = f"{image_path.stem}_tile_{i:03d}.jpg"
                window_anno = self.process_annotations_for_window(
                    annotations, window, window_img_name
                )
                out_json_path = self.output_json_dir / f"{window_img_name[:-4]}.json"
                
                # 保存图像和JSON
                tile.save(str(self.output_img_dir / window_img_name), 
                         quality=95, subsampling=0)
                
                with open(out_json_path, 'w', encoding='utf-8') as f:
                    json.dump(window_anno, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n处理 {image_path.name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def process_dataset(self):
        """处理整个数据集"""
        image_files = self.get_image_files()
        print(f"找到 {len(image_files)} 张图像，准备处理...")
        
        for img_path in tqdm(image_files, desc="处理进度"):
            self.process_image(img_path)
        
        print(f"\n处理完成！结果保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='处理Labelme格式的大型图像数据集')
    parser.add_argument('--img-dir', type=str, default='E:\张永军\STAR//train//img', help='图像文件夹路径')
    parser.add_argument('--json-dir', type=str, default='E:\张永军\STAR//train//object-json', help='JSON标注文件夹路径')
    parser.add_argument('--output-dir', type=str, default='E:\张永军\STAR//train//output', help='输出文件夹路径')
    parser.add_argument('--width', type=int, default=1920, help='目标宽度 (默认: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='目标高度 (默认: 1080)')
    parser.add_argument('--overlap', type=float, default=0.3,
                       help='切割重叠比例 (默认: 0.3, 范围: 0-0.5)')
    parser.add_argument('--min-overlap-area', type=int, default=10000,
                       help='最小重叠面积阈值 (默认: 100*100)')
    parser.add_argument('--min-overlap-ratio-to-window', type=float, default=0.2,
                       help='重叠面积占窗口的最小比例 (默认: 0.2)')
    parser.add_argument('--min-overlap-ratio-to-object', type=float, default=0.6,
                       help='重叠面积占目标的最小比例 (默认: 0.6)')

    args = parser.parse_args()
    
    # 验证参数
    if args.overlap < 0 or args.overlap > 0.5:
        print("警告: 重叠比例应在0-0.5之间，已调整为0.1")
        args.overlap = 0.1
    
    # 创建处理器并运行
    processor = LabelmeDatasetProcessor(
        img_dir=args.img_dir,
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        target_width=args.width,
        target_height=args.height,
        overlap=args.overlap,
        min_overlap_area=args.min_overlap_area,
        min_overlap_ratio_to_window=args.min_overlap_ratio_to_window,
        min_overlap_ratio_to_object=args.min_overlap_ratio_to_object
    )
    
    processor.process_dataset()

if __name__ == "__main__":
    main()