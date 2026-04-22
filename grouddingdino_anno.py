import cv2
import json
import os
from pathlib import Path
import numpy as np
from mmdet.apis import DetInferencer

def mmdet_to_labelme_rectangle(img_path, pre_out, class_names, conf_thres=0.25):
    """
    将mmdetection检测结果转换为labelme矩形格式
    """
    # 读取图片获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None
        
    img_height, img_width = img.shape[:2]
    
    # 构建labelme格式的基础结构
    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    # 解析mmdetection结果
    scores = np.array(pre_out['predictions'][0]['scores'])
    labels = np.array(pre_out['predictions'][0]['labels'])
    bboxes = np.array(pre_out['predictions'][0]['bboxes'])
    
    # 根据置信度阈值过滤
    valid_indices = scores > conf_thres
    thr_labels = labels[valid_indices]
    thr_bboxes = bboxes[valid_indices]
    thr_scores = scores[valid_indices]
    
    # 处理每个检测结果
    for i in range(len(thr_bboxes)):
        bbox = thr_bboxes[i]
        label_id = int(thr_labels[i])
        confidence = thr_scores[i]
        
        # 获取类别名称
        label_name = class_names.get(label_id, f"class_{label_id}")
        
        # 边界框坐标 (x1, y1, x2, y2 格式)
        x1, y1, x2, y2 = bbox
        
        # 转换为labelme的矩形格式（用两个点表示矩形）
        points = [
            [float(x1), float(y1)],  # 左上
            [float(x2), float(y2)]   # 右下
        ]
        
        # 构建shape对象
        shape = {
            "label": label_name,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        
        labelme_data["shapes"].append(shape)
    
    return labelme_data

def auto_annotate_images(input_dir, output_dir, mmdet_model, class_names, opt):
    """
    自动化标注图片为labelme格式
    
    Args:
        input_dir: 输入图片文件夹路径
        output_dir: 输出JSON文件夹路径
        mmdet_model: mmdetection模型
        class_names: 类别名称字典
        opt: 配置参数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 遍历输入目录中的所有图片
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in supported_formats]
    
    print(f"找到 {len(image_files)} 张图片")
    
    for i, img_path in enumerate(image_files):
        print(f"处理第 {i+1}/{len(image_files)} 张图片: {img_path.name}")
        
        try:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            # 使用原有的API调用方式
            args = {
                'inputs': img, 
                'out_dir': 'outputs', 
                'texts': ". ".join(class_names.values()),  # 使用所有类别名称作为文本提示
                'pred_score_thr': opt['conf_thres'], 
                'batch_size': 1, 
                'show': False, 
                'no_save_vis': True, 
                'no_save_pred': True, 
                'print_result': False, 
                'custom_entities': False, 
            }
            
            pre_out = mmdet_model(**args)
            
            # 转换为labelme格式
            labelme_data = mmdet_to_labelme_rectangle(str(img_path), pre_out, class_names, opt['conf_thres'])
            
            # if labelme_data and len(labelme_data["shapes"]) > 0:
                # 保存JSON文件
            json_path = os.path.join(output_dir, f"{img_path.stem}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            print(f"已保存标注文件: {json_path} (包含 {len(labelme_data['shapes'])} 个目标)")
            # else:
            #     print(f"未检测到目标: {img_path.name}")
                
        except Exception as e:
            print(f"处理图片 {img_path.name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("自动化标注完成!")

def main():
    # 加载配置
    with open("./backend/flask_config.json") as fp:
        opt = json.load(fp)
    
    # 加载mmdetection模型 (保持原有方式)
    mmdet_model = DetInferencer(model=opt["det_config"], weights=opt["det_weight"], device='cuda:0')
    chunked_size = opt.pop('chunked_size')
    mmdet_model.model.test_cfg.chunked_size = chunked_size
    
    # 设置类别名称 (根据您的实际情况调整)
    # 这里使用您原来的id2name，如果没有的话可以自定义
 
        # 如果无法导入，使用默认类别
    class_names = {
            0: "挖掘机", 
            1: "铲车", 
            2: "搅拌机",
            3: "渣土车",
            4: "拖拉机",
            5: "有人",
            6: "三轮车",  
            7: "小汽车", 
            8: "雾炮车",
            9: "洒水车",
            10: "垃圾车",
            11: "压路机",
            12: "有货车",
            13: "卡车",
            14: "厢货",
            15: "罐车",
            16: "客车",
            17: "摩托车",       
            # 根据您的实际类别添加更多
    }
       
    
    # 设置输入输出路径
    input_dir = "/mnt/data/lyf/datasets/animals/animal460/images"  # 替换为你的图片文件夹路径
    output_dir = "/mnt/data/lyf/datasets/detect_base/tmp"  # 替换为你的JSON输出路径
    
    # 执行自动化标注
    auto_annotate_images(
        input_dir=input_dir,
        output_dir=output_dir,
        mmdet_model=mmdet_model,
        class_names=class_names,
        opt=opt
    )

if __name__ == "__main__":
    main()