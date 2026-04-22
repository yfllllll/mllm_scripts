import fiftyone as fo  
import os  
import json  
import fiftyone.zoo as foz  
import fiftyone.brain as fob  
  


def load_labelme_labels(label_path, image_width, image_height):  
    detections = []  
    if os.path.getsize(label_path) == 0:  
        print(f"文件 {label_path} 为空。")  
        return []  
    with open(label_path, "r") as f:  
        data = json.load(f)  
        for shape in data["shapes"]:  
            
            points = shape["points"]  
              
            # Extract attributes from the shape  
            attributes = shape.get("attributes", {})  
            qwen_verified = attributes.get("qwen_verified")  
            qwen_final_class = attributes.get("qwen_final_class")  
            qwen_response = attributes.get("qwen_response")  
            label = attributes.get("yolo_class") 
            # Assuming points define a bounding box  
            x_min = min(p[0] for p in points) / image_width  
            y_min = min(p[1] for p in points) / image_height  
            x_max = max(p[0] for p in points) / image_width  
            y_max = max(p[1] for p in points) / image_height  
            width = x_max - x_min  
            height = y_max - y_min  
            bounding_box = [  
                x_min,  
                y_min,  
                width,  
                height,  
            ]  
              
            # Create detection with custom attributes  
            detection = fo.Detection(  
                label=label,  
                bounding_box=bounding_box,  
                qwen_verified=qwen_verified,  
                qwen_final_class=qwen_final_class,  
                qwen_response=qwen_response  
            )  
            detections.append(detection)  
    return detections  
  
def load_test_data(root_dir, folder_list=None):  
    samples = []  
    if folder_list is None or len(folder_list) == 0:  
        # Include the root directory itself and all its subdirectories that contain images  
        folder_list = [root_dir] + [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]  
    else:  
        folder_list = [os.path.join(root_dir, d) for d in folder_list if os.path.isdir(os.path.join(root_dir, d))]  
    for folder in folder_list:  
        for dirpath, _, filenames in os.walk(folder):  
            for filename in filenames:  
                if filename.endswith(".jpg") or filename.endswith(".png"):  
                    image_path = os.path.join(dirpath, filename)  
                    label_path = os.path.join(dirpath, filename.replace(".jpg", ".json").replace(".png", ".json"))  
  
                    if "vis." in filename:
                        continue
                    sample = fo.Sample(filepath=image_path)  
                    sample.compute_metadata()  
                    image_width = sample.metadata.width  
                    image_height = sample.metadata.height  
                    if os.path.exists(label_path):  
                        detections = load_labelme_labels(label_path, image_width, image_height)  
                        sample["ground_truth"] = fo.Detections(detections=detections)  
                          
                    samples.append(sample)  
                          
  
    return samples  
  
# 根目录和文件夹列表  
root_directory = "/mnt/disk/lyf/labelme_annotations"  
folder_list = None # 如果为空，将遍历根目录下的所有文件夹  
  
# 加载测试数据  
samples = load_test_data(root_directory, folder_list)  
  
# 创建FiftyOne集合并添加样本  
dataset = fo.Dataset(name="labelme_dataset", overwrite=True)  
dataset.add_samples(samples)  
  
# 声明动态属性以支持筛选和更好的性能  
dataset.add_sample_field("ground_truth.detections.qwen_verified", fo.StringField)  
dataset.add_sample_field("ground_truth.detections.qwen_final_class", fo.StringField)  
dataset.add_sample_field("ground_truth.detections.qwen_response", fo.StringField)  
  
# 运行预测并在FiftyOne中可视化结果  
# dataset.apply_model(model, label_field="predictions", batch_size=100)  
  
# model = foz.load_zoo_model("resnet18-imagenet-torch",device="cuda:1")  
# embeddings = dataset.compute_embeddings(model, batch_size=24)  
  
  
  
# # print(tmp_index.total_index_size)  # 200  
  
# results = fob.compute_similarity(  
#     dataset, embeddings=embeddings, brain_key="img_sim"  
# )  
# 启动FiftyOne App  
session = fo.launch_app(dataset, port=9008, address='0.0.0.0')  
# 持续运行以保持应用打开  
session.wait()