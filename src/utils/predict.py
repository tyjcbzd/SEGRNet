import os
import time
from operator import add
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.GraphNet import GraphNet
from utils.helpers import create_dir
from scripts.Evaluation import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


'''
Grad-CAM
'''
import torch
import torch.nn.functional as F

def compute_grad_cam(model, input_image, target_layer):
    feature_map = None
    gradients = None

    # Hook to capture the gradients
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Hook to capture the feature map
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_image)

    # We assume output is a single scalar value for which we want to calculate Grad-CAM
    model.zero_grad()
    output.backward(retain_graph=True)

    # Compute weights for Grad-CAM
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    grad_cam_map = F.relu(torch.sum(weights * feature_map, dim=1)).detach()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return grad_cam_map

def show_grad_cam(image, grad_cam_map, title="Grad-CAM"):
    # 将 Grad-CAM 热图放缩至原图大小
    grad_cam_map = grad_cam_map.cpu().numpy()
    grad_cam_map = cv2.resize(grad_cam_map, (image.shape[3], image.shape[2]))
    
    # 将热图标准化到 [0, 1] 范围
    grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min())

    # 可视化热图
    plt.imshow(grad_cam_map, cmap='jet', alpha=0.5)  # 使用 jet 颜色映射
    plt.title(title)
    plt.colorbar()
    plt.show()


'''
保留在这里可以不用修改test.txt
'''
def load_datasets(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        file_names = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) for name in file_names]
        masks = [os.path.join(path, "masks", name) for name in file_names]
        edges = [os.path.join(path, "edges", name) for name in file_names]
        return images, masks, edges, file_names

    train_names_path = f"{path}/test.txt"
    valid_names_path = f"{path}/test.txt"

    train_x, train_y, train_edge, train_names = load_names(path, train_names_path)
    test_x, test_y, test_edge, test_names = load_names(path, valid_names_path)

    return (train_x, train_y, train_edge, train_names), (test_x, test_y, test_edge, test_names)

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

'''
测试参数配置
'''
parameters = dict(
        data_name = "CHAOS",
        dataset_path = "./data/CHAOS",
        csv_filename = "./results/csv/test_CHAOS_results.csv",
        checkpoint_path = f"checkpoints/CHAOS/best_loss_checkpoint.pth",
        save_imgs = False, # 是否保存图片
        cal_results = False, # 是否计算数值结果
        cal_pr= True,
        size=(384, 384),
        lr=1e-4,                     # Loading checkpoint
        # train_log_path = "train_log/train_kvasir_log.txt", #
        save_preds_path = "save_mask/CHAOS/preds/",
        save_gts_path = "save_mask/CHAOS/gts/",
        need_grad_cam = True 
)

if __name__ == "__main__":
    
    file = open(f'{parameters["csv_filename"]}', "w")

    file.write("Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS\n")

    """ 是否保存图片  """
    if parameters["save_imgs"]:
        save_preds_path = parameters["save_preds_path"]
        save_gts_path = parameters["save_gts_path"]

    """ 加载数据集  """
    (train_x, train_y, train_edge, train_names), (test_x, test_y, test_edge, test_names) = load_datasets(parameters["dataset_path"])
    # 针对DSB2018
    # (train_x, train_y, train_names), (test_x, test_y, test_names) = load_data_DSB2018(path)

    """ 保存文件 """
    create_dir("results")

    """ 加载权重 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model = GraphNet()
    # mbs = mbs.to(device)
    model.load_state_dict(torch.load(parameters["checkpoint_path"], map_location=device))
    model = CustomDataParallel(model).to(device)
    model.eval()

    """ 模型热身，使得计算更加准确 """
    model(torch.randn(4, 3, 384, 384))

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    total_score = []

    if parameters["cal_pr"]:
        # 初始化变量用于存储所有的GT和预测结果
        all_gt_flat = []
        all_pred_flat = []
    else:
        print("未打开绘制PR设置")

    for i, (x, y, test_name) in tqdm(enumerate(zip(test_x, test_y, test_names)), total=len(test_x)):
        ## Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, parameters["size"])
        # img_x = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        ## GT Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, parameters["size"])
        # 保存gt masks
        if parameters["save_imgs"]:
            cv2.imwrite(save_gts_path + test_name, mask)

        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        
        # 用于pr曲线
        mask_pr = mask

        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS Calculation """
            start_time = time.time()
            edge_pred, s_g = model(image)
            pred_y = s_g

            if parameters['need_grad_cam']:
                # 对中间层 调用 Grad-CAM 函数
                grad_cam_map = compute_grad_cam(model, image, model.layer4)
                # 对中间层使用 Grad-CAM
                grad_cam_x3 = compute_grad_cam(model, image, model.encoder.layer3)
                grad_cam_x5 = compute_grad_cam(model, image, model.encoder.layer5)
                
                # 对区域特征提取后的层使用 Grad-CAM
                grad_cam_x_r = compute_grad_cam(model, image, model.region)
                
                # 对最终预测使用 Grad-CAM
                grad_cam_edge_pred = compute_grad_cam(model, image, model.edge_conv)
                grad_cam_region_pred = compute_grad_cam(model, image, model.decoder)

                # 展示 Grad-CAM 热图
                show_grad_cam(image, grad_cam_x3, title="Grad-CAM x3")
                show_grad_cam(image, grad_cam_x5, title="Grad-CAM x5")
                show_grad_cam(image, grad_cam_x_r, title="Grad-CAM Region Feature")
                show_grad_cam(image, grad_cam_edge_pred, title="Grad-CAM Edge Prediction")
                show_grad_cam(image, grad_cam_region_pred, title="Grad-CAM Region Prediction")


            end_time = time.time() - start_time
            time_taken.append(end_time)

            # 预测出来是两个通道数
            # pred_y = torch.argmax(pred_y, dim=1)
            #
            # # pred_y = torch.sigmoid(pred_y)
            # pred_y = pred_y.unsqueeze(0)
            
            if parameters['cal_results']:

                score = cal_metrics(mask, pred_y)
                total_score.append(score)
                metrics_score = list(map(add, metrics_score, score))

            # 保存pred masks
            if parameters["save_imgs"]:
                pred_y = pred_y.cpu().detach().numpy().squeeze(0).squeeze(0)
                pred_y = (pred_y * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(save_preds_path + test_name, pred_y)
            else:
                print("未打开保存图片设置")


            if parameters['cal_pr']:
                # 确保mask和pred_y的形状相同，并进行展平
                all_gt_flat.append((mask_pr>0.5).flatten())
                all_pred_flat.append(pred_y.cpu().numpy().flatten())

    if parameters['cal_pr']:
        # 计算所有测试集上的Precision-Recall曲线
        all_gt_flat = np.concatenate(all_gt_flat)
        all_pred_flat = np.concatenate(all_pred_flat)
        # 保存ground truth序列
        np.save("ground_truth_CHAOS.npy", all_gt_flat)  # 保存为整数

        # 保存预测序列
        np.save("predicted_scores_CHAOS.npy", all_pred_flat)  # 保存为浮点数


        precision, recall, thresholds = precision_recall_curve(all_gt_flat, all_pred_flat)

        # 绘制Precision-Recall曲线
        plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    # print(len(metrics_score))

    if parameters['cal_results']:
        
        """ Mean Metrics Score """
        jaccard = metrics_score[0] / len(test_x)
        f1 = metrics_score[1] / len(test_x)
        recall = metrics_score[2] / len(test_x)
        precision = metrics_score[3] / len(test_x)
        specificity = metrics_score[4] / len(test_x)
        acc = metrics_score[5] / len(test_x)
        f2 = metrics_score[6] / len(test_x)

        """ Mean Time Calculation """
        mean_time_taken = np.mean(time_taken)
        print("Mean Time Taken: ", mean_time_taken)
        mean_fps = 1 / mean_time_taken

        print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} "
            f"- Specificity: {specificity:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - Mean Time: {mean_time_taken:1.7f} - Mean FPS: {mean_fps:1.7f}")

        save_str = f"{jaccard:1.4f},{f1:1.4f},{recall:1.4f},{precision:1.4f},{specificity:1.4f},{acc:1.7f},{f2:1.7f}," \
                f"{mean_time_taken:1.7f},{mean_fps:1.7f}\n"
        file.write(save_str)

        print("==========================================")
        print(len(total_score))
        print("finish!!")
        cal_list = []
        for i, item in enumerate(total_score):
            print(f'{i + 1}: score-> {item}')
            if item[0] > 0.7:
                cal_list.append(item)
        
        print("========================")
        print(len(cal_list))
        print(np.average(cal_list, axis=0))
    else:
        print("未打开计算结果设置")
    

print("++++++++++++++++++++++++++finish+++++++++++++++++++++++++")