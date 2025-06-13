import os
import time
from operator import add
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from models.GraphNet import GraphNet
from scripts.Evaluation import *
from utils.helpers import create_dir

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
        data_name = "",
        dataset_path = "",
        csv_filename = "",
        checkpoint_path = f"",
        save_imgs = False,
        cal_results = False,
        cal_pr= True,
        size=(384, 384),
        lr=1e-4,
        save_preds_path = "",
        save_gts_path = "",
        need_grad_cam = True 
)

if __name__ == "__main__":
    
    file = open(f'{parameters["csv_filename"]}', "w")

    file.write("Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS\n")

    if parameters["save_imgs"]:
        save_preds_path = parameters["save_preds_path"]
        save_gts_path = parameters["save_gts_path"]

    (train_x, train_y, train_edge, train_names), (test_x, test_y, test_edge, test_names) = load_datasets(parameters["dataset_path"])

    create_dir("results")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model = GraphNet()
    model.load_state_dict(torch.load(parameters["checkpoint_path"], map_location=device))
    model = CustomDataParallel(model).to(device)
    model.eval()

    # warm up
    model(torch.randn(4, 3, 384, 384))

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    total_score = []

    if parameters["cal_pr"]:
        all_gt_flat = []
        all_pred_flat = []
    else:
        print("No PR")

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
        # gt masks
        if parameters["save_imgs"]:
            cv2.imwrite(save_gts_path + test_name, mask)

        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask_pr = mask

        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS Calculation """
            start_time = time.time()
            edge_pred, s_g = model(image)
            pred_y = s_g
            end_time = time.time() - start_time
            time_taken.append(end_time)
            
            if parameters['cal_results']:

                score = cal_metrics(mask, pred_y)
                total_score.append(score)
                metrics_score = list(map(add, metrics_score, score))

            if parameters["save_imgs"]:
                pred_y = pred_y.cpu().detach().numpy().squeeze(0).squeeze(0)
                pred_y = (pred_y * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(save_preds_path + test_name, pred_y)
            else:
                print("No saving figs")

            if parameters['cal_pr']:
                all_gt_flat.append((mask_pr>0.5).flatten())
                all_pred_flat.append(pred_y.cpu().numpy().flatten())

    if parameters['cal_pr']:
        all_gt_flat = np.concatenate(all_gt_flat)
        all_pred_flat = np.concatenate(all_pred_flat)
        np.save("xxx.npy", all_gt_flat)
        np.save("xxx.npy", all_pred_flat)

        precision, recall, thresholds = precision_recall_curve(all_gt_flat, all_pred_flat)

        plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

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
        print("Nothing")
    

print("++++++++++++++++++++++++++finish+++++++++++++++++++++++++")