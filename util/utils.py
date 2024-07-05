import numpy as np
import logging
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


"""def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    #print(f"output shape: {output.shape}")
    #print(f"target shape: {target.shape}")
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target"""
def intersectionAndUnion(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(-1).copy()
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger



""" Funções novas exclusivas para o SUIM"""
def evaluate(model, valloader, eval_mode, cfg):
    model.eval()
    all_preds = []
    all_masks = []

    for i, batch in enumerate(valloader):
        image, target = batch[:2]  # Supondo que extras são ignorados por enquanto
        
        with torch.no_grad():
            output = model(image.cuda())
        
        pred = output.argmax(dim=1).cpu().numpy()
        target = target.numpy()

        if pred.shape != target.shape:
            #print(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
            pred = resize_or_crop(pred, target.shape)

        all_preds.append(pred)
        all_masks.append(target)
    
    mIoU, iou_class = calculate_metrics(all_preds, all_masks, cfg['nclass'])
    
    return mIoU, iou_class

def resize_or_crop(pred, target_shape):
    resized_pred = np.resize(pred, target_shape)
    return resized_pred

def calculate_metrics(preds, masks, nclass):
    total_intersection = np.zeros(nclass)
    total_union = np.zeros(nclass)
    total_target = np.zeros(nclass)
    
    for pred, mask in zip(preds, masks):
        intersection, union, target = intersectionAndUnion(pred, mask, nclass)
        total_intersection += intersection
        total_union += union
        total_target += target
    
    iou_class = total_intersection / total_union
    mIoU = np.mean(iou_class)
    
    return mIoU, iou_class

"""função para salvar máscaras"""
def save_masks_as_images(masks, output_dir, prefix='mask'):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        # Movendo a máscara de volta para a memória da CPU antes de convertê-la em um array numpy
        mask_cpu = mask.cpu()
        # Convertendo a máscara para o tipo de dados uint8 (byte) e multiplicando por 255 para manter a faixa de valores
        mask_byte = (mask_cpu * 255).clamp(0, 255).byte()
        # Convertendo a máscara para numpy array e em seguida para imagem PIL
        mask_img = Image.fromarray(mask_byte.numpy())
        mask_img.save(os.path.join(output_dir, f"{prefix}_{i}.png"))





"""Função que eu criei pra salvar as predições e a respectiva imagem original"""
def save_prediction_as_images(predictions, original_images, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    existing_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    max_index = -1
    for file_name in existing_files:
        try:
            index = int(file_name.split('_')[-1].split('.')[0])
            if index > max_index:
                max_index = index
        except ValueError:
            pass

    start_index = max_index + 1

    for i, (pred, img) in enumerate(zip(predictions, original_images)):
        pred_np = pred.detach().cpu().numpy()
        pred_img = Image.fromarray(np.uint8(pred_np[0]), mode='L')

        # Salvar a predição
        pred_img_name = os.path.join(save_path, f'prediction_{start_index + i}.png')
        pred_img.save(pred_img_name)

        # Verificar se img é um tensor do PyTorch
        if isinstance(img, torch.Tensor):
            img_tensor = img.clone().detach().cpu()  # Clonar o tensor e movê-lo para a CPU
            save_image(img_tensor, os.path.join(save_path, f'original_image_{start_index + i}.png'))
        else:
            # Converter a imagem original para tensor e salvar usando save_image
            img_tensor = TF.to_tensor(img)
            save_image(img_tensor, os.path.join(save_path, f'original_image_{start_index + i}.png'))

        # Incrementar o índice para o próximo arquivo
        start_index += 1

        """def save_prediction_as_images(predictions, original_images, save_path):
#imagem original é um tensor pytorch]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Obtenha o índice inicial com base no maior número já existente nos arquivos do diretório
    existing_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    max_index = -1
    for file_name in existing_files:
        try:
            # Extrai o número do nome do arquivo
            index = int(file_name.split('_')[-1].split('.')[0])
            if index > max_index:
                max_index = index
        except ValueError:
            pass

    start_index = max_index + 1

    for i, (pred, img) in enumerate(zip(predictions, original_images)):
        # Converte a predição em np
        pred_np = pred.detach().cpu().numpy()
        # Converter a predição para uma imagem
        pred_img = Image.fromarray(np.uint8(pred_np[0]), mode='L')

        # Converter o tensor da imagem original em uma imagem PIL
        img_pil = TF.to_pil_image(img)

        # Salvar a predição e a imagem original
        pred_img_name = os.path.join(save_path, f'prediction_{start_index + i}.png')
        img_name = os.path.join(save_path, f'original_image_{start_index + i}.png')
        pred_img.save(pred_img_name)
        img_pil.save(img_name)
"""
