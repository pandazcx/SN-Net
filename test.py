import torch.utils.data as Data
import time
import argparse
import Dataset
import yaml
import SN_Net as network
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="0", help="gpu to test") #2
    parser.add_argument("-s", "--save_path", type=str, default="./Results") #2
    parser.add_argument("-p", "--path", type=str,default="./Pretrained/SN_Net_real.pth",help="")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device_id = range(torch.cuda.device_count())
    device = torch.device('cuda')

    config_path = "./config.yml"

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    net = network.newton_modify(config["network"])
    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids=device_id)

    checkpoints = torch.load(args.path, map_location=device)
    model_state_dict = checkpoints
    net.load_state_dict(model_state_dict)

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    val_data = config["datasets"]["test"]["path"]
    dataset_val = Dataset.Dataset(0,val_data,aug_mode=2,train=False)
    loader_test = Data.DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)



    psnr_val = 0
    ssim_val = 0
    time_val = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (LQ, HQ) in enumerate(loader_test, 0):
            LQ = LQ.to(device)
            HQ = HQ.to(device)
            t1 = time.perf_counter()
            out_val = net(LQ)
            t2 = time.perf_counter()

            idx_time = t2 - t1
            idx_psnr = batch_PSNR(HQ, out_val)
            idx_ssim = batch_ssim(HQ, out_val)

            recode_str = "[idx %d]:PSNR_val: %.4f SSIM_val: %.4f Time_cost: %.4f" % ((batch_idx + 1), idx_psnr,idx_ssim,idx_time)
            print(recode_str)
            if args.save_path:
                name = str(batch_idx) + ".png"
                image_save(out_val[0], args.save_path, name)

            time_val += idx_time
            psnr_val += idx_psnr
            ssim_val += idx_ssim

    time_val /= len(dataset_val)
    psnr_val /= len(dataset_val)
    ssim_val /= len(dataset_val)
    recode_str = "ToTal : PSNR_val: %.4f SSIM_val: %.4f Time_cost: %.4f" % (psnr_val, ssim_val,time_val)
    print(recode_str)




