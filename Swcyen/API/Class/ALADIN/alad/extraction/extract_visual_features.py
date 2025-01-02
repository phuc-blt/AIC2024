from __future__ import absolute_import, division, print_function
import tqdm
import h5py
import os
import torch
import numpy as np
from alad.extraction.retrieval_utils import load_oscar

def main():
    args, student_model, test_loader = load_oscar()
    list_cuda_devices()
    print(os.getenv('CUDA_VISIBLE_DEVICES'))
    extract_features(args, student_model, test_loader)


def extract_features(args, model, data_loader):
    # switch to evaluate mode
    model.eval()

    with h5py.File(args.features_h5, 'w') as f:
        feats = f.create_dataset("features", (len(data_loader.dataset), 768), dtype=np.float32)
        dt = h5py.string_dtype(encoding='utf-8')
        img_ids = f.create_dataset('image_names', (len(data_loader.dataset), ), dtype=dt)

        for i, batch_data in enumerate(tqdm.tqdm(data_loader)):
            dataset_idxs, image_names, example_imgs = batch_data
            dataset_idxs = list(dataset_idxs)

            # compute the embeddings and save in hdf5
            with torch.no_grad():
                img_cross_attention, _, _, _, img_length, _, _ = model.forward_emb(example_imgs, None)
                feats[dataset_idxs, :] = img_cross_attention.cpu().numpy()
                for image_name, idx in zip(image_names, dataset_idxs):
                    img_ids[idx] = np.array(image_name.encode("utf-8"), dtype=dt)

def list_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of available CUDA devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")
if __name__ == "__main__":
    main()
