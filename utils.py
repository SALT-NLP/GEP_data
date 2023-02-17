import clip
import torch
from PIL import Image
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

np.set_printoptions(precision=2)


device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def group_img_file(img_file_list):
    res = [[] for _ in range(5)]
    for file in img_file_list:
        if "_0" in file:
            res[0].append(file)
        elif "_1" in file:
            res[1].append(file)
        elif "_2" in file:
            res[2].append(file)
        elif "_3" in file:
            res[3].append(file)
        elif "_4" in file:
            res[4].append(file)
    return res


def get_feature(image_folder, text_folder):
    img_file_list = os.listdir(image_folder)
    text_file_list = os.listdir(text_folder)

    assert len(img_file_list) == 5 * len(text_file_list), f"This script works for the original setting of the paper, 5 images per prompt. \n {len(img_file_list)}, {len(text_file_list)}"

    def myFunc1(e):
        return int(e[:-4])
    
    def myFunc2(e):
        return int(e[:-6])

    text_file_list.sort(key=myFunc1)
    img_file_lists = group_img_file(img_file_list)
    for alist in img_file_lists:
        alist.sort(key=myFunc2)

    def get_image_feature(image_list, image_folder):
        image_features = []
        for batch_idx in range(0, len(image_list), 8):
            c_image_pos = [image_folder + image_list[idx] for idx in range(batch_idx, min(batch_idx + 8, len(image_list)))]
            c_images = [preprocess(Image.open(c_image).convert("RGB")) for c_image in c_image_pos]
            with torch.no_grad():
                image_features.append(model.encode_image(torch.tensor(np.stack(c_images)).to(device)).float())
        return torch.concat(image_features, dim=0)

    img_fes = torch.cat([get_image_feature(alist, image_folder).unsqueeze(0) for alist in img_file_lists],dim=0)

    def get_text_from_file(position):
        f = open(position, "r")
        return f.read()

    def get_text_feature(text_list, text_folder):
        text_features = []
        for batch_idx in range(0, len(text_list), 8):
            c_text_pos = [text_folder + text_list[idx] for idx in range(batch_idx, min(batch_idx + 8, len(text_list)))]
            c_texts = clip.tokenize([get_text_from_file(c_text) for c_text in c_text_pos]).to(device)
            with torch.no_grad():
                text_features.append(model.encode_text(c_texts).to(device).float())
        return torch.concat(text_features, dim=0)

    text_fes = get_text_feature(text_file_list, text_folder)

    image_features = img_fes / img_fes.norm(dim=2, keepdim=True)
    text_features = text_fes / text_fes.norm(dim=1, keepdim=True)

    return image_features.transpose(0,1), text_features



def run_cm_classifiers(m_img, f_img, people, attribute_set, contexts):
    len_con = len(contexts) * len(people)
    text_list = [person + item + context for item in attribute_set for context in contexts for person in people]

    @torch.no_grad()
    def get_text_feature(text_list):
        text_features = []
        for batch_idx in range(0, len(text_list), 8):
            c_texts = [text_list[idx] for idx in range(batch_idx, min(batch_idx + 8, len(text_list)))]
            text_input = clip.tokenize(c_texts).to(device)
            text_features.append(model.encode_text(text_input).to(device).float())
        return torch.cat(text_features, dim=0)

    text_fes = get_text_feature(text_list)
    with torch.no_grad():
        text_fes = text_fes / text_fes.norm(dim=1, keepdim=True)
    
    man_res = [[] for _ in range(15)]
    woman_res = [[] for _ in range(15)]

    # ensemble time
    times = 10

    for gender, img_fea in (('man', m_img), ('woman', f_img)):
        res = [[] for _ in range(15)]

        for trial in range(times):
            for i in range(1, 16):
                zero_features = text_fes[:len_con]
                one_features = text_fes[len_con * i: len_con * i + len_con]
                features = torch.cat([zero_features, one_features], dim=0).cpu().numpy()
                labels = np.array([0] * len_con + [1] * len_con)

                test_feature = img_fea.reshape(80, -1).cpu().numpy()

                X = features
                Y = labels
                clf = make_pipeline(SGDClassifier(max_iter=5000, loss='log_loss', tol=1e-3, random_state=trial, validation_fraction=0.1, early_stopping=True))
                clf.fit(X, Y)

                pred = clf.predict_proba(test_feature)[:, 1]
                res[i - 1].append(np.mean(pred[:80], axis=0))

        res = [np.mean(np.array(item), axis=0) for item in res]

        if gender == 'man':
            man_res = res
        elif gender == 'woman':
            woman_res = res

    y = np.array([woman_res[i] - man_res[i] for i in range(15)])
    print("GEP vector auto")
    print(y)
    print("GEP score auto")
    print(np.mean(np.abs(y)))