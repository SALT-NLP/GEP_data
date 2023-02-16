import sys
import os
from os.path import abspath, dirname
from tqdm import tqdm

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from ldm.generate import Generate

g = Generate()

def generate(input_folder, output_folder, pos=0, UU=5):
    """
    For example, input_folder='/mnt/sdb1/coco/woman_text/', output_folder='/mnt/sdb1/coco/woman_img_gen/'
    
    """
    prompts = []
    for file in os.listdir(input_folder):
        id = file[:-4]
        with open(input_folder + file) as file_in:
            lines = []
            for line in file_in:
                lines.append(line)
            assert len(lines) == 1
        prompts.append([lines[0], id])

    def myFunc(e):
        return int(e[1])

    prompts.sort(key=myFunc)

    for seed, (prompt, id) in tqdm(enumerate(prompts), total=len(prompts)):
        seed += 1
        seed = UU * seed + pos
        results = g.prompt2image(prompt=prompt, outdir=output_folder, seed=seed, steps=50)
        for row in results:
            im   = row[0]
            im.save(output_folder + f'{id}_{pos}.png')


text_folder = 'v1_text'
gen_folder = 'v1_stable_gen'

pos = 0
generate(f'/mnt/sdb1/coco/man_{text_folder}/', f'/mnt/sdb1/coco/man_{gen_folder}/', pos)
generate(f'/mnt/sdb1/coco/woman_{text_folder}/', f'/mnt/sdb1/coco/woman_{gen_folder}/', pos)
pos = 1
generate(f'/mnt/sdb1/coco/man_{text_folder}/', f'/mnt/sdb1/coco/man_{gen_folder}/', pos)
generate(f'/mnt/sdb1/coco/woman_{text_folder}/', f'/mnt/sdb1/coco/woman_{gen_folder}/', pos)
pos = 2
generate(f'/mnt/sdb1/coco/man_{text_folder}/', f'/mnt/sdb1/coco/man_{gen_folder}/', pos)
generate(f'/mnt/sdb1/coco/woman_{text_folder}/', f'/mnt/sdb1/coco/woman_{gen_folder}/', pos)
pos = 3
generate(f'/mnt/sdb1/coco/man_{text_folder}/', f'/mnt/sdb1/coco/man_{gen_folder}/', pos)
generate(f'/mnt/sdb1/coco/woman_{text_folder}/', f'/mnt/sdb1/coco/woman_{gen_folder}/', pos)
pos = 4
generate(f'/mnt/sdb1/coco/man_{text_folder}/', f'/mnt/sdb1/coco/man_{gen_folder}/', pos)
generate(f'/mnt/sdb1/coco/woman_{text_folder}/', f'/mnt/sdb1/coco/woman_{gen_folder}/', pos)