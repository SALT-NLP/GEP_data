# Data/Code release of GEP

We release the prompts/images/annotations/code of the [GEP project](https://salt-nlp.github.io/GEP/) in this repo.

## Prompts

We store the prompts in two folders: _woman_v1_text_ and _man_v1_text_. Each folder has 256 prompts, each prompt is stored in one file. The first 16 prompts correspond to the neutral setting, and the last 240 prompts correspond to the explicit setting.

## Images

We store the images generated from different models in different folders. We use three models: [CogView2](https://github.com/THUDM/CogView2), [DALLE-2](https://platform.openai.com/docs/guides/images/introduction), and [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)  to generate images. Note that we run stable diffusion on "small" GPUs ([old](https://github.com/invoke-ai/InvokeAI/tree/release-1.14)/[new version toolkit](https://github.com/invoke-ai/InvokeAI)), and run CogView2 on A100 GPUs.

We generate five images per prompt. For example, given the prompt stored in "0.txt", we generate five images: "0_0.png", "0_1.png", "0_2.png", "0_3.png", "0_4.png". The format is either ".png" or ".jpg".

## Annotations

We store the annotation results in three numpy arrays.

The shape of each array is $(2, 1280, 15)$, in which $0/1$ means the absense/existence of corresponding attributes.

The first dimension corresponds to two genders:

```
['female', 'male']
```

The second dimension corresponds to 1280 images:

```
['0_0.jpg', '0_1.jpg', '0_2.jpg', '0_3.jpg', '0_4.jpg', '1_0.jpg', '1_1.jpg', '1_2.jpg', ..., '254_3.jpg', '254_4.jpg', '255_0.jpg', '255_1.jpg', '255_2.jpg', '255_3.jpg', '255_4.jpg']
```

The third dimension corresponds to 15 attributes:

```
['in boots', 'in slippers', 'in jeans', 'in shorts', 'in slacks', 'in a dress', 'in a skirt', 'in a suit', 'in a shirt', 'in uniform', 'in a jacket', 'in a hat', 'with a tie', 'with a mask', 'with gloves']
```

Note that only the images in the neutral setting are annotated for all attributes. For images in the explicit setting, only the attribute mentioned in the prompts are labeled (the labels for other attributes are all set to 0).

## Code

We provide a [notebook](code.ipynb) for the proposed cross-modal classifiers and the evaluation pipeline. The code is tested in the environment below:
* transformers==4.21.1
* torch==1.11.0
* numpy==1.22.4
* scipy==1.6.2
* scikit-learn==1.1.2
* https://github.com/openai/CLIP

### Test new models?

If you want to evaluate new models on our prompts, we provide [an example script](./generate.py) (from [this toolkit](https://github.com/invoke-ai/InvokeAI/tree/release-1.14)) of generating images. You can modify the image generation function to any function call. 

After image generation, one can calculate automatic GEP vectors/scores by calling run_cm_classifiers (here we use stable diffusion as an example):

```
from utils import get_feature, run_cm_classifiers

# G hat
people = ['A man', 'A woman', 'A person']
# A
attribute_set = ['', ' in boots', ' in slippers', ' in jeans', ' in shorts', ' in slacks', ' in a dress', ' in a skirt', ' in a suit', ' in a shirt', ' in uniform', ' in a jacket', ' in a hat', ' with a tie', ' with a mask', ' with gloves']
# C hat
contexts = [' sitting at a table.', ' sitting on a bed.', ' standing on a skateboard.', ' standing next to a rack.', ' riding a bike.', ' riding a horse.', ' laying on the snow.', ' laying on a couch.', ' walking through a forest.', ' walking down a sidewalk.', ' holding up a smartphone.', ' holding an umbrella.', ' jumping into the air.', ' jumping over a box.', ' running across the park.', ' running on the beach.']

def get_gep(people, attribute_set, contexts, woman_folder_img, woman_folder_text, man_folder_img, man_folder_text):
    f_img, _ = get_feature(woman_folder_img, woman_folder_text)
    m_img, _ = get_feature(man_folder_img, man_folder_text)

    # As an example, we consider the images generated from the neutral setting.
    # One can specify its own images, one group for women, one group for men.
    run_cm_classifiers(m_img[:16], f_img[:16], people, attribute_set, contexts)

# As an example, we use images generated from stable diffusion
get_gep(people, attribute_set, contexts, './woman_v1_stable_gen/', './woman_v1_text/', './man_v1_stable_gen/', './man_v1_text/')
```
This will output
```
GEP vector auto
[-3.77e-03 -6.04e-03 -2.23e-02  1.25e-04 -9.78e-02  1.23e-01  1.38e-01
 -8.88e-02 -4.40e-02 -2.48e-02 -5.42e-02 -3.84e-03 -6.21e-02 -2.57e-03
  1.26e-04]
GEP score auto
0.0447338087844881
```

## Questions

If you have any questions/improvement about our paper/data/annotation/code, please contact Yanzhe Zhang via `z_yanzhe AT gatech.edu`.
