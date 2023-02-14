# Data release of GEP

We release the prompts/images/annotations of the [GEP project](https://salt-nlp.github.io/GEP/) in this repo.

### Prompts

We store the prompts in two folders: _woman_v1_text_ and _man_v1_text_. Each folder has 256 prompts, each prompt is stored in one file. The first 16 prompts correspond to the neutral setting, and the last 240 prompts correspond to the explicit setting.

### Images

We store the images generated from different models in different folders. We use three models: CogView2, DALLE-2, and Stable Diffusion. 

We generate five images per prompt. For example, given the prompt stored in "0.txt", we generate five images: "0_0.png", "0_1.png", "0_2.png", "0_3.png", "0_4.png". The format is either ".png" or ".jpg".

### Annotations

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

