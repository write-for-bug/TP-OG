import os
from tqdm import tqdm
from PIL import Image
import json
import torch
from PIL import Image
from typing import List
import time

def load_id_name_dict():
    with open('./imagenet1k-subset100.json') as f:
        res = json.load(f)
    return res


'''存储数据集的路径,OpenOOD的路径结构不一样'''
DATASET_PATH_DICT = {
    "ImageNet100": "./datasets/ImageNet100",

}



















'''useless'''
@torch.inference_mode
def generate_instruct_descriptions(pipeline, messages, **kwargs):

    output = pipeline(
        messages,
        do_sample=True,
        temperature=0.7,
        use_cache=False,
        max_new_tokens=200,
        top_p=0.9,
        top_k=100,
        repetition_penalty=1.2,
    )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_texts = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
    return decoded_texts

@torch.inference_mode
def generate_descriptions(model, processor, images, prompts, **kwargs):
    inputs = processor(text=prompts, images=images, return_tensors="pt").cuda()
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        use_cache=False,
        max_new_tokens=64,
        top_p=0.9,
        top_k=100,
        repetition_penalty=1,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_texts = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
    return decoded_texts


@torch.inference_mode
def generate_images(dspipe, pos_embeds, neg_embeds, img_lsd_sample, **kwargs):
    guidance_scale = kwargs.get("guidance_scale", 4)
    num_inference_steps = kwargs.get("num_inference_steps", 50)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", True)

    images = dspipe(prompt_embeds=pos_embeds,  # positive embeds:class_id+class features
                    negative_prompt_embeds=neg_embeds,  # negative embeds:class_id
                    image=img_lsd_sample,  # sampled from image latent space
                    device='cuda',
                    guidance_scale=guidance_scale,  # 还可以调节这个参数控制特征强弱
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=True,
                    ).images
    return images


def save_images(output_dir, class_id, images):#: Image | List[Image]
    save_dir = os.path.join(output_dir, class_id)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(images, Image.Image):
        images = [images]
    for i, image in enumerate(images):
        img_path = os.path.join(save_dir, f'{int(time.time()*1000)}.jpeg')
        image.save(img_path, format='jpeg')
        print(f'Saving fake {class_id} to {img_path}')


# useless
# def img2txt(train_dir, output_dir, device, processor, model, **kwargs):
#     os.makedirs(output_dir, exist_ok=True)
#
#     id_name_dict = load_id_name_dict()
#
#     for id, class_name in tqdm(id_name_dict.items()):
#         output_file = os.path.join(output_dir, id + '-' + class_name + '.json')
#         if os.path.exists(output_file): continue
#         image_info = {}
#         id_path = os.path.join(train_dir, id)
#         if not os.path.isdir(id_path): continue
#
#         image_Path = os.listdir(id_path)
#         for img_name in tqdm(image_Path):
#             image_path = os.path.join(id_path, img_name)
#             image = Image.open(image_path)
#             prompt = f'the {id_name_dict[id]} in the image is'
#             inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
#             generated_ids = model.generate(**inputs, **kwargs)
#             generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#
#             print(generated_text)
#
#         with open(output_file, 'w', ) as f:
#             json.dump(image_info, f)
#
#
# def txt2embeddings(txt_dir, output_dir, device, tokenizer, model):
#     os.makedirs(output_dir, exist_ok=True)
#     json_files = os.listdir(txt_dir)
#     for file_name in tqdm(json_files):
#         if os.path.exists(os.path.join(output_dir, file_name.split('-')[0]) + '.pt'): continue
#         with open(os.path.join(txt_dir, file_name)) as f:
#             description = json.load(f)
#         text_embeddings = torch.empty(10, 5, 77, 768)  # 10个样本，5个特征
#         for i, (id, texts) in tqdm(enumerate(description.items())):
#             inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
#                                return_tensors='pt').to(device)
#             with torch.no_grad():
#                 output = model(**inputs)
#                 text_embeddings[i] = output.last_hidden_state  # shape like:[len(texts),77,768]
#         text_embeddings = text_embeddings.cpu()
#         torch.save(text_embeddings, os.path.join(output_dir, file_name.split('-')[0]) + '.pt')
#
