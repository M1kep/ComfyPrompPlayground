import os
from typing import List, Tuple, Any

import numpy as np
from PIL import Image

import folder_paths
import comfy.sd
import comfy.ops
from comfy.text_encoders.sd2_clip import SD2ClipModel
from comfy.sdxl_clip import SDXLClipModel
from comfy.supported_models_base import ClipTarget
from custom_nodes.KepPromptLang.lib.clip_model import (
    PromptLangSDXLClipModel,
    PromptLangSD1ClipModel,
)

from custom_nodes.KepPromptLang.lib.tokenizer import (
    PromptLangSDXLTokenizer,
    PromptLangSD1Tokenizer,
)


class EmptyClass:
    pass


class SpecialClipLoader:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "source_clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "conditioning"

    @staticmethod
    def load_clip(source_clip: comfy.sd.CLIP) -> Tuple[comfy.sd.CLIP]:

        if isinstance(source_clip.cond_stage_model, SDXLClipModel):
            clip_target = ClipTarget(PromptLangSDXLTokenizer, PromptLangSDXLClipModel)
            clip = comfy.sd.CLIP(clip_target, embedding_directory=source_clip.tokenizer.clip_g.embedding_directory)
            comfy.sd.load_clip_weights(clip.cond_stage_model.clip_g,source_clip.cond_stage_model.clip_g.state_dict())
            comfy.sd.load_clip_weights(
                clip.cond_stage_model.clip_l, source_clip.cond_stage_model.clip_l.state_dict()
            )
        elif isinstance(source_clip, SD2ClipModel):
            raise ValueError("SD2 Clip model is not supported.")
        else:
            clip_target = ClipTarget(PromptLangSD1Tokenizer, PromptLangSD1ClipModel)
            clip = comfy.sd.CLIP(clip_target, embedding_directory=source_clip.tokenizer.clip_l.embedding_directory)
            comfy.sd.load_clip_weights(
                clip.cond_stage_model, source_clip.cond_stage_model.state_dict()
            )
        return (clip,)


def tensor2img(tensor_img) -> Image.Image:
    i = 255.0 * tensor_img.cpu().numpy()
    i_np_arr = np.clip(i, 0, 255, out=i).astype(np.uint8, copy=False)
    return Image.fromarray(i_np_arr)

class BuildGif:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_output_directory()
        pass

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "images": ("IMAGE",),
                "split_every": ("INT", {"default": -1}),
                "frame_duration": ("INT", {"default": 125}),
                "output_mode": (
                    ["One Per Split", "Big Grid"],
                    {"default": "Big Grid"},
                ),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ()
    # RETURN_NAMES = ("Gifs",)
    INPUT_IS_LIST = True
    FUNCTION = "build_gif"
    # OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True

    CATEGORY = "List Stuff"

    def build_gif(self, images: List[Any], split_every: List[int], frame_duration: List[int], output_mode: List[str]):
        print("Build GIF called!")
        print(f"{type(images)}")

        if len(split_every) > 1:
            raise Exception("List input for split every is not supported.")

        if len(output_mode) > 1:
            raise Exception("List input for output_mode is not supported.")
        output_mode = output_mode[0]

        if len(frame_duration) > 1:
            raise Exception("List input for frame_duration is not supported.")
        frame_duration = frame_duration[0]

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix="Gif", output_dir=self.output_dir, image_width=0, image_height=0)
        split_every_val = split_every[0]
        batch_size = images[0].size()[0]
        if split_every_val == -1:
            split_chunks = 1
            split_every_val = len(images)
        else:
            split_chunks = int(len(images) / split_every_val)

        num_wide = batch_size
        num_tall = split_chunks

        chunked_batches = [
            images[split_every_val * chunk_idx : split_every_val * (chunk_idx + 1)]
            for chunk_idx in range(split_chunks)
        ]

        frames = []
        results = list()

        if output_mode == "Big Grid":
            # For every image in gif
            for idx_in_chunk in range(split_every_val):
                img_shape = images[0][0].shape
                img_frame = Image.new(
                    "RGB", size=(num_wide * img_shape[0], num_tall * img_shape[1])
                )
                # For every chunk of images
                for split_idx in range(split_chunks):
                    img_chunk = chunked_batches[split_idx]
                    for batch_idx, img_tensor in enumerate(img_chunk[idx_in_chunk]):
                        img = tensor2img(img_tensor)
                        img_frame.paste(
                            img, (batch_idx * img_shape[0], split_idx * img_shape[1])
                        )
                frames.append(img_frame)

            file = f"{filename}_{counter:05}_"
            save_path = (
                f"{os.path.join(full_output_folder, file)}"
            )
            frames[0].save(
                f"{save_path}.webp",
                # quality=100,
                # method=6,
                lossless=True,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=frame_duration,
                loop=0,
            )
            results.append({
                "filename": f"{file}.webp",
                "subfolder": subfolder,
                "type": "output"
            })
        elif output_mode == "One Per Split":
            for split_idx in range(int(split_chunks)):
                split_start = split_every_val * split_idx
                split_end = split_every_val * (split_idx + 1)
                for batch_idx in range(batch_size):
                    file = f"{filename}_{counter:05}_"
                    save_path = (
                        f"{os.path.join(full_output_folder, file)}"
                    )
                    counter += 1
                    print(save_path)
                    tensor2img(images[split_start][batch_idx]).save(
                        f"{save_path}.webp",
                        save_all=True,
                        append_images=[
                            tensor2img(nested_batch[batch_idx])
                            for nested_batch in images[split_start + 1 : split_end]
                        ],
                        optimize=False,
                        duration=frame_duration,
                        loop=0,
                    )
                    results.append({
                        "filename": f"{file}.webp",
                        "subfolder": subfolder,
                        "type": "output"
                    })
        return { "ui": { "images": results } }
