"""Modified from the LLaVA implementation:
    https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py
"""
import json
import os

from pipeline.data_utils.datasets.base_task import optionize
from pipeline.data_utils.datasets.sqa_dataset import parse_answer, parse_question
from tasks.base_dataset import Example, TaskDataset
import utils


class SQADataset(TaskDataset):
    def __init__(
        self, root, processor, template_name: str, split="test",
    ):
        self.root = root
        self.split = split
        self.image_folder = os.path.join(root, split)
        annotation_path = os.path.join(root, "problems.json")

        with open(annotation_path, "r") as f:
            annotation_file = json.load(f)

        self.data = [ {"id": int(k), **v} for k, v in annotation_file.items() if v.get("split") == split ]

        self.processor = processor
        self.set_templatizer("eval-sqa", template_name)

        utils.print_rank_0(f"ScienceQA total {split} split dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question_id = item["id"]
        parsed_question = {
            "question": item["question"],
            "options": item["choices"],
            "context": item["hint"],
        }

        parsed_answer = {
            "lecture": item["lecture"],
            "solution": item["solution"],
            "answer_index": item["answer"],
            "answer": chr(ord("A") + item["answer"]),
        }

        if "image" in item and item["image"] is not None:
            imgpath = os.path.join(self.image_folder, str(question_id), item["image"])
            image = utils.load_image(imgpath)
            image_prompt = "Human: <image>"
        else:
            image = None
            image_prompt = None
            imgpath = None

        option, answer = optionize(parsed_question["options"], parsed_answer["answer_index"])
        parsed_answer["answer"] = answer
        parsed_data = {
            "question": parsed_question["question"],
            "context": parsed_question["context"],
            "option": option,
            "lecture": parsed_answer["lecture"],
            "solution": parsed_answer["solution"],
            "answer": answer,
        }
        prompt = self.build_prompt(parsed_data, image_prompt)

        data = {
            "prompt": prompt,
            "question": item["question"],
            "context": item["hint"],
            "options": item["choices"],
            "question_id": question_id,
            "image_path": str(imgpath),
        }

        data.update(parsed_answer)
        ex = Example(index, image, prompt, data)

        return ex
