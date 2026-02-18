from pathlib import Path
from tasks.base_dataset import TaskDataset, Example
from pipeline.data_utils.datasets.base_task import optionize
import utils


class CPODDataset(TaskDataset):
    """CPOD dataset"""
    def __init__(self, root, processor, template_name: str, template_pattern="cpod"):
        root = Path(root)
        self.root = root
        self.image_root = self.root / "images"

        self.data = utils.load(root / "CPOD.json")

        self.processor = processor
        self.set_templatizer(template_pattern, template_name)
        utils.print_rank_0(f"CPOD total dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dic = self.data[index]
        data_id = dic["id"]
        image_path = self.image_root / dic["image"]
        image = utils.load_image(image_path)
        question = dic["question"]

        prompt = self.build_prompt(question)

        data = {
            "id": data_id,
            "question": question,
            "prompt": prompt,
            "image_path": str(image_path),
            "question_id": dic["id"],
            "question_category": dic["category"],
        }
        ex = Example(index, image, prompt, data)
        return ex
