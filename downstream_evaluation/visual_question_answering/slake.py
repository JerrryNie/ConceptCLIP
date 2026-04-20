import os
import json
from typing import Union

import pandas as pd
import datasets

# define path
TRAIN_META = "data/001_Slake1.0/train.json"
VAL_META = "data/001_Slake1.0/validate.json"
TEST_META = "data/001_Slake1.0/test.json"
IMAGE_DIR = "data/001_Slake1.0/imgs"


# define dataset info
_FEATURES = datasets.Features(
    {
        "image_path": datasets.Value("string"),
        "question_id": datasets.Value("int32"),
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "type": datasets.Value("int32"),
        "label": datasets.Value("int32"),
    },
)


def format_string(inputs: Union[str, int, float, None]):
    return str(inputs).strip().lower()


# answer map, size 497
answers = set()
for meta in (TRAIN_META, VAL_META, TEST_META):
    with open(meta, "r") as fp:
        metadata = json.load(fp)
        for item in metadata:
            answer = format_string(item["answer"])
            if len(answer) > 0:
                answers.add(answer)

ANSWER2ID = {key: value for value, key in enumerate(answers)}
print(f"number of answers: {len(answers)}")

class SLAKE(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=datasets.Version("3.1.0"))
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="None",
            features=_FEATURES,
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager=None):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": TRAIN_META,
                    "image_directory": IMAGE_DIR,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": VAL_META,
                    "image_directory": IMAGE_DIR,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": TEST_META,
                    "image_directory": IMAGE_DIR,
                },
            ),
        ]

    def _generate_examples(self, metadata_path: str, image_directory: str):
        metadata = pd.read_json(metadata_path)

        for _, row in metadata.iterrows():
            question_id = row["qid"]
            question = format_string(row["question"])
            answer = format_string(row["answer"])
            type = 0 if row["answer_type"] == "CLOSED" else 1

            # ignore empty answers
            if len(answer) <= 0:
                continue

            image_path = os.path.join(image_directory, row["img_name"])

            yield question_id, {
                "question": question,
                "answer": answer,
                "question_id": question_id,
                "type": type,
                "image_path": image_path,
                "label": ANSWER2ID[answer],
            }
