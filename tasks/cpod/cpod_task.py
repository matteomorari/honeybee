import prettytable

from tasks.base_task import Task, TaskScore
from collections import defaultdict

class CPODScore(TaskScore):
    def get_summary(self, max_level=1):
        return {"acc": self.scores["overall"]["acc"]}

    def dumps(self):
        tb = prettytable.PrettyTable()
        tb.field_names = ["Group", "Name", "Total", "Correct", "Accuracy (%)"]
        tb.float_format = ".2"

        # ---- Overall ----
        overall = self.scores["overall"]
        tb.add_row([
            "Overall",
            "-",
            overall["total"],
            overall["correct"],
            overall["acc"],
        ], divider=True)

        # ---- Per Position ----
        for pos, data in sorted(self.scores["per_position"].items()):
            tb.add_row([
                "Position",
                pos,
                data["total"],
                data["correct"],
                data["acc"],
            ])

        tb.add_divider()

        # ---- Per Category ----
        for cat, data in sorted(self.scores["per_category"].items()):
            tb.add_row([
                "Category",
                cat,
                data["total"],
                data["correct"],
                data["acc"],
            ])

        return tb.get_string()

def calc_score(results):
    total_len = 0
    correct = 0
    incorrect = 0

    per_position = defaultdict(lambda: {"total": 0, "correct": 0})
    per_category = defaultdict(lambda: {"total": 0, "correct": 0})

    for res in results.values():
        total_len += 1
        is_correct = res["pred"].replace(".", "").lower() == "yes"

        if is_correct:
            correct += 1
        else:
            incorrect += 1

        # ---- Per position ----
        pos = res["position"]
        per_position[pos]["total"] += 1
        if is_correct:
            per_position[pos]["correct"] += 1

        # ---- Per category ----
        cat = res["question_category"]
        per_category[cat]["total"] += 1
        if is_correct:
            per_category[cat]["correct"] += 1

    # Compute accuracy for position
    for pos in per_position:
        per_position[pos]["acc"] = (
            per_position[pos]["correct"] / per_position[pos]["total"] * 100
        )

    # Compute accuracy for category
    for cat in per_category:
        per_category[cat]["acc"] = (
            per_category[cat]["correct"] / per_category[cat]["total"] * 100
        )

    scores = {
        "overall": {
            "acc": correct / total_len * 100 if total_len > 0 else 0,
            "total": total_len,
            "correct": correct,
            "incorrect": incorrect,
        },
        "per_position": dict(per_position),
        "per_category": dict(per_category),
    }

    return scores

class CPODTask(Task):
    def compute_score(self, results):
        scores = calc_score(results)
        return CPODScore(scores)

    def dump_submission_file(self, result_dir: str, results: dict):
        raise NotImplementedError()
