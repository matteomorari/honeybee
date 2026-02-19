import prettytable

from tasks.base_task import Task, TaskScore


class CPODScore(TaskScore):
    def get_summary(self, max_level=1):
        return {"acc": self.scores["acc"]}

    def dumps(self):
        tb = prettytable.PrettyTable()
        tb.field_names = list(self.scores.keys())
        tb.float_format = ".2"
        total_scores = []
        for _, value in self.scores.items():
            total_scores.append(value)
        tb.add_row(total_scores, divider=True)

        return tb.get_string()

def calc_score(results):
    total_len = 0
    correct = 0
    incorrect = 0

    for res in results.values():
        total_len += 1
        if res["pred"].replace(".", "").lower() == "yes":
            correct += 1
        else:
            incorrect += 1

    scores = {
        "acc": correct / total_len * 100,
        "total": total_len,
        "correct": correct,
        "incorrect": incorrect
    }

    return scores

class CPODTask(Task):
    def compute_score(self, results):
        scores = calc_score(results)
        return CPODScore(scores)

    def dump_submission_file(self, result_dir: str, results: dict):
        raise NotImplementedError()
