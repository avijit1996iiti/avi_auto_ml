from models.model_processor import auto_ab_runner

# auto_ab_runner(configs_path="config_classification.yaml")
auto_ab_runner(configs_path="config_regression.yaml")
print("Execution Done")


class Solution:
    def getLastMoment(self, n: int, left: List[int], right: List[int]) -> int: 
        return max(
            max(left) if left else 0,
            n - min(right) if right else 0
            )

        