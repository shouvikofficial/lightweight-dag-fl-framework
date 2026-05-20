import random


class TipSelector:
    """
    Select DAG tips for referencing new transactions.
    """

    def __init__(self, dag):

        self.dag = dag

    def random_tip_selection(self, count=2):
        """
        Randomly select DAG tips.
        """

        tips = self.dag.get_tips()

        if len(tips) == 0:
            return []

        if len(tips) <= count:
            return [tip.transaction.transaction_id for tip in tips]

        selected = random.sample(tips, count)

        return [
            tip.transaction.transaction_id
            for tip in selected
        ]

    def highest_accuracy_tip_selection(self, count=2):
        """
        Select tips with highest model accuracy.
        """

        tips = self.dag.get_tips()

        if len(tips) == 0:
            return []

        sorted_tips = sorted(
            tips,
            key=lambda x: x.transaction.accuracy,
            reverse=True
        )

        selected = sorted_tips[:count]

        return [
            tip.transaction.transaction_id
            for tip in selected
        ]