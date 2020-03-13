

class Trajectory(object):
    """Records a tuple of (action, reward, child_visit_dist, root_value).

    Currently implemented as a list of those four.
    """

    def __init__(self):
        self.action_history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []

    def feed(self, action, reward, child_visit_dist, root_value):
        self.action_history.append(action)
        self.rewards.append(reward)
        self.child_visits.append(child_visit_dist)
        self.root_values.append(root_value)