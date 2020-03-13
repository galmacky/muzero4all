
import math
import numpy as np

# TODO: set up auto lint
# TODO: set up auto test on github
# TODO: set up code review on github


class Node(object):
    """A node for MCTS core."""

    def __init__(self, states=None, prior=1.):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.states = states
        self.reward = 0
        self.is_final = False

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __repr__(self):
        return "{{v: {}, p: {}, v_sum: {}, s: {}, r: {}, c: {}}}".format(
                self.visit_count, self.prior, self.value_sum, self.states, self.reward, self.children)


class MctsCore(object):
    """A core engine for MCTS."""

    def __init__(self, env, model, discount=1., ucb_score_fn=None):
        self._env = env
        self.model = model
        self.discount = discount
        if ucb_score_fn is not None:
            self._ucb_score_fn = ucb_score_fn
        else:
            self._ucb_score_fn = self._ucb_score

        self._pb_c_base = 19652
        self._pb_c_init = 1.25

        self._action_history = []

    def _ucb_score(self, parent, child):
        # return child.value() + math.log(2) * math.sqrt(math.log(parent.visit_count + 1) / (child.visit_count + 1))
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) /
                        self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    def initialize(self):
        states = self.model.get_initial_states()
        self._root = Node(states)
        assert self._root.states is not None
        self.expand_node(self._root)
        assert self._root.expanded(), (
                'You should be able to take an action from the root.')

    def rollout(self):
        node, search_path, last_action = self.select_node()
        self.expand_node(node)
        parent = search_path[-2]
        assert parent.states is not None
        value = self.evaluate_node(node, parent.states, last_action)
        self.backpropagate(search_path, value)

    def select_node(self):
        node = self._root
        search_path = [node]

        last_action = None
        while not node.is_final and node.expanded():
            action, node = self.select_child(node)
            last_action = action
            search_path.append(node)
        return node, search_path, last_action

    def select_child(self, node):
        _, action, child = max(self.get_ucb_distribution(node))
        return action, child

    def get_ucb_distribution(self, node):
        return [(self._ucb_score_fn(node, child), action, child)
                for action, child in node.children.items()]

    def expand_node(self, node):
        if node.is_final or node.expanded():
            return
        for action in self._env.action_space:
            node.children[action] = Node()

    def evaluate_node(self, node, parent_state, last_action):
        states, is_final, reward, policy_dict, predicted_value = self.model.step(
                parent_state, last_action)
        for action in node.children.keys():
            node.children[action].prior = policy_dict[action]
        node.states = states
        node.reward = reward
        node.is_final = is_final
        return predicted_value

    def backpropagate(self, search_path, value):
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.discount * value

    def get_policy_distribution(self):
        """
            Returns:
                A dict of {action: the ratio of top-level visit counts}.
        """
        # TODO: remove this once we have proper test.
        sum = 0
        for _, child in self._root.children.items():
            sum += child.visit_count
        assert sum == self._root.visit_count

        # TODO: use tf instead.
        # Note that action_space is a range. This is basically 'size' of the range.
        policy = np.zeros((self._env.action_space[-1] + 1))
        for action in self._env.action_space:
            if action in self._root.children:
                policy[action] = float(self._root.children[action].visit_count) / self._root.visit_count

        return policy

    def get_value(self):
        return self._root.value()

    def get_root_for_testing(self):
        return self._root
