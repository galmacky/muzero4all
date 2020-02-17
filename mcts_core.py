
import abc
import math

# TODO: documentation
# TODO: set up auto lint
# TODO: set up auto test on github
# TODO: set up code review on github

class Node(object):

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


class MctsEnv(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, discount, action_space):
        self.discount = discount
        self.action_space = action_space

    @abc.abstractmethod
    def step(self, states, action):
        pass


# TODO: documentation
class MctsCore(object):

    def __init__(self, num_simulations, env, ucb_score_fn=None):
        if ucb_score_fn is not None:
            self._ucb_score_fn = ucb_score_fn
        else:
            self._ucb_score_fn = self._ucb_score
        self._env = env

        self._pb_c_base = 19652
        self._pb_c_init = 1.25

        self._action_history = []

    def _ucb_score(self, parent, child):
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) /
                        self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    def initialize(self, states):
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
        while node.expanded():
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
        if node.expanded():
            return
        for action in self._env.action_space:
            node.children[action] = Node()

    def evaluate_node(self, node, parent_state, last_action):
        states, is_final, reward, policy_dict, predicted_value = (
                self._env.step(parent_state, last_action))
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
            value = node.reward + self._env.discount * value

    def get_policy_distribution(self):
        # TODO: return distribution of visit counts
        pass

    def get_root_for_testing(self):
        return self._root
