

class Trajectory(object):
    """Records a tuple of (action, reward, child_visit_dist, root_value).

    Currently implemented as a list of those four.
    """

    def __init__(self, discount):
        self.action_history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount
        self.game_states = []

    def feed(self, action, reward, child_visit_dist, root_value,
             game_state):
        # Each of these are the same size, per each state we've visited in the
        # trajectory.
        if not (isinstance(action, int) or isinstance(action, float)):
            action = action
        if not (isinstance(child_visit_dist, dict)):
            child_visit_dist = child_visit_dist
        if not (isinstance(root_value, int) or isinstance(root_value, float)):
            root_value = root_value
        self.action_history.append(action)
        self.rewards.append(reward)
        self.child_visits.append(child_visit_dist)
        self.root_values.append(root_value)
        self.game_states.append(game_state)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0.

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            # NOTE: deviation from muzero's pseudocode: we changed this code to
            # return current index's reward.
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1] # FIX case where
                """
                PREDICTIONS:

gradient_scale:  1.0
value:  tf.Tensor([[-1.0044323]], shape=(1, 1), dtype=float32)
reward:  0
policy_logits:  tf.Tensor(
[[0.4355181  0.16353947 0.         0.         0.25271562 0.
  0.         0.         0.        ]], shape=(1, 9), dtype=float32)

TARGETS:

target_value:  -1.0
target_reward:  -1.0
target_policy:  tf.Tensor([0.18 0.11 0.14 0.08 0.13 0.08 0.09 0.11 0.08], shape=(9,), dtype=float64)
value_loss_contrib:  tf.Tensor([[1.9645466e-05]], shape=(1, 1), dtype=float32)
reward_loss_contrib:  tf.Tensor(1.0, shape=(), dtype=float32)
policy_loss_contrib:  tf.Tensor([2.1744442], shape=(1,), dtype=float32)
total_loss_contrib:  tf.Tensor([[3.1744637]], shape=(1, 1), dtype=float32)
                """
            else:
                last_reward = 0.

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                #TODO(fjur): Should the top level node be used?
                targets.append((0., 0., self.child_visits[0]))
        return targets

    def make_image(self, state_index: int):
        return self.game_states[state_index]