

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
        self.action_history.append(action.numpy())
        self.rewards.append(reward)
        self.child_visits.append(child_visit_dist.numpy())
        self.root_values.append(root_value.numpy())
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
            if current_index > 0 and current_index < len(self.rewards):
                last_reward = self.rewards[current_index]
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