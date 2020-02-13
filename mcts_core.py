

class Node(object):

  def __init__(self, prior: float = 0.):
    self.visit_count = 0
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.states = None
    self.reward = 0

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class MctsCore(object):

  def __init__(self, num_simulations=100, ucb_score_fn=None, discount: float = 0.99):
    self._num_simulations = num_simulations
    if ucb_score_fn is not None:
      self._ucb_score_fn = ucb_score_fn
    else:
      self._ucb_score_fn = self.ucb_score
    self._discount = discount

    self._pb_c_base = 19652
    self._pb_c_init = 1.25

    self._action_history = []

  def _ucb_score(self, parent, child):
    pc_b = math.log((parent.visit_count + self._pb_c_base + 1) /
                    self._pb_c_base) + self._pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score

  def _action(self, time_step):
    pass

