class AtariConfig():
    # model and training config
    support_size = 10
    action_size = 9
    representation_size   = 6
    hidden_size       = 6
    #These two are defined since value/reward are technically of 2*support_size + 1
    reward_size = 1
    value_size = 1
    batch_size         = 1
   