## MinAtar

NUM_FRAMES and NUM_EPOCHS are synonymous

```
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 500000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
```

#### NUM_FRAMES = 500000

Average return over 10 episodes: 5.80 ± 1.71

#### NUM_FRAMES = 1000000

```
(MinAtar) [jbejjani@holygpu8a19104 examples]$ python dqn_play.py -g breakout -m breakout_data_and_weights_1M_epochs --gif breakout_1M_epochs.gif
Recording episodes to '/n/holylabs/LABS/sham_lab/Users/jbejjani/MinAtar/examples/breakout_1M_epochs.gif'
Episode   1/10: return = 15.0
Episode   2/10: return = 1.0
Episode   3/10: return = 9.0
Episode   4/10: return = 2.0
Episode   5/10: return = 18.0
Episode   6/10: return = 23.0
Episode   7/10: return = 7.0
Episode   8/10: return = 2.0
Episode   9/10: return = 15.0
Episode  10/10: return = 19.0
GIF saved: /n/holylabs/LABS/sham_lab/Users/jbejjani/MinAtar/examples/breakout_1M_epochs.gif
------------------------------------------------------------
Average return over 10 episodes: 11.10 ± 2.39
```

#### NUM_FRAMES = 5000000

```
Cuda available?: True
Recording episodes to '/n/holylabs/sham_lab/Users/jbejjani/MinAtar/examples/breakout.gif'
Episode  1/10: return = 23.0
Episode  2/10: return = 6.0
Episode  3/10: return = 4.0
Episode  4/10: return = 0.0
Episode  5/10: return = 53.0
Episode  6/10: return = 8.0
Episode  7/10: return = 8.0
Episode  8/10: return = 28.0
Episode  9/10: return = 22.0
Episode 10/10: return = 15.0
GIF saved: /n/holylabs/sham_lab/Users/jbejjani/MinAtar/examples/breakout.gif
------------------------------------------------------------
Average return over 10 episodes: 16.70 ± 4.71
```


## MaxAtar

### old, untuned config:

```
# DQN config
BATCH_SIZE = 32
PARALLEL_ENVS = 8
REPLAY_BUFFER_SIZE = 32*10000
REPLAY_START_SIZE = 5000
DISCOUNT = 0.9
START_EPSILON = 0.1
END_EPSILON = 0.1
FIRST_N_FRAMES = 100000
TARGET_UPDATE = 0.1
TARGET_UPDATE_FREQUENCY = 1000
BATCHES_PER_STEP = 1

LEARNING_RATE = 1e-2
MOMENTUM = 0
```

#### NUM_EPOCHS = 500000

```
Training finished. Final 10 losses: [[0.011272463 ]
 [0.0026446239]
 [0.0032086056]
 [0.0027209031]
 [0.0035677776]
 [0.0030361454]
 [0.0031995294]
 [0.0020661983]
 [0.0022610826]
 [0.0051264293]]
Recording gameplay to 'breakout_500k_epochs_old_config.gif'...
GIF saved: breakout_500k_epochs_old_config.gif
Returns: [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
------------------------------------------------------------
Average return over 10 episodes: 3.00 ± 0.00
```

#### NUM_EPOCHS = 1000000

```
Training finished. Final 10 losses: [[0.0029264255]
 [0.004233336 ]
 [0.005169246 ]
 [0.004288215 ]
 [0.0037576868]
 [0.0029171924]
 [0.0024457015]
 [0.0030286903]
 [0.0031215788]
 [0.002938544 ]]
Recording gameplay to 'breakout_1M_epochs_old_config.gif'...
GIF saved: breakout_1M_epochs_old_config.gif
Returns: [4. 4. 4. 3. 3. 3. 4. 4. 4. 4.]
------------------------------------------------------------
Average return over 10 episodes: 3.70 ± 0.14
```

### new, tuned config, to match MinAtar (mostly):

```
# DQN config
BATCH_SIZE = 32
PARALLEL_ENVS = 8
REPLAY_BUFFER_SIZE = 100000  # 32*10000
REPLAY_START_SIZE = 5000
DISCOUNT = 0.99
START_EPSILON = 1.0
END_EPSILON = 0.1
FIRST_N_FRAMES = 100000
TARGET_UPDATE = 0.1
TARGET_UPDATE_FREQUENCY = 1000
BATCHES_PER_STEP = 1

LEARNING_RATE = 3e-3 # 1e-2
MOMENTUM = 0.9  # dampens oscillations, parly mimics RMSProp's running-average effect. 'poor-man's RMSProp'
```

#### NUM_EPOCHS = 500000

```
Training finished. Final 10 losses: [[0.011364803 ]
 [0.009089822 ]
 [0.008381685 ]
 [0.012707137 ]
 [0.011064261 ]
 [0.009663958 ]
 [0.014516454 ]
 [0.010895448 ]
 [0.0071388027]
 [0.008901309 ]]
Recording gameplay to 'breakout_500k_epochs_tuned_config.gif'...
GIF saved: breakout_500k_epochs_tuned_config.gif
Returns: [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
------------------------------------------------------------
Average return over 10 episodes: 4.00 ± 0.00
```

#### NUM_EPOCHS = 1000000

```
Training finished. Final 10 losses: [[0.02498027 ]
 [0.01644262 ]
 [0.017707642]
 [0.03039239 ]
 [0.013818276]
 [0.016458549]
 [0.012076841]
 [0.013666862]
 [0.011503857]
 [0.013607275]]
Recording gameplay to 'breakout_1M_epochs_tuned_config.gif'...
GIF saved: breakout_1M_epochs_tuned_config.gif
Returns: [6. 6. 6. 5. 5. 6. 6. 6. 6. 6.]
------------------------------------------------------------
Average return over 10 episodes: 5.80 ± 0.13
```

#### NUM_EPOCHS = 5000000

```
Training finished. Final 10 losses: [[0.013413146 ]
 [0.018894233 ]
 [0.009375179 ]
 [0.015958067 ]
 [0.030664934 ]
 [0.0105292415]
 [0.008088412 ]
 [0.01715374  ]
 [0.013264814 ]
 [0.015893096 ]]
Recording gameplay to 'breakout_5M_epochs_tuned_config.gif'...
GIF saved: breakout_5M_epochs_tuned_config.gif
Returns: [7. 7. 7. 6. 6. 6. 7. 7. 7. 7.]
------------------------------------------------------------
Average return over 10 episodes: 6.70 ± 0.14
```

```
Training finished. Final 10 losses: [[0.00024307246]
 [0.14837682  ]
 [0.0004209219 ]
 [0.00009902957]
 [0.00067735324]
 [0.0005600989 ]
 [0.00019184707]
 [0.00038482982]
 [0.0001399755 ]
 [0.0004497031 ]]
Recording gameplay to 'breakout_5M_epochs_seed2134_sticky_ramping_1q.gif'...
GIF saved: breakout_5M_epochs_seed2134_sticky_ramping_1q.gif
Returns: [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
------------------------------------------------------------
Average return over 10 episodes: 7.00 ± 0.00
```

```
Training finished. Final 10 losses: [[0.00078634993]
 [0.0054082163 ]
 [0.0049290466 ]
 [0.0011484146 ]
 [0.289373     ]
 [0.9070021    ]
 [0.008135218  ]
 [0.00355149   ]
 [0.0049625295 ]
 [0.330093     ]]
Recording gameplay to 'breakout_5M_epochs_seed3214_sticky_ramping_1q.gif'...
GIF saved: breakout_5M_epochs_seed3214_sticky_ramping_1q.gif
Returns: [ 6. 20. 15. 20.  3. 21.  0.  4. 11. 27.]
------------------------------------------------------------
Average return over 10 episodes: 12.70 ± 2.76
```

```
Training finished. Final 10 losses: [[0.00031531346]
 [0.0012620578 ]
 [0.0002560017 ]
 [0.00060846144]
 [0.0010000725 ]
 [0.15092944   ]
 [0.0010507116 ]
 [0.0006428918 ]
 [0.0005816926 ]
 [0.0001479842 ]]
Recording gameplay to 'breakout_5M_epochs_seed4321_sticky_ramping_1q.gif'...
GIF saved: breakout_5M_epochs_seed4321_sticky_ramping_1q.gif
Returns: [7. 8. 7. 7. 7. 7. 9. 7. 7. 7.]
------------------------------------------------------------
Average return over 10 episodes: 7.30 ± 0.20
```
