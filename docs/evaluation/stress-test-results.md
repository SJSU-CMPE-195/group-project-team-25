
## Stress Test Results

### Test Configuration
- Tool: Adversarial bot-tier evaluation on held-out test sessions
- Duration: 21 observation windows for humans, approx. 15-16 windows fo bots


### Results
| Metric                | Value                                                                       |
|-----------------------|-----------------------------------------------------------------------------|
| Best RL Accuracy      | 98.8% (legacy) \ 97.4% (revised)                                            |
| Best Tier-5 Detection | 96.8% (DG+Augmentation, revised)                                            |
| Best XGBoost Accuracy | 99.48%                                                                      |
| False Positive        | Zero for AGBoost, near-zero for most RL configs (dependant on reward setup) |

### Observations
- The XGBoost classifier system has the strongest performance and the hardest stress case is LLM-powered bots, which vary by reward design and algorithm choice.
- The major bottleneck is generalization, this is tested on a single application with a small human-test group used for collecting data.
- The key optimization would be a broader real-world data collection, this would introduce variety into the human data set and allow it to more accurately represent the public population.
