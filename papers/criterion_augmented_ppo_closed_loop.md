# From Supervised Criteria to Closed-Loop Policy Utility: A Two-Arm PPO Study in an Action-Conditioned Synthetic World

**Draft status:** research manuscript draft, 26 October 2026

**Author:** Flora Lee (sole author)

**Affiliation:** Independent researcher and software developer (no institutional affiliation)

## Abstract

Supervised experiments can test whether a validation-selected criterion package changes held-out prediction utility. A further question is whether such a package helps when a policy's actions alter the next state and the resulting reward. We report a two-arm online proximal policy optimization (PPO) study in a synthetic action-conditioned environment. The arms were task-only PPO and PPO augmented with a validation-filtered criterion procedure. Forty independent base seeds were evaluated on an OOD-inverted split. Within each seed, the primary external-utility estimate averaged six positive-C and residual-strength cells before paired comparison. The mean episode-return difference was +0.238646 (95% Student-t interval [+0.187720, +0.289571]), exceeding the protocol's smallest effect of interest of 0.003. The 13 October 2026 supplement classified this primary efficacy result as passing. The original anti_hacking_gate.json remains RL_GATE_FAIL_OR_NOT_READY because seeds 96, 98, and 106 had improved rank and dynamics scores without positive mean utility differences. Context 1 also showed a negative utility difference of -0.008962 per decision. These findings support a positive average effect of the complete criterion-augmented PPO procedure in this fixed synthetic environment, while documenting local losses and a failed composite gate. They do not identify which criterion component caused the change or establish real-world policy improvement.

## 1. Introduction

The D4 measurement study compared operational indicators of representation, temporal association, and task utility. Subsequent interpretation retained those comparisons while accounting for grid boundaries, repeated threshold crossings, undefined landmarks, and utility marks already at C=0 [1]. A separate supervised intervention then tested whether validation selection among executable criteria could change held-out task-prediction utility. That study used fresh supervised training at each target condition and did not include actions that changed an environment.

This paper extends the program into an action-conditioned setting. The policy receives observations, selects one of four actions, and each action changes the transition distribution or transition outcome. We compare task-only PPO with a treatment that adds a validation-filtered criterion package to training. The primary question is whether this package changes hidden external return under the fixed synthetic environment and PPO configuration. PPO alternates environment interaction with minibatch optimization of a clipped policy objective [2]. Here, the auxiliary criteria also shape the reward used for PPO updates, so the treatment is a composite training procedure.

The work is motivated by the gap between internal predictive readouts and external behavior. A policy may improve latent-state or transition-prediction measures without increasing reward. Such proxy and outcome differences require direct reporting rather than assuming that a better readout entails better action selection [3].

## 2. Experiment card and estimand

**Question.** In the defined action-conditioned synthetic world, does validation-filtered criterion-augmented PPO increase OOD-inverted external episode return relative to task-only PPO?

**Intervention.** Select one weighting of rank, dynamics, and utility criteria using a validation anchor, then use the selected weights for reward shaping and auxiliary losses during PPO training.

**Primary outcome.** Mean raw environment return over an eight-step evaluation episode, aggregated over six positive-C/residual-strength cells within each seed.

**Independent unit.** Base seed. Paired differences are computed after the within-seed cell average. Episodes, decision steps, contexts, and cells are not treated as independent seed-level units.

**Supported inference.** The effect of this complete selected criterion-augmented PPO procedure on external return in the specified synthetic world, training schedule, and hidden split.

**Not established.** A causal mechanism through rank or dynamics prediction; a general planning or world-model capability; benefit in every seed, context, or cell; or transfer to real-world tasks or language models.

## 3. Methods

### 3.1 Action-conditioned environment

The environment has 64 latent states represented as an upper and lower component, four discrete actions, 32-dimensional observations, and eight-step episodes. Action 0 uses the base transition. Action 1 advances the lower component. Action 2 suppresses upper-component flipping. Action 3 applies both modifications. Thus action selection affects the next state.

The observation combines a noisy encoding of the latent state with a four-way spurious feature. Training and validation use a mixed set of domains. In the primary ood_inverted evaluation, the spurious label is transformed from the target action and upper-state parity. The policy receives the observation; latent codes and transition labels are supplied to the auxiliary training criteria through the environment information stream.

At episode reset, the environment generates a target-action label from the current state and a reference base transition, then holds the label fixed for the eight-step episode. The log field prev_target_action names this generated target label; it is not the previous action selected by the policy. A step yields action_reward=1 when the selected action matches that target and state_reward=1 when the next state's upper and lower parity condition is satisfied. Raw environment reward is 0.8 times action_reward plus 0.2 times state_reward. The primary utility is the sum of this raw reward over an eight-step episode, ranging from 0 to 8.

### 3.2 Treatment, selection, and training

The two formal arms were rl_task_only and rl_oracle_filtered. The latter's name refers to a predeclared validation selection rule. Four candidate coefficient triplets were used: balanced (0.40, 0.40, 0.40), rank-focused (0.60, 0.30, 0.30), dynamics-focused (0.30, 0.60, 0.30), and utility-focused (0.30, 0.30, 0.60). All share an L1 budget of 1.20.

For each base seed, candidate selection occurred at K=8, C=3, residual strength 0.5, and the original variant. Candidate PPO selectors ran for four updates. Each candidate was evaluated over two selector repeats and three validation replicas on validation_mixed. The selected candidate maximized mean validation external return minus one standard error. It was reused across the six target cells. The ood_inverted split was not used for criterion selection, checkpoint selection, or early stopping.

The actor-critic network used the D4-style residual-bypass bottleneck with hidden width 64 and 12-dimensional embedding. Each online training run used 100 PPO updates, rollout length 256, eight parallel environments, minibatches of 64, four PPO epochs per update, learning rate 3e-4, discount 0.99, GAE lambda 0.95, clipping coefficient 0.20, entropy coefficient 0.01, value coefficient 0.50, and gradient clipping at 1.0.

The selected criteria were used in two linked ways. First, a weighted rank/dynamics/utility signal was added to environment reward with shaping scale 0.20. The bounded signals were s_rank = tanh(log p(code_t | q_t)/4), s_dyn = tanh((log p(code_t+1 | q_t) + log p(code_t+2 | q_t))/8), and s_utility = 2 r_hat_t - 1. The shaped training reward was r_env + 0.20 times the weighted sum of these signals. Second, the same weights combined cross-entropy losses for current latent code and one- and two-step latent codes with a mean-squared error loss for raw environment reward; this auxiliary loss entered the PPO objective with coefficient 0.05. The task-only arm set all criterion weights to zero. The primary comparison therefore estimates the joint package, including reward shaping and auxiliary representation losses. It does not isolate either component.

Each seed used K=8, C in {0, 3, 8, 24}, residual strength in {0.5, 1.0}, and the original variant. The primary analysis used C in {3, 8, 24} crossed with both residual strengths, yielding six cells. C=0 was retained for diagnostics. Seeds 72 through 111 formed the 40-seed confirmation. Training and evaluation environment noise used common random-number streams across arms within each seed/cell; model initialization was arm-specific.

### 3.3 Outcomes and analysis

Evaluation used 256 episodes per cell on ood_inverted. For the primary contrast, each seed's task-only and filtered-arm returns were each averaged over the six primary cells. We then computed one paired difference per seed and summarized the 40 seed-level differences with a two-sided 95% Student-t interval. The protocol's smallest effect of interest was 0.003.

The context analysis grouped hidden evaluation decisions by the generated target-action label 0 through 3. The reporting amendment classified context effects, action-match rates, and action distributions as descriptive; they were not separate efficacy gates and intervals were not multiplicity-adjusted.

Before the formal treatment comparison, the environment's action-conditioned transition kernel and multistep reachability were audited, and an oracle-state baseline interface was checked for runnability. These are environment-readiness checks. They do not estimate the treatment effect.

## 4. Results

### 4.1 Overall hidden external utility

The mean paired difference in episode return, rl_oracle_filtered minus rl_task_only, was +0.238646 over 40 independent base seeds. The 95% t interval was [+0.187720, +0.289571]. The mean exceeded the SESOI of 0.003, and the lower confidence bound was above zero. The 13 October 2026 supplement classified the primary overall efficacy result as passed.

The original machine-readable anti-hacking record remains RL_GATE_FAIL_OR_NOT_READY. Its only failed check was target_no_systematic_proxy_only. This check flagged three seeds where the filtered arm's mean rank and dynamics scores were higher while the seed-level mean return difference was not positive:

| Base seed | Mean return difference | Rank-score difference | Dynamics-score difference |
|---:|---:|---:|---:|
| 96 | -0.2263 | +0.9710 | +0.5123 |
| 98 | -0.0138 | +0.9769 | +0.5319 |
| 106 | -0.0366 | +0.9814 | +0.5240 |

These seed-level exceptions coexist with the positive 40-seed mean. They show that improved state and transition prediction scores did not ensure positive return for every seed. Reward decomposition identifies local differences in action reward and state reward, but it does not establish the algorithmic cause of those policy changes.

### 4.2 Context estimates

The four context estimates were reported descriptively under the amended protocol:

| Generated target-action context | Mean utility difference per decision | 95% seed-level t interval | Action-match rate: task-only to filtered |
|---:|---:|---:|---:|
| 0 | +0.048485 | [+0.034106, +0.062864] | 53.8% to 60.1% |
| 1 | -0.008962 | [-0.013615, -0.004308] | 2.3% to 1.0% |
| 2 | +0.029550 | [+0.016224, +0.042875] | 62.6% to 65.9% |
| 3 | +0.031864 | [+0.014597, +0.049132] | 51.7% to 55.4% |

Context 1 had a negative mean difference and a reduced action-match rate. The overall positive estimate must therefore be read with this local loss. Context estimates were not confirmation gates, and their reported intervals were not adjusted for multiple comparisons.

### 4.3 Combined interpretation of the two records

The primary external-utility contrast met its stated effect criterion. The broader machine gate did not pass because it also required the absence of a seed-level proxy-only case. Both records are retained because they refer to different decision rules. The outcome-level estimate supports a positive average effect in this protocol. The complete gate remains failed, and proxy improvements do not reliably imply improvement in every seed or context.

## 5. Discussion

The PPO experiment extends the supervised criterion study by putting actions inside the data-generating loop. The policy chooses actions, the environment transitions according to those choices, and the next observation and raw reward depend on the resulting state. The overall hidden-return difference is positive under the selected procedure and fixed training schedule.

The result answers a bounded efficacy question. It does not establish that the policy acquired a general causal model of the environment. The treatment uses state and transition labels through auxiliary heads, while the hidden evaluation score is raw environment return. Selection occurs on validation return, and the final effect reflects both criterion selection and two training pathways: reward shaping and auxiliary loss. A mechanism claim would require an additional design that separately manipulates these components and measures the policy changes that connect them to return.

The context and seed results are part of the finding. Context 1 lost utility on average, and three seeds improved both monitored proxy scores without positive overall return. The context analysis is descriptive under the stated amendment. The machine gate's failure is also retained; the main efficacy classification does not convert it into an all-checks-pass statement.

External validity is limited to this small synthetic world, this hidden spurious-feature transformation, four actions, eight-step episodes, two treatment arms, and the fixed PPO configuration. No natural-language model or real-world deployment was evaluated. The state classifier's labels are available to the auxiliary criterion, and the criterion weights are selected using validation external return. Neither detail demonstrates self-generated principle reasoning.

A useful next study would independently vary reward shaping and auxiliary losses while preserving the same base-seed paired estimand and hidden split. It should retain context-level estimates and the seed-level proxy-only diagnostic. That design would clarify which pathway contributes to the average effect and where the proxy-to-utility relation breaks down.

## 6. Data, code, and provenance

The local formal bundle is under **outputs/principled_rl_confirmation_2arm_20261012_merged/**. Key records include config.json, anti_hacking_gate.json, merge_manifest.json, episode_returns.csv, context_performance_summary.csv, context_decision_steps.csv, principle_selection.csv, and the transition-kernel/reachability audits. The detailed result narrative and seed-level analysis are under **outputs/remote_followup_20261012/analysis/**. The runner is **repro/cstar_principled_rl.py** and the environment implementation is **repro/cstar_closed_loop_env.py**.

The repository's .gitignore excludes outputs/. The complete formal output bundle and the follow-up runner were not part of the current public repository snapshot when this draft was prepared. A tracked, versioned release containing compact seed-level outcomes, context summaries, gate records, protocol amendments, provenance hashes, and source code is pending. The full decision-step log is large; a compact release should retain the independent-seed results and enough data to regenerate the reported context summaries.

## 7. Author contributions and AI assistance

Flora Lee is the sole author and conducted this work as an independent researcher and software developer. She formulated the research questions, designed and directed the studies, interpreted the results, reviewed anomalous outcomes, and prepared and approved the manuscript. AI-assisted tools supported model and code implementation, training execution, comparison, aggregation, and analysis under her direction. The tools are acknowledged as tools and are not authors; Flora Lee takes responsibility for the final manuscript and its claims.

## References

1. D4 project manuscript, “When One Critical Coupling Is Not Enough: Residual Bypass Is Associated with Operational Marker Separation.” https://github.com/Floralee2024/d4-closed-loop-dynamics-paper/blob/master/paper.md
2. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” 2017. https://arxiv.org/abs/1707.06347
3. D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, “Concrete Problems in AI Safety,” 2016. https://arxiv.org/abs/1606.06565
