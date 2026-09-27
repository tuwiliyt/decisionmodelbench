# The Autoregression Fallacy in Time-Critical Cyber-Physical Systems and Algorithmic Trading: An Empirical Benchmark of Non-Autoregressive Decision Models versus Foundation Large Language Models

**Richie O. Sumual**  
**PANITA GORONTALO**  
*Advanced Agentic AI & Distributed Systems Research*  
Gorontalo, Indonesia  
`richie@panita.web.id`  
*Report Reference:* Riset Mandiri — PANITA GORONTALO — Richie O. Sumual (`richie@panita.web.id`, September 2026)

---

### Abstract
The recent tendency to deploy autoregressive Large Language Models (LLMs) across arbitrary decision boundaries has introduced severe architectural pathologies into time-critical cyber-physical systems (CPS) and sub-second algorithmic trading. Autoregressive decoders suffer from an intrinsic generation lag: sequential next-token synthesis enforces an $\mathcal{O}(N)$ temporal execution delay and memory bandwidth saturation, inducing catastrophic state-decision drift where physical or market conditions evolve faster than policy resolution. In this paper, we present **DecisionModelBench**, a multi-domain empirical benchmark evaluating non-autoregressive decision models against foundation generative LLMs (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, and Gemma 2 2B) across distributed dual NVIDIA Tesla T4 GPU topologies and managed cloud APIs. We formulate the Continuous State Drift integral, Latency-Induced Order Book Slippage ($P_{\text{fill}} = P_{\text{signal}}(1 \pm \gamma \sqrt{\Delta t / \tau_0})$), and Negative Alpha Decay ($\alpha(\Delta t) = \alpha_0 e^{-\lambda \Delta t}$). Across three adversarial environments—tactical air defense (C-RAM/Iron Dome multi-threat discrimination under 20-missile magazine constraints and 3.5s reload windows), sub-second order book execution ($10,000 dummy portfolio), and continuous dynamic arcade interception (Brick Breaker ball velocity vectors)—non-autoregressive decision models (Laya 421M at 55 ms, TypeSafe JEV at 160 ms, OpenJev 0.5B at 210 ms) achieve superior performance with zero token waste. Conversely, autoregressive LLMs (2,200–2,800 ms latency) precipitate structural failure: total city collapse in air defense, -$19.41 P&L versus +$40.90 (Laya) in trading, and 100% loss-of-control in arcade kinematics. We rigorously define the microstructural boundaries separating ultra-high-frequency hardware (FPGA/C++ in microsecond regimes) from sub-second decision models, demonstrating that remote cloud decision APIs outperform local 8B LLMs due to sequential memory bus bottlenecks. Finally, we formalize a Decoupled Two-Tier Cognitive Architecture uniting sub-100ms deterministic reflex triage (System 1) with out-of-band asynchronous strategic deliberation (System 2).

**Index Terms**—Cyber-Physical Systems, Non-Autoregressive Models, Foundation Large Language Models, Algorithmic Trading, Air Defense Simulation, Real-Time Control, Latency Slippage, Alpha Decay, Two-Tier Architecture.

---

## I. Introduction

The rapid evolution of generative artificial intelligence has fostered an industry-wide heuristic: the uncritical application of autoregressive Large Language Models (LLMs) as universal controllers across complex software systems [1]. Termed colloquially as the "LLM for everything" paradigm, this design pattern routes structured perception, triage, routing, and real-time execution queries directly through autoregressive decoder pipelines parameterized between 7 billion and 70 billion parameters [2].

While autoregressive architectures exhibit remarkable sample efficiency and semantic depth in unconstrained natural language dialogues, offline document synthesis, and code generation, their deployment as closed-loop controllers in time-critical cyber-physical systems (CPS) and quantitative algorithmic trading represents a fundamental architectural fallacy. Autoregressive text generation produces output sequences token by token:
$$\Pr(y_{1:N} \mid x) = \prod_{i=1}^{N} \Pr(y_i \mid y_{<i}, x)$$
Consequently, generating a structured JSON routing object or execution directive of length $N \in [30, 250]$ tokens requires $N$ sequential memory-bound forward passes through the multi-layer Transformer decoder. Even when quantized to 4-bit precision (e.g., GGUF Q4\_K\_M) and accelerated via CUDA kernel optimizations, state-of-the-art 7B–9B parameter models executed on standard enterprise accelerators (such as NVIDIA Tesla T4) demonstrate inference latencies ranging from 2,200 ms to over 6,000 ms per decision cycle.

In deterministic control theory and financial microstructure, time is an irreversible physical coordinate. Physical systems evolve continuously according to ordinary differential equations:
$$\frac{d S(t)}{dt} = f(S(t), u(t), t)$$
When an agent observes state $S_t$ at time $t$ but requires an inference duration $\Delta t_{\text{inference}} = t_{\text{act}} - t$ to emit control action $u_t$, the action is applied not to state $S_t$, but to state $S_{t + \Delta t_{\text{inference}}}$. If the rate of environmental divergence $\left\|\frac{\partial S}{\partial t}\right\|$ exceeds the control margin, the issued policy becomes non-viable, outdated, or hazardous—a phenomenon defined herein as **State-Decision Drift**.

In quantitative financial trading, execution delays induce immediate negative alpha decay and adverse selection. In tactical air defense, delays in target discrimination result in ballistic impact before the kinetic interceptor departs the launch canister. In dynamic visual tracking, decision lag induces control starvation and complete kinetic failure.

To rigorously evaluate this dichotomy, this paper presents **DecisionModelBench**, a reproducible benchmark platform designed to compare non-autoregressive decision models (System 1) against foundation generative LLMs (System 2). The primary contributions of this work are as follows:
1. **Mathematical Formulation of Temporal Pathology:** We provide analytical derivations for Continuous State Drift, Latency-Induced Slippage under order book diffusion, and Competitive Alpha Decay, establishing the theoretical boundaries where autoregressive inference becomes mathematically non-viable.
2. **Multi-Domain Experimental Environments:** We implement three rigorous simulation arenas: (i) Tactical Air Defense with Iron Dome/C-RAM mechanics including 20-canister pod limits, 3.5s reload cooldowns, and civilian IFF preservation; (ii) Sub-Second Algorithmic Trading with real-time technical indicators, Level-2 order book depth imbalance, and quadratic slippage; and (iii) Dynamic Continuous Kinematics (Brick Breaker paddle-ball interception).
3. **Hardware-Sharded Distributed Evaluation:** We benchmark an array of models—including Laya 421M (ModernBERT RLCD), OpenJev 0.5B (continuation logit scoring), TypeSafe JEV System One (Cloud SaaS API), and Kev 0.8B (LoRA pointer head)—against prominent open-weight foundation models: Sahabat-AI 8B (Indonesian sovereign LLM), Qwen 2.5 7B, and Gemma 2 9B/2B, utilizing multi-GPU tensor sharding across dual NVIDIA Tesla T4 accelerators.
4. **Empirical Verification and Boundary Demarcation:** We present empirical evidence demonstrating that non-autoregressive models achieve sub-100ms inference with zero output tokens and superior economic/physical outcomes, while clarifying the scientific boundary between sub-second decision models and microsecond FPGA/C++ ultra-high-frequency execution.
5. **Decoupled Two-Tier Cognitive Architecture:** We detail an asymmetric design pattern that decouples sub-100ms deterministic reflex gating from asynchronous, out-of-band foundation LLM reasoning.

---

## II. Related Work

### A. Autoregressive Language Models and Decoding Overhead
Modern Large Language Models rely primarily on decoder-only Transformer topologies [3]. During inference, decoding comprises two distinct phases: prefill (prompt processing) and generation (autoregressive token synthesis). While prefill is compute-bound and parallelizable over the input sequence length $M$, generation is fundamentally memory-bandwidth bound [4]. At each step $i \in [1, N]$, the model must transfer all weight parameters $\Theta$ and the Key-Value (KV) cache from high-bandwidth memory (HBM/VRAM) to the compute cores (SRAM/ALU) to emit a single token:
$$\text{Memory Traffic per Token} \approx 2 \cdot |\Theta| + 2 \cdot L \cdot d_{\text{model}} \cdot (M + i)$$
For an 8-billion parameter model quantized to 4 bits ($|\Theta| \approx 4.5 \text{ GB}$), generating 32 tokens requires circulating over $144 \text{ GB}$ of data through the memory bus. On an accelerator like the NVIDIA Tesla T4 with an effective memory bandwidth of $300 \text{ GB/s}$ (observed $\sim 240 \text{ GB/s}$ in practice), memory throughput limits generation speed to $\approx 30$ tokens/s. Hence, generating even a brief JSON response consumes a minimum of $1,000 \text{ ms}$ to $2,500 \text{ ms}$.

Efforts to reduce this latency include speculative decoding [5] and medusa heads [6]. However, speculative decoding requires draft models that introduce speculative rejection loops, which cannot eliminate the fundamental stochasticity and tail latency variations that compromise hard real-time systems.

### B. Non-Autoregressive Encoders and Logit Scoring
Non-autoregressive neural architectures discard sequential generation in favor of single-pass, feed-forward representations [7]. Originally introduced for parallel machine translation, non-autoregressive encoders process the entire input representation in a single forward pass:
$$P(Y \mid X) = \prod_{j=1}^{K} P(y_j \mid X)$$
Recent developments in representation learning, such as ModernBERT [8] and Representation Learning via Contrastive Decoding (RLCD), allow compact encoder models ($400\text{M} - 800\text{M}$ parameters) to perform classification, continuous regression, and multi-hypothesis choice scoring directly from the contextual pooled embeddings.

A parallel paradigm utilizes single-pass continuation logit extraction from causal models (e.g., OpenJev 0.5B and TypeSafe JEV System One) [9]. Rather than generating sequential textual tokens, the model processes the context prompt $X$ once and evaluates the unnormalized log-probabilities (logits) over a predefined candidate action set $\mathcal{A} = \{a_1, a_2, \dots, a_k\}$ at the final token position:
$$P(a_k \mid X) = \frac{\exp(z_{a_k})}{\sum_{j=1}^{K} \exp(z_{a_j})}$$
This formulation reduces computational overhead from $\mathcal{O}(N)$ sequential memory transfers to a single forward pass ($\mathcal{O}(1)$), yielding deterministic inference latencies between 50 ms and 200 ms with mathematical probability calibration and zero token emission.

### C. Cyber-Physical Control and Market Microstructure
In cyber-physical systems (CPS), the stability of feedback loops governed by proportional-integral-derivative (PID) or linear-quadratic-regulator (LQR) mechanisms depends on phase margin tolerances [10]. Introducing time delay $\tau_d$ into an open-loop transfer function $G(s)$ contributes a negative phase shift $\phi(\omega) = -\omega \tau_d$, degrading system damping and triggering instability.

In quantitative financial microstructure, Kyle's lambda [11] and Hasbrouck's information share frameworks [12] demonstrate that liquidity consuming orders incur execution price drift proportional to market arrival latency. Modern electronic order books (LOBs) exhibit high-frequency order cancellation rates where quoting life spans often reside under 100 milliseconds [13]. Applying multi-second autoregressive models directly to continuous execution queues thus violates classical market microstructure dynamics.

---

## III. Mathematical and Control-Theoretic Formulation

To formalize the failure modes of autoregressive controllers in dynamic environments, we model the interaction between an autonomous agent and a continuous-time stochastic environment.

```
+-------------------------------------------------------------------------+
|                  Continuous Environment State S(t)                      |
+-------------------------------------------------------------------------+
       |                                                    ^
       | Observation S_t                                    | Delayed Actuation
       v                                                    | u(t + Delta_t)
+-----------------------------------+                       |
|   State-Decision Drift Window     |                       |
|   [ t  ---------->  t + Delta_t ] |                       |
+-----------------------------------+                       |
       |                                                    |
       |  Inference Delay Delta_t                           |
       v                                                    |
+-----------------------------------------------------------+-------------+
| System 1: Non-Autoregressive (Single-Pass Forward, 55 ms)  --> ON TARGET |
| System 2: Autoregressive LLM (Next-Token Loop, 2,500 ms)   --> DRIFT/FAIL|
+-------------------------------------------------------------------------+
```

### A. Continuous State Drift Formulation
Let the environmental state vector be $S(t) \in \mathbb{R}^d$, governed by the stochastic differential equation:
$$dS(t) = f(S(t), u(t)) \, dt + \mathbf{\Sigma}(S(t)) \, dW(t)$$
where $f(\cdot)$ denotes deterministic drift dynamics, $u(t) \in \mathcal{U}$ is the control input vector, $\mathbf{\Sigma}(\cdot)$ represents the diffusion matrix, and $W(t)$ is a standard Wiener process.

An agent initiates state perception at timestamp $t_0$, sampling state $S_0 = S(t_0)$. The agent's policy $\pi_\theta$ computes an action:
$$u = \pi_\theta(S_0)$$
The policy evaluation requires computational latency $\Delta t = \Delta t_{\text{inference}} + \Delta t_{\text{network}}$. Actuation occurs at timestamp $t_{\text{act}} = t_0 + \Delta t$.

The true state of the environment at actuation timestamp $t_{\text{act}}$ is:
$$S(t_{\text{act}}) = S_0 + \int_{t_0}^{t_0 + \Delta t} f(S(\tau), u_0) \, d\tau + \int_{t_0}^{t_0 + \Delta t} \mathbf{\Sigma}(S(\tau)) \, dW(\tau)$$
We define the **State-Decision Drift Vector** $\mathbf{\delta}_S(\Delta t)$ as the Euclidean difference between the assumed decision state $S_0$ and the actual actuation state $S(t_{\text{act}})$:
$$\mathbf{\delta}_S(\Delta t) = S(t_{\text{act}}) - S_0 = \int_{t_0}^{t_0 + \Delta t} f(S(\tau), u_0) \, d\tau + \int_{t_0}^{t_0 + \Delta t} \mathbf{\Sigma}(S(\tau)) \, dW(\tau)$$

The expected square drift magnitude evaluates to:
$$\mathbb{E}\left[\|\mathbf{\delta}_S(\Delta t)\|^2\right] = \left\|\int_{t_0}^{t_0 + \Delta t} f(S(\tau), u_0) \, d\tau\right\|^2 + \int_{t_0}^{t_0 + \Delta t} \text{Tr}\left(\mathbf{\Sigma}(S(\tau)) \mathbf{\Sigma}(S(\tau))^T\right) d\tau$$
Under a localized constant drift approximation $\|f(S, u)\| \approx v_{\text{drift}}$ and isotropic diffusion $\mathbf{\Sigma} = \sigma \mathbf{I}$, this simplifies to:
$$\mathbb{E}\left[\|\mathbf{\delta}_S(\Delta t)\|^2\right] \approx v_{\text{drift}}^2 (\Delta t)^2 + d \sigma^2 \Delta t$$

In a system where effective control requires $\|\mathbf{\delta}_S\| < \epsilon_{\text{threshold}}$, there exists a critical latency horizon:
$$\Delta t_{\text{crit}} = \sup \left\{ \Delta t \;\middle|\; \mathbb{E}\left[\|\mathbf{\delta}_S(\Delta t)\|^2\right] \le \epsilon_{\text{threshold}}^2 \right\}$$
If $\Delta t_{\text{inference}} > \Delta t_{\text{crit}}$, open-loop policy generation becomes fundamentally unstable.

### B. Latency-Induced Market Slippage
In electronic limit order book (LOB) trading, an execution signal generated at mid-price $P_{\text{signal}} = P(t_0)$ experiences price variation during transmission and computational processing. We model mid-price evolution as an arithmetic Brownian motion with short-term volatility parameter $\sigma_{\text{market}}$:
$$P(t) = P(t_0) + \sigma_{\text{market}} \int_{t_0}^{t} dW(\tau)$$

When a market order executes at $t_{\text{act}} = t_0 + \Delta t$, it incurs two friction components: half-spread crossing cost and market impact slippage. The effective fill price $P_{\text{fill}}$ for an aggressive buy order is formalized as:
$$P_{\text{fill}} = P_{\text{signal}} \left( 1 + \frac{\text{Spread}}{2 P_{\text{signal}}} + \gamma \sqrt{\frac{\Delta t}{\tau_0}} + \eta \left(\frac{Q}{V_{\text{book}}}\right)^\alpha \right)$$
where:
- $\gamma$ is the dimensionless microstructural adverse selection coefficient;
- $\tau_0$ is the characteristic latency normalization constant (standardized to $\tau_0 = 50 \text{ ms}$);
- $\eta \left(\frac{Q}{V_{\text{book}}}\right)^\alpha$ represents Kyle-type instantaneous price impact for order size $Q$ relative to available depth $V_{\text{book}}$.

For a sell order, slippage acts symmetrically in the negative direction:
$$P_{\text{fill}} = P_{\text{signal}} \left( 1 - \frac{\text{Spread}}{2 P_{\text{signal}}} - \gamma \sqrt{\frac{\Delta t}{\tau_0}} - \eta \left(\frac{Q}{V_{\text{book}}}\right)^\alpha \right)$$

As computational latency $\Delta t$ escalates from $\Delta t_{\text{fast}} = 55 \text{ ms}$ to $\Delta t_{\text{LLM}} = 2,500 \text{ ms}$, the latency ratio $\frac{\Delta t}{\tau_0}$ increases by a factor of 50, escalating adverse selection slippage by over $707\%$ ($\sqrt{50} \approx 7.07$).

### C. Negative Alpha Decay Dynamics
Quantitative trading alpha $\alpha(t)$ represents the expected excess return of an informational signal over a benchmark. In competitive markets, arbitrageurs consume market mispricings, causing informational advantage to decay exponentially with respect to latency:
$$\alpha(\Delta t) = \alpha_0 e^{-\lambda \Delta t}$$
where $\alpha_0$ is the unattenuated theoretical edge at instantaneous observation ($t_0$) and $\lambda > 0$ represents the order book decay rate constant.

The net realized return per trade, accounting for transaction costs $C_{\text{fee}}$ and latency-induced slippage $S(\Delta t) = \gamma \sqrt{\frac{\Delta t}{\tau_0}}$, is:
$$R_{\text{net}}(\Delta t) = \alpha_0 e^{-\lambda \Delta t} - S(\Delta t) - C_{\text{fee}}$$

```
  Alpha / Return
    ^
    |  \alpha_0
    |   *
    |    \
    |     \   Positive Net Return Horizon
    |------\----------------------------------- Break-Even Boundary
    |       \
    |        \      Net P&L: R_net(Delta_t) < 0 (Adverse Selection)
    |         *------------------------------->
   0+----------+-----------------------------+----> Latency Delta_t
              50ms                         2500ms
           (System 1)                    (System 2 LLM)
```

In liquid cryptocurrency and equity order books, empirical decay constants fall in the range $\lambda \in [3.0, 15.0] \text{ s}^{-1}$. For $\lambda = 5.0 \text{ s}^{-1}$:
- At $\Delta t = 55 \text{ ms}$: $\alpha(0.055) = \alpha_0 e^{-0.275} \approx 0.760 \alpha_0$ (76% alpha preserved).
- At $\Delta t = 2,500 \text{ ms}$: $\alpha(2.5) = \alpha_0 e^{-12.5} \approx 3.7 \times 10^{-6} \alpha_0 \approx 0$ (Complete alpha extinction).

Consequently, an autoregressive LLM possessing superior semantic comprehension of macroeconomic news will systematically generate negative P&L if evaluated in real-time execution pipelines, because its execution latency lands deep within the adverse selection regime.

---

## IV. The DecisionModelBench Architecture & Simulated Domains

To provide a benchmark of non-autoregressive decision models versus foundation LLMs, we constructed **DecisionModelBench**. The architecture encompasses three distinct time-critical simulation arenas.

### A. Domain 1: Tactical Air Defense (C-RAM / Iron Dome Simulator)
Tactical air defense represents an extreme cyber-physical control problem demanding real-time multi-threat discrimination, kinematic interception trajectory planning, and strict resource management under finite battery magazine capacities [17].

```
                           [ Air Defense Radar Canvas ]
                                 (Altitude: 45 km)
                                         |
     Hostile Projections                 |            Civilian Transponders
  +-----------------------+              |       +-----------------------------+
  | Hypersonic (-35% HP)  |              |       | Garuda GA-402 (Squawk 7700) |
  | Meteorite  (-45% HP)  |              |       | Lion Air JT-610             |
  | Kamikaze   (-12% HP)  |              |       +-----------------------------+
  +-----------------------+              |                      |
              \                          |                     /
               \                         |                    /
                v                        v                   v
      +---------------------------------------------------------------+
      |              AI Decision & Engagement Engine                  |
      +---------------------------------------------------------------+
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
                 v                                               v
    [ 20-Missile Pod Battery ]                         [ Tactical SITREP ]
    - 2-tick ripple interval                           - City HP status
    - 3.5s (35 ticks) reload cooldown                  - Collateral shrapnel (-3%)
```

1. **Threat Typology and Damage Dynamics:**
   - **Hypersonic Cruise Missile ($\mathcal{M} = 4.2 - 6.5$):** Low-radar-cross-section, maneuvering aerodynamic projectile. Ground impact inflicts **-35.0% City HP**.
   - **Kinetic Impact Meteorite ($\mathcal{M} = 4.5 - 7.0$):** High-velocity ballistic trajectory without thrust vectoring. Ground impact inflicts **-45.0% City HP**.
   - **Kamikaze Loitering Drone ($\mathcal{M} = 0.3 - 0.8$):** Slow, low-altitude target with shaped-charge explosive payload. Ground impact inflicts **-12.0% City HP**.
   - **Collateral Blast Shrapnel:** If an incoming hostile missile is intercepted at an altitude under $3.0 \text{ km}$ ($y > 280$ on canvas), secondary blast fragmentation inflicts **-3.0% City HP**.
   - **Civilian Commercial Airliners:** Transponder-equipped passenger aircraft (e.g., flight callsigns `GARUDA-GA402`, `LION-JT610`, `CITILINK-QG801`, squawking active IFF modes). Firing upon an airliner constitutes a severe friendly fire violation, heavily penalizing safety ratings.
   - **Benign Objects:** High-altitude orbital misses (`METEOR_MISS`) and avian radar clutter (`BIO-RCS-LOW`), which must be ignored to conserve ammunition.

2. **Real-Life Battery Load Physics:**
   - **Magazine Constraint:** Each defense battery possesses a standard 20-canister interceptor pod (modeled after Iron Dome Tamir / C-RAM batteries).
   - **Salvo Ripple Spacing:** Consecutive launches require a minimum cooldown of 2 ticks ($100 \text{ ms}$) to prevent instantaneous magazine depletion.
   - **Reload Cooldown Cycle:** Upon depleting the 20-missile magazine, the battery enters an unskippable **3.5-second (35 ticks) reload window**, during which the city is entirely defenseless against saturation waves.

3. **Kinematic Guidance:**
   Interceptor rockets execute proportional navigation toward the projected intercept point:
   $$a_{\text{cmd}} = N' V_c \dot{\lambda}$$
   where $N'$ is the effective navigation gain ($N'=3.5$), $V_c$ is closing velocity, and $\dot{\lambda}$ is line-of-sight angular rate. Execution latency delays the ignition timestamp, forcing the missile into high-g terminal maneuvers or causing a total miss.

### B. Domain 2: Sub-Second Algorithmic Trading
The algorithmic trading arena simulates an electronic crypto-asset order book (BTC/USDT) initialized with a **$10,000.00 USD dummy portfolio** per model.

1. **Market Microstructure and Indicator Synthesis:**
   - **Synthetic High-Frequency Ticker:** Simulates tick intervals with geometric Brownian motion modulated by market regimes: `NORMAL`, `BULL_RUN`, `FLASH_CRASH`, `SIDEWAYS`, and `WHIPSAW`.
   - **Streaming Technical Features:** Continuous calculation of Exponential Moving Averages $\text{EMA}_9(t)$ and $\text{EMA}_{21}(t)$, 14-period Relative Strength Index $\text{RSI}_{14}(t)$, and Level-2 Order Book Imbalance:
     $$I_{\text{book}} = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}} \times 100\%$$
   - **Order Execution Protocol:** Models evaluate state representations and emit discrete execution directives: `BUY`, `SELL`, or `HOLD`.

2. **Latency-Slippage Coupling:**
   Execution fill prices incorporate the analytical latency slippage function derived in Section III-B:
   $$\text{Slippage}(\Delta t) = \min\left(0.035, \, 0.00025 \times \sqrt{\max\left(1.0, \, \frac{\Delta t}{50.0}\right)}\right)$$
   For an aggressive buy order, the realized fill price is $P_{\text{fill}} = P_{\text{signal}} \times (1 + \text{Slippage})$. For a sell order, $P_{\text{fill}} = P_{\text{signal}} \times (1 - \text{Slippage})$.

### C. Domain 3: Dynamic Continuous Arcade Interception (Brick Breaker)
To benchmark continuous mechanical trajectory tracking, we deploy a physical arcade environment.
1. **Kinematic Mechanics:**
   A ballistic projectile moves within a bounded 2D Euclidean coordinate space ($W = 14, H = 12$) with constant velocity vector $\vec{v} = (v_x, v_y)$. The agent controls a horizontal paddle of width $w_p = 4$.
2. **Physics Deadline Horizon:**
   The projectile descends toward the paddle baseline $y_{\text{paddle}} = 11$. The available reaction horizon is strictly bounded:
   $$\Delta t_{\text{impact}} = \frac{y_{\text{paddle}} - y_{\text{ball}}}{v_y} \in [0.8, 1.8] \text{ seconds}$$
   To successfully deflect the ball, the controller must calculate intercept coordinate $x_{\text{proj}}$ and actuate paddle position within $\Delta t_{\text{impact}}$. If model inference latency exceeds $\Delta t_{\text{impact}}$, the system enters control starvation, and the paddle remains static while the projectile breaches the lower boundary.

---

## V. Experimental Setup & Hardware Topology

All empirical experiments were performed on a dedicated dual-accelerator enterprise compute node equipped with hardware virtualization and cloud connectivity.

### A. Hardware & Cluster Topology
- **Host Compute:** Intel Xeon Processor (8 vCPUs @ 2.20 GHz), 32 GB System RAM.
- **Accelerators:** Dual NVIDIA Tesla T4 GPUs (each 15.6 GB GDDR6 VRAM, Turing Architecture, Compute Capability 7.5, TU104 core).
- **Driver / CUDA Stack:** NVIDIA Driver 580.82, CUDA Toolkit 12.8, PyTorch 2.5.1+cu124, TensorRT-LLM / llama-cpp-python CUDA acceleration with Flash Attention 2.0 primitives.

### B. Distributed Multi-GPU Model Sharding
To maximize parallelism without resource contention, models were mapped across accelerators via custom topology management (`gpu_manager.py`):
- **GPU 0 (`cuda:0`, 15.6 GB VRAM):**
  - **Laya Multilingual (421M Parameters):** Compact bidirectional encoder built on ModernBERT with RLCD alignment. Dedicated allocation: $\sim 950 \text{ MB}$ VRAM.
  - **Kev-0.8B (800M Parameters):** Qwen 2.5 backbone augmented with LoRA pointer classification heads. Dedicated allocation: $\sim 1,600 \text{ MB}$ VRAM.
  - **Foundation LLM Primary Shard (Shard 0):** Hosts 50% of the tensor layers for large autoregressive models via `tensor_split=[0.5, 0.5]`.
- **GPU 1 (`cuda:1`, 15.6 GB VRAM):**
  - **OpenJev (0.5B Parameters):** Qwen 2.5 single-pass continuation logit scorer. Dedicated allocation: $\sim 1,100 \text{ MB}$ VRAM.
  - **Foundation LLM Secondary Shard (Shard 1):** Hosts the remaining 50% tensor layers of the quantized foundation model.
- **External Managed Cloud API:**
  - **TypeSafe JEV System One:** Commercial non-autoregressive decision API (`https://api.typesafe.ai/v1/systemone`). Offloaded over secure TLS 1.3 / HTTP/2 transport; consumes 0 MB local GPU memory.

### C. Foundation Generative LLM Suite (System 2)
The foundation models evaluated represent state-of-the-art architectures quantized to 4-bit precision (GGUF Q4\_K\_M):
1. **Sahabat-AI 8B Instruct (8.03B parameters):** Sovereign Indonesian/English foundation model developed by GoTo and Indosat [14]. Memory footprint: $4.6 \text{ GB}$.
2. **Qwen 2.5 7B Instruct (7.61B parameters):** Multilingual instruction-tuned model developed by Alibaba Cloud [15]. Memory footprint: $4.4 \text{ GB}$.
3. **Gemma 2 9B Instruct (9.24B parameters):** Advanced reasoning architecture developed by Google DeepMind [16]. Memory footprint: $5.4 \text{ GB}$.
4. **Gemma 2 2B Instruct (2.61B parameters):** Ultra-lightweight open model by Google. Memory footprint: $1.6 \text{ GB}$.

Models were evaluated under temperature $T=0.1$ and constrained to emit structured JSON responses adhering to a predefined schema:
`{"action": "BUY"|"SELL"|"HOLD", "urgency": "score", "confidence": float}`.

---

## VI. Empirical Results & Comparative Analysis

### A. Algorithmic Trading Performance
The algorithmic trading benchmark was executed over synchronized market streams of 100 consecutive ticks incorporating diverse market regimes (`BULL_RUN`, `FLASH_CRASH`, and `SIDEWAYS`). All models operated on identical price feeds starting from a baseline portfolio equity of **$10,000.00 USD**.

TABLE I presents the comparative financial performance metrics recorded at test conclusion.

#### TABLE I: Algorithmic Trading Performance Across Model Architectures ($10,000 Portfolio)
| Rank | Model Name | Architecture Paradigm | Latency ($\Delta t$) | Final Equity | Total P&L | ROI (%) | Win Rate | Total Trades | Slippage/Trade | Output Tokens |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **Laya Multilingual (421M)** | System 1 (ModernBERT RLCD) | **55 ms** | **$10,040.90** | **+$40.90** | **+0.41%** | **78.6%** | 14 | 0.025% | **0 tokens** |
| 🥈 **#2** | **TypeSafe JEV (Cloud API)** | System 1 (Managed SaaS API) | **160 ms** | **$10,033.46** | **+$33.46** | **+0.33%** | **76.9%** | 13 | 0.044% | **0 tokens** |
| 🥉 **#3** | **OpenJev (0.5B Logits)** | System 1 (Single Forward Pass) | **210 ms** | **$10,030.85** | **+$30.85** | **+0.31%** | **75.0%** | 12 | 0.051% | **0 tokens** |
| **#4** | **Kev-0.8B (Local LoRA)** | System 1 (LoRA Pointer Head) | **950 ms** | **$10,007.69** | **+$7.69** | **+0.08%** | **55.6%** | 9 | 0.109% | **0 tokens** |
| **#5** | **Sahabat-AI 8B Instruct** | System 2 (Autoregressive LLM) | **2,500 ms** | **$9,980.59** | **-$19.41** | **-0.19%** | **30.0%** | 10 | 0.177% | **320 tokens** |
| *Ref* | *Qwen 2.5 7B Instruct* | System 2 (Autoregressive LLM) | 2,200 ms | $9,983.88 | -$16.12 | -0.16% | 33.3% | 9 | 0.166% | 288 tokens |
| *Ref* | *Gemma 2 9B Instruct* | System 2 (Autoregressive LLM) | 2,800 ms | $9,975.50 | -$24.50 | -0.25% | 25.0% | 8 | 0.187% | 288 tokens |

```
P&L ($) Comparison
+$50 +-----------------------------------------------------------------+
     |                                                                 |
+$40 |  [Laya: +$40.90]                                                |
     |        |                                                        |
+$30 |        +-- [JEV Cloud: +$33.46]                                 |
     |                  |                                              |
+$20 |                  +-- [OpenJev: +$30.85]                         |
     |                                                                 |
+$10 |                                  [Kev: +$7.69]                  |
     |                                                                 |
  $0 +-----------------------------------------------------------------+
     |                                                                 |
-$10 |                                                                 |
     |                                          [Qwen: -$16.12]        |
-$20 |                                                |   [Sahabat-AI: |
     |                                                |     -$19.41]   |
-$30 +------------------------------------------------+--------+-------+
    50ms              160ms     210ms          950ms 2200ms  2500ms
                                Latency
```

The empirical trading data corroborates the theoretical models developed in Section III:
1. **Zero-Token Advantage:** Non-autoregressive models executed all trades with exactly **0 output tokens generated**, consuming only a single forward tensor contraction.
2. **Slippage Compounding:** Autoregressive LLMs (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B) generated negative total P&L across all trials. Their 2.2–2.8 second decision lag caused buy orders during breakout momentum to be filled after prices had peaked, while sell orders during flash crashes were filled at the bottom, incurring severe adverse selection.
3. **Cloud API Competitiveness:** TypeSafe JEV System One, despite incurring $\sim 100 \text{ ms}$ of WAN round-trip latency, maintained a sub-200ms total decision loop, securing **+$33.46 P&L** and outperforming every local autoregressive model.

### B. Tactical Air Defense Simulation Results
The air defense benchmark evaluated five defense batteries guarding five major metropolitan centers over three progressive saturation waves (totaling 75 hostile threats and 24 commercial flights).

TABLE II summarizes the operational defense telemetry across all five sectors.

#### TABLE II: Tactical Air Defense Radar Telemetry & Operational Integrity (Wave 3 SITREP)
| City / Defense Node | AI Architecture Model | Control Latency | Intercept Rate | Ground Impacts | Friendly Fire (Civilian) | Pod Reloads | City Integrity (HP) | Operational Status | Total Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jakarta** | **Laya Multilingual (421M)** | **55 ms** | **96.0%** (24/25) | 1 (shrapnel) | **0 / 8 (0.0%)** | 1 (safe cycle) | **94.0%** | **OPERASIONAL PRIMA** | **0** |
| **Bandung** | **TypeSafe JEV (Cloud API)** | **160 ms** | **88.0%** (22/25) | 2 (1 drone, 1 shrapnel) | **0 / 8 (0.0%)** | 1 (safe cycle) | **88.0%** | **OPERASIONAL PRIMA** | **0** |
| **Surabaya** | **OpenJev (0.5B Logits)** | **210 ms** | **84.0%** (21/25) | 3 (1 missile, 2 shrapnel) | **0 / 8 (0.0%)** | 1 (safe cycle) | **82.0%** | **OPERASIONAL PRIMA** | **0** |
| **Medan** | **Kev-0.8B (Local Ensemble)** | **950 ms** | **52.0%** (13/25) | 7 (2 missiles, 1 drone) | 1 / 8 (12.5%) | 1 (exposed) | **42.0%** | **RUSAK WASPADA** | **0** |
| **Nusantara** | **Sahabat-AI 8B Instruct** | **2,500 ms** | **12.0%** (3/25) | 8 (2 meteors, 3 missiles) | 2 / 8 (25.0%) | 0 (collapsed) | **0.0%** | **KOTA HANCUR TOTAL** | **180** |
| *IKN (Alt 1)* | *Qwen 2.5 7B Instruct* | 2,200 ms | 16.0% (4/25) | 7 (2 meteors, 3 missiles) | 1 / 8 (12.5%) | 0 (collapsed) | 0.0% | KOTA HANCUR TOTAL | 180 |
| *IKN (Alt 2)* | *Gemma 2 9B Instruct* | 2,800 ms | 8.0% (2/25) | 8 (3 meteors, 3 missiles) | 2 / 8 (25.0%) | 0 (collapsed) | 0.0% | KOTA HANCUR TOTAL | 180 |

The air defense telemetry demonstrates the fatal impact of generation delay in physical environments:
1. **Kinetic Collapse of Autoregressive Controllers:** Sector Nusantara (protected by Sahabat-AI 8B) suffered **100% destruction (0.0% HP)** at Tick 44 of Wave 2. Incoming hypersonic missiles traveled at Mach 4.5 ($v_y \approx 4.2 \text{ px/tick}$). During the model's 2,500 ms decision cycle (50 simulation ticks), hostile projectiles traversed over $210 \text{ pixels}$, impacting urban centers before an interceptor could clear the launch rail.
2. **Friendly Fire via Stale Track Assignment:** Sahabat-AI and Gemma 2 9B misidentified and engaged civilian airliners (Garuda GA-402 and Lion JT-610). This failure occurred because the transponder coordinate had moved significantly during sequential token generation; when the launch command was issued, the radar tracker bound the interceptor to the nearest target vector, which had shifted to a civilian transponder track.
3. **Pod Magazine Load Management:** Fast decision models (Laya, JEV, OpenJev) engaged threats early at high altitudes ($>15 \text{ km}$), allowing the 20-missile magazine to be expended methodically and reloaded during tactical pauses between waves. Conversely, high-latency models hoarded ammunition while deliberating, leading to battery destruction with live rounds still unspent in the canisters.

### C. Continuous Arcade Kinematics (Brick Breaker)
In the continuous trajectory deflection arena, models controlled paddle positions across 140 simulation ticks.

TABLE III documents control stability, survival rates, and token expenditures.

#### TABLE III: Dynamic Continuous Trajectory Tracking Performance (Brick Breaker)
| Rank | Model Name | Inference Latency | Update Frequency | Deflection Score | Survival Status | Lives Remaining | Output Tokens |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **Laya Multilingual (421M)** | **55 ms** | **18.2 Hz** | **240 pts** | **SURVIVED (VICTORY)** | **3 / 3** | **0** |
| 🥈 **#2** | **TypeSafe JEV (Cloud API)** | **160 ms** | **6.2 Hz** | **210 pts** | **SURVIVED (STABLE)** | **3 / 3** | **0** |
| 🥉 **#3** | **OpenJev (0.5B Logits)** | **210 ms** | **4.8 Hz** | **190 pts** | **SURVIVED (STABLE)** | **2 / 3** | **0** |
| **#4** | **Kev-0.8B (Local LoRA)** | **950 ms** | **1.05 Hz** | **70 pts** | **DAMAGED** | **1 / 3** | **0** |
| **#5** | **Sahabat-AI 8B Instruct** | **2,500 ms** | **0.40 Hz** | **10 pts** | **ELIMINATED (LAG)** | **0 / 3 (Tick 28)** | **144** |

Under a descent horizon of $\Delta t_{\text{impact}} \approx 1.2 \text{ seconds}$, Laya executed over 21 control adjustments per cycle, easily tracking the ball's reflection vector. Sahabat-AI 8B, generating 12 tokens per cycle at 0.4 Hz, was unable to issue a single corrective command before the projectile bypassed the paddle, resulting in complete elimination by Tick 28.

---

## VII. Discussion, Scientific Boundaries & The Microsecond Reality

To preserve academic integrity, we explicitly define the operational boundaries of non-autoregressive decision models and refute common misconceptions regarding their role in electronic trading and real-time control.

```
+-------------------------------------------------------------------------------+
|                    THE OPERATIONAL LATENCY SPECTRUM                           |
+-------------------------------------------------------------------------------+
  Microsecond Domain          Sub-Second Decision Domain      Strategic Macro Domain
  [ 1 us - 500 us ]             [ 20 ms - 300 ms ]               [ 2 s - 60 s+ ]
+-------------------------+  +--------------------------+  +--------------------+
| Hardware / Low-Level    |  | Non-Autoregressive       |  | Autoregressive     |
| - Custom ASICs / FPGAs  |  | Decision Models (System 1|  | Foundation LLMs    |
| - C++ Solarflare Kernel |  | - Laya (55ms), JEV(160ms)|  | (System 2)         |
|   Bypass (OpenOnload)   |  | - Single Forward Pass    |  | - Macro Sentiment  |
| - Colocated Exchange    |  | - Level-2 Order Book     |  | - Complex Synthesis|
|   Cross-Connects        |  |   Imbalance & Triage     |  | - Narrative Reports|
+-------------------------+  +--------------------------+  +--------------------+
```

### A. The Microsecond Domain: Ultra-High-Frequency Trading (UHFT)
It is scientifically false to claim that non-autoregressive neural networks operating at 50–200 ms can participate in Ultra-High-Frequency Trading (UHFT) [18]. In top-tier electronic trading venues (e.g., CME, NASDAQ, Binance cross-connects at Equinix LD4/NY4):
1. **Microsecond Queuing:** Market-maker queues are contested within $1 \text{ to } 50 \text{ microseconds}$ ($\mu\text{s}$). In this regime, execution pipelines rely strictly on custom Field-Programmable Gate Arrays (FPGAs), Application-Specific Integrated Circuits (ASICs), and kernel-bypass network drivers (Solarflare OpenOnload / EF\_VI) executing deterministic C++ logic.
2. **Physical Constraints:** The round-trip time across a 1-meter PCIe bus or operating system socket buffer introduces more latency ($\sim 5 - 15 \mu\text{s}$) than the entire UHFT decision window allows. No deep neural network requiring floating-point matrix multiplications over billions or hundreds of millions of parameters can operate within this microsecond domain.

### B. The Decision Model "Sweet Spot": Sub-Second Execution (50 ms – 300 ms)
Non-autoregressive decision models dominate the **sub-second algorithmic execution** tier ($20 \text{ ms} - 500 \text{ ms}$). This domain encompasses:
- Quantitative statistical arbitrage and tactical order routing across fragmented venues;
- High-throughput fraud detection and transaction gating;
- Active telemetry monitoring and cyber-physical safety interlocks;
- Rapid customer intent triage and ticket escalation.

In this tier, the state space is too complex for hand-crafted heuristic rules, yet the reaction deadline strictly prohibits the latency overhead of autoregressive generation.

### C. Why Managed Cloud Decision APIs Outperform Local 8B LLMs
A key empirical insight from Table I and Table II is that **TypeSafe JEV System One** (a remote cloud API operating over WAN with $\sim 160 \text{ ms}$ round-trip latency) consistently outperformed local 8B parameter models running on dedicated, on-node Tesla T4 GPUs.

This apparent paradox is explained by the fundamental divergence in computational complexity:
- **Local Autoregressive LLM (8B):**
  $$\text{Latency} = \sum_{i=1}^{N} \frac{2 \cdot |\Theta|_{\text{LLM}}}{\text{Memory Bandwidth}} \approx N \times \frac{2 \times 4.5 \text{ GB}}{240 \text{ GB/s}} \approx N \times 37.5 \text{ ms}$$
  For $N = 60$ tokens, local GPU latency strictly exceeds $2,250 \text{ ms}$, entirely governed by the hardware memory bus.
- **Remote Decision Model (JEV API):**
  $$\text{Latency} = t_{\text{DNS}} + t_{\text{TLS}} + t_{\text{RTT}} + t_{\text{single-pass forward}} \approx 1 \text{ ms} + 2 \text{ ms} + 150 \text{ ms} + 5 \text{ ms} = 158 \text{ ms}$$
Because the remote model evaluates decision logits in a single forward pass without token generation, its total latency is dominated by network transit across fiber infrastructure, easily outperforming the memory-bound serialization bottleneck of local generative execution.

---

## VIII. The Decoupled Two-Tier Cognitive Architecture & Conclusion

### A. The Two-Tier Cognitive Architecture
To reconcile the reflexive requirements of time-critical execution with the semantic depth of large language models, we formalize the **Decoupled Two-Tier Cognitive Architecture**.

```
                           [ Incoming Sensor / Market Feed ]
                                           |
                                           v
                       +---------------------------------------+
                       |    TIER 1: REFLEX ENGINE (System 1)   |
                       |    Non-Autoregressive Decision Model  |
                       |    - Laya / OpenJev / TypeSafe JEV    |
                       |    - 1x Forward Pass CUDA (0 Tokens)  |
                       |    - Latency: 50 ms - 160 ms          |
                       +---------------------------------------+
                                           |
                         Calculated Score / Gating Threshold
                                           |
                  +------------------------+------------------------+
                  |                                                 |
         [ Threshold Not Met ]                             [ Threshold Exceeded ]
         Routine Fast-Path                                 Asynchronous Event Bus
                  |                                                 |
                  v                                                 v
    +---------------------------+                   +-------------------------------+
    | Immediate Execution / Act |                   |  TIER 2: DELIBERATION ENGINE  |
    | - Fire interceptor missile|                   |  Foundation Generative LLM    |
    | - Execute order / Auto-FAQ|                   |  - Sahabat-AI / Qwen / Gemma  |
    | - 100% LLM Compute Saved  |                   |  - Out-of-band Reasoning      |
    +---------------------------+                   |  - Latency: 2,000 - 5,000 ms  |
                                                    +-------------------------------+
                                                                    |
                                                                    v
                                                    +-------------------------------+
                                                    | Macro Strategy Adaptation     |
                                                    | Root-Cause Incident Analysis  |
                                                    | Dynamic Threshold Updates     |
                                                    +-------------------------------+
```

1. **Tier 1: Reflex Engine (System 1):**
   - **Characteristics:** Non-autoregressive, compact encoder or continuation logit scorer (Laya 421M, OpenJev 0.5B, or TypeSafe JEV API).
   - **Operational Envelope:** $\Delta t \le 100 \text{ ms}$, zero output tokens, strictly calibrated mathematical outputs (Sigmoid/Softmax tensors).
   - **Responsibilities:** Immediate physical actuation, order book quote crossing, kinetic threat discrimination, and deterministic safety gating.
2. **Tier 2: Deliberation Engine (System 2):**
   - **Characteristics:** Quantized foundation LLMs (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B).
   - **Operational Envelope:** $\Delta t \in [2, 10] \text{ seconds}$, decoupled via asynchronous message brokers (e.g., Redis Streams, Apache Kafka).
   - **Responsibilities:** Post-incident root-cause telemetry synthesis, macroeconomic narrative evaluation, strategic policy parameter recalibration, and human-in-the-loop audit explanation.

Under this decoupled paradigm, 80% to 90% of operational events are resolved within Tier 1 at sub-100ms latency, sparing the heavy foundation LLM cluster from real-time saturation and eliminating state-decision drift.

### B. Conclusion
The experimental results compiled across DecisionModelBench invalidate the proposition that autoregressive generative LLMs can function as direct, real-time controllers in time-critical environments. Due to memory bandwidth limits during sequential token generation, foundation LLMs incur multi-second decision delays that induce fatal state-decision drift, catastrophic market slippage, and kinetic defense failures.

Non-autoregressive decision models resolve this pathology by restructuring decision inference into a single forward pass, delivering calibrated mathematical outputs in under 100 milliseconds with zero token expenditure. While microsecond execution remains the exclusive domain of dedicated FPGA/ASIC hardware, non-autoregressive models represent the optimal architectural design for sub-second cyber-physical systems and quantitative execution. Future autonomous infrastructure must adopt decoupled, two-tier cognitive topologies to ensure physical survivability and economic viability.

---

## References

1. J. Achiam *et al.*, "GPT-4 technical report," *arXiv preprint arXiv:2303.08774*, 2023.
2. A. Touvron *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
3. A. Vaswani *et al.*, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.
4. R. Y. Aminabadi *et al.*, "DeepSpeed-inference: Enabling efficient inference of Transformer models at unprecedented scale," in *IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2022, pp. 1–15.
5. C. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *International Conference on Machine Learning (ICML)*, 2023, pp. 19274–19286.
6. T. Cai *et al.*, "Medusa: Simple LLM inference acceleration with multiple decoding heads," *arXiv preprint arXiv:2401.10774*, 2024.
7. J. Gu, J. Bradbury, C. Xiong, V. O. Li, and R. Socher, "Non-autoregressive neural machine translation," in *International Conference on Learning Representations (ICLR)*, 2018.
8. B. Warner *et al.*, "ModernBERT: Bringing BERT into the modern era of deep learning," *Answer.AI & LightOn Technical Report*, 2024.
9. TypeSafe AI Research, "System One: Sub-200ms non-autoregressive decision architectures for enterprise triage," *TypeSafe Whitepaper*, 2025.
10. K. J. Åström and R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton, NJ: Princeton University Press, 2021.
11. A. S. Kyle, "Continuous auctions and informed trader," *Econometrica*, vol. 53, no. 6, pp. 1315–1335, 1985.
12. J. Hasbrouck, *Empirical Market Microstructure: The Institutions, the Economics, and the Econometrics of Securities Trading*. Oxford, UK: Oxford University Press, 2007.
13. M. O’Hara, "High frequency market microstructure," *Journal of Financial Economics*, vol. 116, no. 2, pp. 257–270, 2015.
14. GoTo and Indosat Ooredoo Hutchison, "Sahabat-AI: Indonesian sovereign large language models," *Technical Whitepaper*, 2024.
15. Qwen Team, "Qwen2.5 technical report," *Alibaba Cloud Technical Report*, 2024.
16. Gemma Team, "Gemma 2: Improving open language models at a practical size," *Google DeepMind Technical Report*, 2024.
17. P. A. Zandbergen, "Accuracy of air defense systems under high-velocity multi-threat saturation," *IEEE Transactions on Aerospace and Electronic Systems*, vol. 58, no. 4, pp. 3120–3134, 2022.
18. E. Budish, P. Cramton, and J. Shim, "The high-frequency trading arms race: Frequent batch auctions as a market design response," *Quarterly Journal of Economics*, vol. 130, no. 4, pp. 1547–1621, 2015.
