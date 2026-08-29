# VISTA-Net++: A Verified Belief-Centric Framework for Safe Network Management

## 1. Problem Formulation

The central difficulty of network management is not language parsing itself, but safe sequential decision-making under partial observability, temporal drift, heterogeneous evidence, and hard operational constraints. A practical system must determine what the network state most likely is, what evidence remains missing, which action is worth taking next, and whether the action can be justified before execution.

Given a user intent \(q\), a sequence of observations \(O_{1:t}\), an action space \(A\), a constraint set \(C\), a case memory \(M\), and a simulator \(S\), the objective is to produce a safe decision trajectory \(\pi\), a verifiable proof trace \(P\), and a final outcome \(y\):

$$
\pi = [a_1, a_2, \ldots, a_T], \qquad
P = [p_1, p_2, \ldots, p_T].
$$

The optimization target is

$$
\max_{\pi}\ \mathbb{E}[\mathrm{TaskSuccess}(\pi)]
- \lambda_{\mathrm{risk}} \,\mathrm{Risk}(\pi)
- \lambda_{\mathrm{cost}} \,\mathrm{Cost}(\pi)
- \lambda_{\mathrm{human}} \,\mathrm{HumanTurns}(\pi),
$$

subject to

$$
\Pr\!\big[\mathrm{InvariantViolation}(\pi)\big] \le \epsilon .
$$

This formulation forces the system to optimize task completion jointly with safety, interaction cost, and operational efficiency, rather than reducing the problem to one-shot command generation.

## 2. Core Perspective

The proposed framework is built on a simple premise: safe network management requires a persistent belief state rather than a transient text context. Once the problem is viewed through that lens, four requirements become unavoidable. First, the state representation must fuse topology, configuration, telemetry, tickets, and change history into a unified structure. Second, uncertainty must be explicit, because action quality depends on unresolved hypotheses rather than only on observed facts. Third, action selection must weigh information gain against operational risk. Fourth, every executable decision must carry a structured proof object that makes its assumptions, invariants, and rollback path inspectable.

Under this perspective, diagnosis, planning, synthesis, and assurance are no longer disjoint tasks. They become different manifestations of the same computational loop: belief update, hypothesis refinement, candidate action construction, counterfactual validation, and proof-carrying output.

## 3. Framework Overview

VISTA-Net++ consists of six tightly coupled stages: an intent compiler, a temporal typed belief graph builder, a belief encoder, a hypothesis generator, an information-gain planner, and a proof-carrying verifier backed by counterfactual simulation. The entire pipeline is centered on a single evolving state object \(B_t\).

The intent compiler transforms natural-language requests into a typed optimization target:

$$
G = \mathrm{Compile}(q,\mathrm{policy},\mathrm{role},\mathrm{environment}).
$$

The resulting goal object is

$$
G =
\left\{
\begin{array}{l}
\mathrm{task\_type},\\
\mathrm{objective},\\
\mathrm{invariants},\\
\mathrm{risk\_budget},\\
\mathrm{allowed\_actions},\\
\mathrm{protected\_scope},\\
\mathrm{stop\_condition}
\end{array}
\right\}.
$$

This representation decouples intent understanding from low-level execution and ensures that downstream modules reason over typed constraints rather than raw prompts.

## 4. Temporal Typed Belief Graph

The state at time \(t\) is represented as

$$
B_t = (V_t, E_t, X_t, U_t),
$$

where \(V_t\) is the node set, \(E_t\) is the edge set, \(X_t\) stores typed attributes, and \(U_t\) stores uncertainty metadata. Nodes include devices, interfaces, links, policies, routes, telemetry windows, incidents, change events, tickets, and evidence objects. Edges encode structural and causal relations such as connectivity, policy application, temporal precedence, evidence support, and evidence conflict.

Each node carries type, value, timestamp, source, confidence, and access scope. The initial representation of a node \(v\) is

$$
h_v^{(0)} =
\mathrm{Emb}_{\mathrm{type}}(\mathrm{type}_v)
+ \mathrm{Emb}_{\mathrm{attr}}(\mathrm{attr}_v)
+ \mathrm{Emb}_{\mathrm{time}}(\Delta t_v)
+ \mathrm{Emb}_{\mathrm{src}}(\mathrm{src}_v)
+ \mathrm{Emb}_{\mathrm{conf}}(\mathrm{conf}_v).
$$

This representation makes uncertainty first-class rather than implicit. The system therefore reasons not only over what is known, but also over when it was observed, where it came from, and how strongly it should be trusted.

## 5. Belief Encoder

Network states evolve on multiple temporal horizons. Short spikes, medium-range degradations, and long-range change effects should not be conflated into a single flat vector. The belief encoder therefore constructs horizon-specific summaries:

$$
s_v =
\mathrm{Concat}\!\Big(
\mathrm{TEnc}_{5m}(x_v),\,
\mathrm{TEnc}_{1h}(x_v),\,
\mathrm{TEnc}_{24h}(x_v)
\Big).
$$

Graph aggregation then propagates evidence through typed structural dependencies:

$$
m_v^{(l)} = \sum_{u \in \mathcal{N}(v)}
\mathrm{Attn}\!\big(\mathrm{type}(u,v), h_u^{(l)}, h_v^{(l)}\big),
$$

$$
h_v^{(l+1)} = \mathrm{FFN}\!\big(h_v^{(l)} + m_v^{(l)}\big),
\qquad
z_t = \mathrm{Pool}\!\big(\{h_v^{(L)}\}\big).
$$

The resulting latent state \(z_t\) summarizes both structural context and temporal evolution, providing the planner with a state representation that is explicitly aligned with operational reasoning.

## 6. Case Memory Retrieval

Historical cases are encoded as procedural records instead of plain text passages:

$$
r =
\left\{
\begin{array}{l}
\mathrm{goal\_ir},\\
\mathrm{topology\_pattern},\\
\mathrm{evidence\_pattern},\\
\mathrm{hypothesis\_path},\\
\mathrm{action\_template},\\
\mathrm{verifier\_outcome},\\
\mathrm{rollback\_plan}
\end{array}
\right\}.
$$

Retrieval is performed by matching the current state and objective to prior solved procedures:

$$
\mathrm{score}(r,G,B_t) =
w_{\mathrm{sem}}\,\mathrm{sim}_{\mathrm{goal}}(G,r.\mathrm{goal\_ir})
+ w_{\mathrm{topo}}\,\mathrm{sim}_{\mathrm{topology}}(B_t,r.\mathrm{topology\_pattern})
+ w_{\mathrm{evi}}\,\mathrm{sim}_{\mathrm{evidence}}(B_t,r.\mathrm{evidence\_pattern})
+ w_{\mathrm{proc}}\,\mathrm{sim}_{\mathrm{procedure}}(\hat{\tau}_t,r.\mathrm{hypothesis\_path})
+ w_{\mathrm{inv}}\,\mathrm{overlap}(G.\mathrm{invariants},r.\mathrm{verifier\_outcome}.\mathrm{invariants}).
$$

This mechanism provides templates, failure patterns, and verifier-compatible action structures without collapsing the decision process into case imitation.

## 7. Hypothesis Space and Posterior Inference

Instead of committing early to a single interpretation, the framework maintains a hypothesis set

$$
H_t = \{h_1, h_2, \ldots, h_K\}.
$$

Each hypothesis may correspond to a root cause, a policy-state inconsistency, a bottleneck location, a risky execution branch, or an unverified change effect. Posterior inference is defined by

$$
p(h_k \mid B_t,G,R_t)
\propto
p(O_t \mid h_k,B_{t-1})\,
p(h_k \mid B_{t-1},G,R_t).
$$

In practice, the posterior is parameterized as

$$
p_t(h_k) = \mathrm{softmax}\!\big(\phi(z_t,g_t,r_t,h_k)\big).
$$

This step is essential because action quality is governed by the remaining ambiguity in the state space. Once the posterior is explicit, action selection can optimize for uncertainty reduction rather than only local heuristics.

## 8. Information-Gain Planning

The candidate action set is

$$
A_t =
\{\mathrm{query\_metric},\mathrm{inspect\_config},\mathrm{run\_probe},
\mathrm{retrieve\_case},\mathrm{ask\_human},\mathrm{simulate\_change},
\mathrm{synthesize\_patch},\mathrm{execute\_patch},\mathrm{rollback}\}.
$$

Constraint-aware pruning yields

$$
A_t^{\mathrm{safe}} =
\{\, a \in A_t \mid \mathrm{access}(a)\ \mathrm{allowed}\ \land\ \widehat{\mathrm{risk}}(a) \le \tau_t \,\}.
$$

The planner evaluates an action by balancing task utility, information gain, risk, execution cost, and human burden:

$$
U(a) =
w_{\mathrm{task}}\,\Delta \mathrm{TaskValue}(a)
+ w_{\mathrm{info}}\,\mathrm{IG}(a;H_t \mid B_t)
- w_{\mathrm{risk}}\,\mathrm{Risk}(a \mid B_t)
- w_{\mathrm{cost}}\,\mathrm{Cost}(a)
- w_{\mathrm{human}}\,\mathrm{HumanBurden}(a).
$$

The information gain term is

$$
\mathrm{IG}(a;H_t \mid B_t) =
\mathcal{H}\!\big(p(H_t \mid B_t)\big)
- \mathbb{E}_{o}\!\left[
\mathcal{H}\!\big(p(H_t \mid B_t,o,a)\big)
\right].
$$

The optimal action is then

$$
a_t = \arg\max_{a \in A_t^{\mathrm{safe}}} U(a).
$$

This design allows the system to prefer evidence acquisition when uncertainty dominates and to prefer execution only when the residual uncertainty is sufficiently controlled.

## 9. Typed Action Construction

Executable decisions are not emitted directly as raw commands. Instead, the model first constructs a typed action object:

$$
\mathrm{ActionIR} =
\left\{
\begin{array}{l}
\mathrm{op},\\
\mathrm{target},\\
\mathrm{symbolic\_params},\\
\mathrm{preconditions},\\
\mathrm{expected\_effect},\\
\mathrm{invariants\_checked},\\
\mathrm{approval\_level},\\
\mathrm{rollback\_plan}
\end{array}
\right\}.
$$

The action template is generated as

$$
T_t = \mathrm{Decoder}(G,B_t,H_t,R_t).
$$

This factorization is important because it isolates symbolic structure from final parameter filling, making the resulting action compatible with verification and rollback analysis.

## 10. Counterfactual Evaluation and Proof-Carrying Verification

Single-world dry runs are often insufficient because operational risk depends on unresolved hypotheses, workload drift, and latent dependencies. A set of worlds is therefore instantiated as

$$
W_t = \{(h_i,\mathrm{load}_j,\mathrm{env}_k)\}.
$$

The simulator evaluates each candidate under these worlds:

$$
\mathrm{SimScore}(a) = \mathrm{Aggregate}_{w \in W_t}\, S(a,w).
$$

Risk is estimated by hypothesis-weighted violation severity:

$$
\mathrm{Risk}(a \mid B_t) =
\sum_{h} p(h \mid B_t)\,\mathrm{ViolationSeverity}(a,h),
$$

$$
\mathrm{ViolationSeverity}(a,h) =
\sum_{c}
\mathbf{1}\!\big[\mathrm{constraint}_c\ \mathrm{violated\ in\ simulation}\big]
\cdot \mathrm{severity}_c.
$$

The final decision gate is

$$
\mathrm{Pass}(a) =
\mathrm{SyntaxOK}(a)
\cdot \mathrm{TypeOK}(a)
\cdot \mathrm{LocalInvariantOK}(a,B_t)
\cdot \mathrm{GlobalInvariantOK}(a,B_t)
\cdot \mathrm{SemanticEqOK}(a,G)
\cdot \mathrm{SimSafeOK}(a,W_t).
$$

The semantic condition is

$$
\mathrm{SemanticEqOK}(a,G)=1
\iff
\mathrm{Semantics}(\mathrm{Compile}(a)) \models G.\mathrm{objective}
\ \land\
\mathrm{preserves}\!\big(G.\mathrm{invariants}\big).
$$

Instead of producing a binary accept-or-reject signal, the framework returns a proof package:

$$
P(a) =
\left\{
\begin{array}{l}
\mathrm{evidence\_set},\\
\mathrm{active\_hypotheses},\\
\mathrm{assumptions},\\
\mathrm{checked\_invariants},\\
\mathrm{semantic\_equivalence\_result},\\
\mathrm{simulation\_summary},\\
\mathrm{confidence},\\
\mathrm{approval\_level},\\
\mathrm{rollback\_plan}
\end{array}
\right\}.
$$

This makes the final decision auditable and operationally actionable.

## 11. Training Objectives

Each training instance is represented as

$$
(q, O_{1:t}, G, B_t, R_t, H_t, a_t, P_t, y_t).
$$

The total objective is

$$
\mathcal{L}_{\mathrm{total}} =
\mathcal{L}_{\mathrm{intent}}
+ \lambda_1 \mathcal{L}_{\mathrm{retr}}
+ \lambda_2 \mathcal{L}_{\mathrm{hyp}}
+ \lambda_3 \mathcal{L}_{\mathrm{plan}}
+ \lambda_4 \mathcal{L}_{\mathrm{verify}}
+ \lambda_5 \mathcal{L}_{\mathrm{equiv}}
+ \lambda_6 \mathcal{L}_{\mathrm{calib}}
+ \lambda_7 \mathcal{L}_{\mathrm{temp}}.
$$

The individual terms are defined as

$$
\mathcal{L}_{\mathrm{intent}} =
\mathrm{CE}(G_{\mathrm{pred}}^{\mathrm{task}}, G_{\mathrm{true}}^{\mathrm{task}})
+ \mathrm{CE}(G_{\mathrm{pred}}^{\mathrm{inv}}, G_{\mathrm{true}}^{\mathrm{inv}})
+ \mathrm{CE}(G_{\mathrm{pred}}^{\mathrm{act}}, G_{\mathrm{true}}^{\mathrm{act}}),
$$

$$
\mathcal{L}_{\mathrm{retr}} =
- \log
\frac{\exp(\mathrm{score}(r^{+}))}
{\sum_{r}\exp(\mathrm{score}(r))},
$$

$$
\mathcal{L}_{\mathrm{hyp}} =
- \sum_{k} y_k \log p(h_k \mid B_t,G,R_t),
$$

$$
\mathcal{L}_{\mathrm{plan}} =
\mathrm{CE}(a_t^{\mathrm{pred}}, a_t^{\star})
- \eta\, Q_{\mathrm{safe}}(a_t^{\mathrm{pred}}),
$$

$$
\mathcal{L}_{\mathrm{verify}} =
\mathrm{CE}(\mathrm{pass}_{\mathrm{pred}}, \mathrm{pass}_{\mathrm{true}})
+ \mathrm{margin\_loss}(\mathrm{safe\_score}, \mathrm{unsafe\_score}),
$$

$$
\mathcal{L}_{\mathrm{equiv}} =
\mathrm{CE}(\mathrm{eq}_{\mathrm{pred}}, \mathrm{eq}_{\mathrm{true}}),
$$

$$
\mathcal{L}_{\mathrm{calib}} =
\mathrm{Brier}(\mathrm{confidence}_{\mathrm{pred}}, \mathrm{outcome}_{\mathrm{true}})
+ \mathrm{ECE},
$$

$$
\mathcal{L}_{\mathrm{temp}} =
\sum_{s} \mathrm{ReconLoss}(\mathrm{state}_{s}^{\mathrm{pred}}, \mathrm{state}_{s}^{\mathrm{true}}).
$$

The temporal auxiliary term is crucial because it forces the state encoder to retain evolution structure instead of collapsing into a purely task-specific embedding.

## 12. Inference Procedure

At inference time, the framework iterates through belief update, hypothesis refinement, planning, simulation, and verification:

$$
\textbf{for}\ t = 1,\ldots,T:
\quad
G \leftarrow \mathrm{Compile}(q),\ 
B_t \leftarrow \mathrm{UpdateBelief}(B_{t-1}, O_t),\ 
R_t \leftarrow \mathrm{Retrieve}(G,B_t,M),
$$

$$
p(H_t) \leftarrow \mathrm{InferHypothesis}(B_t,G,R_t),\ 
A_t^{\mathrm{safe}} \leftarrow \mathrm{EnumerateAndFilter}(B_t,G,p(H_t)),
$$

$$
a_t \leftarrow \mathrm{Plan}(A_t^{\mathrm{safe}},B_t,G,p(H_t)).
$$

$$
\textbf{if}\ a_t \ \textbf{is evidence action},\ 
O_{t+1} \leftarrow \mathrm{ExecuteReadOnly}(a_t),\ 
\textbf{continue}.
$$

$$
T_t \leftarrow \mathrm{DecodeActionTemplate}(a_t,G,B_t,R_t),\ 
P_t \leftarrow \mathrm{VerifyAndSimulate}(T_t,B_t,G,W_t).
$$

$$
\textbf{if}\ \mathrm{pass}\ \land\ \mathrm{stop\_condition},
\ \textbf{return}\ \mathrm{plan},\mathrm{proof};
\ \textbf{else}\ \mathrm{update\ belief\ or\ escalate}.
$$

## 13. Why the Design is Coherent

The framework is intentionally centered on one state object, one uncertainty object, one planner, and one verifier interface. The architecture therefore avoids the common failure mode of assembling retrieval, workflow, simulation, and validation as loosely connected add-ons. Every stage writes to or reads from the same belief state; every executable decision is judged against the same goal object; every safe action must survive the same proof-producing gate. This is what removes the sense of patchwork and turns the system into a single, internally consistent method.
