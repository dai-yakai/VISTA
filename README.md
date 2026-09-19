<h1 align="center">
  VISTA: Verified Belief-Centric LLM Agents<br>
  for Safe Network Management
</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-2605.xxxxx-b31b1b" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/IEEE-Under%20Review-0076b6" alt="IEEE Under Review"></a>
  <a href="#"><img src="https://img.shields.io/badge/Project-Page-5ca904" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-7a7d12" alt="License MIT"></a>
  <a href="mailto:your.email@example.com"><img src="https://img.shields.io/badge/Contact-Email-c75b12" alt="Contact Email"></a>
</p>

<p align="center">
  <a href="#">Yakai Dai</a>,
  <a href="#">Jia Chen</a><sup>*</sup>,
  <a href="#">Xu Huang</a>,
  <a href="#">Rui Dong</a>,
  <a href="#">Tao Fu</a>
</p>

<p align="center">
  <sup>*</sup> Corresponding author
</p>

---

## 📢 News

This work is currently under review. After the paper is accepted, we will progressively release the source code, models, and related materials under the MIT License for the community to use and build on.

## Abstract

Large language model (LLM) agents now commit changes in production network management, where one change can touch routing-protocol state or a Kubernetes NetworkPolicy. The prevailing pipeline emits the change as a single text answer, so the network state, the competing root-cause hypotheses, and the operational invariants travel through the same token stream as the answer. On the NetArena dynamic benchmark this design is unsafe: supervised fine-tuning of a 7B coder backbone on capacity planning reaches 1.0 correctness at 0.0 per-step safety, and the strongest retrieval-augmented prompting baseline matches that correctness while leaving 41% of executed actions unsafe. VISTA gives each hidden quantity a persistent owner: it maintains a typed belief graph fusing topology, configuration, and telemetry, reasons over an explicit hypothesis trace, scores probes and configuration writes under one information-gain utility, and gates every committed action behind a counterfactual verifier whose structured failure reasons re-enter the next round. Against five reproduced NetArena baselines on a shared Qwen2.5-Coder 7B backbone, VISTA holds capacity-planning correctness at 1.000 and raises per-step safety from 0.589 to 0.709; at paper scale safety reaches 0.857 at 0.994 correctness, with routing and Kubernetes held at refusal-heavy operating points.

## Resources

| Item | Status |
| --- | --- |
| Paper | Coming soon |
| Code | Coming soon |
| Dataset | Coming soon |
| Documentation | Coming soon |

## Citation

```bibtex
@article{dai2026vista,
  title={VISTA: Verified Belief-Centric LLM Agents for Safe Network Management},
  author={Dai, Yakai and Chen, Jia and Huang, Xu and Dong, Rui and Fu, Tao},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project will be released under the MIT License.
