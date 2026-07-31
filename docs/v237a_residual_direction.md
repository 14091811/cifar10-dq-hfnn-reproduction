# V237a residual-direction supervision

V237a keeps the V236 model unchanged and replaces standalone quantum-head CE
with a complementary residual-direction objective:

```text
base_prob = softmax(stop_gradient(base_logits))
residual_target = one_hot(label) - base_prob
centered_q = q_logits - mean(q_logits)

loss = CE(base_logits + alpha * q_logits, label)
     + 0.05 * (1 - cosine(centered_q, residual_target))
```

The target is the negative gradient direction of cross-entropy with respect to
the detached classical logits. It asks the quantum branch to model what the
CNN is missing instead of independently duplicating the full classifier.

The CRX and Local-RX variants remain exactly parameter matched at 733 added
parameters. All optimizer, schedule, Haar, residual-scale, and data settings
are inherited unchanged from V236 aux005.
