# GPT-5.6 Sol local evaluator result — 2026-09-18

Branch: `experiment/openai-sol`

Configuration:
- ASR: `large-v3-turbo`
- LLM provider: OpenAI
- Model: `gpt-5.6-sol`
- Reasoning effort: `high`
- Evidence pipeline: existing segment/quote alignment from this branch
- Dataset: official local evaluator, 39 training conversations / 390 questions
- This was a development-only local-evaluator experiment, not an online competition validation run.

## Result

```text
Attempt statistics
  questions            390
  correct              384
  unanswered           0
  conversations        39
  failed conversations 0
  timeouts             0

Accuracy by question type
  positive             0.969  (189/195)
  hard_negative        1.000  (142/142)
  off_topic            1.000  (53/53)

Evidence localization
  mean tIoU                0.505  (over 195 annotated yes questions)
  no span returned         6
  tIoU when answered yes   0.521  (diagnostic, n=189, not scored)

Round trip
  per conversation            15785 ms mean
  worst conversation          41479 ms
  per question                 1578 ms
  budget                      60000 ms
  worst request budget used   69%

Accuracy:  0.985
Mean tIoU: 0.505
Score:     0.697
```

## Interpretation

The frontier-model experiment nearly solved classification:
- 98.5% overall accuracy
- 100% hard-negative accuracy
- 100% off-topic accuracy
- 96.9% positive accuracy

The dominant bottleneck is evidence localization rather than yes/no reasoning. Mean tIoU was only 0.505, with 6 positive questions returning no span.

The score decomposes approximately as:
- accuracy contribution: `0.985 * 0.4 = 0.394`
- evidence contribution: `0.505 * 0.6 = 0.303`
- total: `0.697`

This result should not be directly compared as a model-vs-model benchmark with the best hidden online-validation score because the datasets differ.

For reference, the current best online validation result previously observed was:
- score: `0.7414863491598819`
- model setup: Qwen3.5 27B + tight evidence pipeline
- hidden validation dataset, 19 conversations / 190 questions

## Suggested next experiment

Keep the strong classifier and improve evidence localization. In particular:
1. log all low-tIoU cases with question, gold span, predicted span, selected segment IDs, evidence quote, and nearby transcript;
2. inspect zero-overlap cases;
3. test direct timestamp prediction or stronger word-boundary alignment instead of relying only on segment IDs + quote matching.
