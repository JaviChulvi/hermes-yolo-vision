# Conversation evaluation protocol

Status: the original 216-run study is complete and missed the predefined comparative release target. Its frozen protocol and failures remain unchanged. Later corrections are evaluated separately; they do not replace the original result.

The comparison uses the same Hermes version, existing model account, reasoning setting, CPU, media, model weights, and installed dependencies in three arms:

1. Stock Hermes with ordinary terminal/file/native-vision tools.
2. The same environment with a strong YOLO Python/CLI skill (primary baseline).
3. The native plugin and its bundled skill.

Use fresh homes and conversations, interleave arm order, and keep native vision available to all arms. Share warmed model caches fairly; measure installation and first-use downloads separately. Do not let the baseline read the plugin source, skill, job records, or answers. Provider access stays on the authorized host. No new paid services.

Valid photos use Hermes's normal native image attachment path. Corrupt JPEG bytes are attached as a file/document, so the agent can inspect the downloaded file: submitting those bytes directly as a provider image fails before any agent or tool can act. Development exposed that transport error; retain those original failed runs and use a separately identified file-attachment recovery test.

## Cases and scoring

Eight development conversations exercise setup, image detection, video pose, a segmentation export, tracking, text prompts, a follow-up, and recovery. The held-out suite has 24 conversations: two per vision task (14), two tracking, two YOLOE, two follow-ups, and four negative/recovery cases. Repeat each case three times per arm: 216 held-out runs, with at most 20 tool calls and five minutes per conversation.

Success requires the requested operation, valid and relevant artifacts, the case’s frozen target checks, and a truthful final answer. Tool invocation, nonempty detections, or agreement with direct Ultralytics inference alone is insufficient. Preserve failures and timeouts. Dense maps require shape/value checks; without depth ground truth, report map validity rather than depth accuracy. Artifact/answer review must use the same blinded rubric for all arms and disclose whether grading is automated, AI-assisted, or human. Do not substitute an LLM’s self-assessment for artifact checks.

The 24 conversations are descriptive comparison units, not independent images: several reuse a scene or derived frame. Repeats do not create 72 independent cases, and no inferential significance claim is made. Report paired completion, elapsed time including failures, tool calls, invalid artifacts, per-task results, and p50/p95 latency. Speed comparisons use paired valid completions and show excluded pairs. Engineering release targets are at least 90% completion and either 15 percentage points more completion than the skill baseline or 25% less paired latency with no more than five percentage points less completion. These thresholds are not a statistical noninferiority claim.

| Arm | Passed conversations / 72 | All-attempted latency p50 / p95 | Total tool calls |
| --- | --- | --- | --- |
| Stock | 65 (90.3%) | 51.95 / 192.24 s | 462 |
| Strong skill | 68 (94.4%) | 49.28 / 147.98 s | 394 |
| Plugin + skill | 66 (91.7%) | 41.74 / 129.62 s | 468 |

These are the [original results](results/heldout-v1.json) for candidate `952d6077`, graded with the unchanged rubric using AI-assisted artifact review and independent numerical checks. All 216 attempts are included. On the 64 case/repeat pairs where both plugin and strong skill passed, the median per-pair latency ratio was **0.8805** and the summed-time ratio **0.8761**: about 12% less time, below the required 25% reduction. Plugin completion was 2.8 percentage points lower, not the required 15-point improvement. The predefined gate therefore **failed** despite exceeding 90% completion.

The plugin's six failures were three missing outline deliveries, two missing connecting skeleton overlays, and one provider rejection before final delivery after successful inference. The strong skill had four failures; stock had seven. Per-case outcomes, failed checks, all-attempted latencies, paired exclusions, source/runtime hashes and provider-field limits are included in the JSON report.

The study stopped after 118 attempts at its disk guard following auxiliary environment setup, then resumed without rerunning completed cases. The report preserves the administrative harness patch and a sensitivity analysis excluding the two potentially affected case/repeat pairs: 62 valid pairs remain, with median/summed ratios **0.8805 / 0.8806**. Neither analysis meets the speed target. This was a prepared-runtime, preloaded-guide comparison; installation and actual Telegram receipt are outside its timer, and common artifact-manifest instrumentation may impose unequal overhead.

[Separate correction checks](results/corrections-06ff6b4.json) for `06ff6b4` passed **15/15** previously seen image/follow-up cases under the same rubric. Six synthetic late-completion replays produced repeated file references in all three original-guide runs and none in the three corrected-guide runs. Five Linux media checks also passed. These results do not change the original benchmark gate, demonstrate general workout-counting accuracy, or establish exactly-once Telegram delivery. Further temporal-analysis regressions and real-channel acceptance remain pending.

Stop on provider quota/credential failure and report incomplete coverage. Never switch providers or silently remove failed cases to obtain a passing result.

## Application coverage

The application choices come from upstream capabilities, not assumptions that the plugin already solves them:

- **Gym:** tracked joints, observed range of motion, repetition counts, and explicit uncertainty when joints are hidden. [Workout monitoring](https://docs.ultralytics.com/guides/workouts-monitoring/) supplies the upstream AIGym approach.
- **Daily life:** locate a named household object, summarize visible objects, isolate an object, and interpret uncertain classifications. [YOLOE](https://docs.ultralytics.com/models/yoloe/) describes text-prompted localization.
- **Business:** product silhouettes/cutouts, occupancy over time, crossings, queue counts, and aerial rotated boxes. [Segmentation isolation](https://docs.ultralytics.com/guides/isolating-segmentation-objects/), [queue management](https://docs.ultralytics.com/guides/queue-management/), and [object counting](https://docs.ultralytics.com/guides/object-counting/) describe distinct outputs that must not be conflated.

Public demonstration clips with burned-in boxes/counts are excluded from held-out accuracy comparisons. Store source URLs, license/provenance notes, checksums, transforms, and independently frozen labels alongside each fixture manifest. Synthetic static clips remain functional tests only.

## Channel acceptance

CLI conversations must actually call native tools and produce readable artifacts. Telegram acceptance additionally requires an incoming attachment, the originating-chat route, receipt of annotated image/video and data ZIP, polling without injection permission, cancellation, `/new` suppression, compaction followed by `/new`, and repeated result retrieval without duplicate unsolicited delivery. A mocked adapter or `inject_message` returning true does not establish actual delivery.

No finite benchmark tests every possible use case. Publish measured coverage and failures; avoid universal readiness claims.

## Run on an existing Hermes host

Install the `eval` extra into the shared inference runtime, prepare media with that Python, then run the harness with the Python used by the existing Hermes installation. The extra includes Shapely, required to initialize upstream workout/counting Solutions recommended by the strong baseline; all arms get it. Keep credentials on that host; `--auth-home` references its existing provider login. The runner creates isolated homes and an opaque run mapping. It freezes code/skill hashes, weights, runtime versions, model, reasoning setting, fixture hash and Hermes revision in `study.json`; a changed identity requires a new output directory.

```bash
uv pip install --python /path/to/vision/.venv/bin/python --torch-backend cpu '/path/to/hermes-yolo-vision[eval]'
/path/to/vision/.venv/bin/python evals/prepare.py /tmp/yolo-fixtures
/path/to/hermes/python evals/run.py \
  --hermes /path/to/hermes-agent \
  --runtime /path/to/vision/.venv/bin/python \
  --plugin /path/to/hermes-yolo-vision \
  --fixtures /tmp/yolo-fixtures \
  --output /tmp/yolo-development \
  --auth-home "$HOME/.hermes" \
  --models /path/to/warmed-model-cache
```

Development defaults to one repetition. After its checks and version freeze, use a new output directory with `--phase heldout --repeats 3`. The existing provider configuration is used unchanged except for the same low reasoning setting across all arms. The runner checks the explicit interpreter through Hermes's real terminal tool before starting the measured conversation; `-I` prevents the terminal's Python path from selecting Hermes's core dependencies.

The [frozen rubric](RUBRIC.md), [case definitions](cases.py), [fixture provenance](fixtures.json), and [COCO/DOTA annotation subset](annotations.json) define correctness. Results initially remain **ungraded**. Review each case against the rubric with arm labels hidden, record evidence, and only then aggregate. Raw run directories include authenticated-profile links and local traces: do not publish them wholesale.

Use `python evals/inspect_artifacts.py /path/to/study --output /path/to/review --copy-artifacts` with the inference runtime to check saved files and prepare neutral artifact names for review. For evidence copied from another host, add `--source-root /original/study/path`. The inspector reports decoding, dimensions, timestamps and array validity; it leaves grades pending. Apply ground-truth overlap and answer checks separately. Arm labels and paths are hidden in the audit JSON, but artifact formats or embedded metadata may still suggest an implementation; this is not guaranteed reviewer blinding.

[metrics.py](metrics.py) provides box/mask/polygon IoU, one-to-one matching and visible-joint PCK for independent checks. Install the `eval` extra for COCO mask decoding. Normalize each artifact using its documented coordinate/resize conventions and record that normalization with the measured values; these functions do not infer an artifact schema or assign conversation grades.
