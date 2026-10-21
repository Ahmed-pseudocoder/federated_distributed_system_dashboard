# Project Explainer: Adaptive Fault-Tolerant Federated Learning System

## 1. One-minute overview

This project demonstrates a small but real federated-learning platform in which multiple independent workers train a shared MNIST image-classification model without sending their raw training images to a central server.

The coordinator sends the current global model to the workers. Each worker trains that model on its own non-IID data partition and returns only a completed model update, sample count, metrics, heartbeat, and telemetry. The coordinator combines valid updates using weighted Federated Averaging, evaluates the new global model, and saves a checkpoint.

The main reliability demonstration is a genuine worker-process failure. Client 2 terminates during local training at a configured round, epoch, and completed batch. The coordinator detects the missing heartbeat, identifies the exact last successful training point, rejects the missing/incomplete update, restores the most recent valid global checkpoint, removes the failed worker from scheduling, and continues training with healthy workers.

The system has two deployment modes:

1. **Local process mode:** three independent worker processes run on one computer. This is convenient for development and repeatable demonstrations.
2. **LAN mode:** two laptops run independent HTTP workers. The coordinator monitors their real hostnames, IP addresses, PIDs, heartbeats, telemetry, and model updates across the network.

> Presentation summary: “We built a federated-learning coordinator that can monitor independent workers, survive a real worker crash, recover from a checkpoint, dynamically continue with healthy workers, and show the complete event timeline on a live dashboard.”

---

## 2. Problem being solved

Traditional centralized machine learning requires collecting training data on one server. That is not always acceptable because data may be private, large, geographically distributed, or owned by separate organizations.

Federated learning solves the data-locality problem, but it creates distributed-systems problems:

- workers may be slow or unavailable;
- a worker may crash halfway through training;
- the coordinator must not aggregate partial updates;
- the system must know the worker’s last confirmed training point;
- training should continue without restarting from round 1;
- operators need a clear way to investigate failures;
- hyperparameters still need to be selected using real experiments.

This project combines the machine-learning and distributed-systems sides of that problem in one demonstrable MVP.

---

## 3. Project objectives

The implementation is designed to prove the following points:

- training data remains local to each client;
- workers perform genuine local PyTorch training;
- model updates are combined using sample-weighted FedAvg;
- workers are independent operating-system processes;
- workers send real heartbeats and batch-level telemetry;
- a worker failure is a real process termination, not a fake UI status;
- the coordinator detects failure using missed heartbeats;
- incomplete work is excluded from aggregation;
- the latest valid global checkpoint is retained or restored;
- future workload is assigned only to available workers;
- federated training continues after failure;
- all important events and metrics are persisted;
- the dashboard displays only actual persisted runtime information;
- HPO trials use the same federated engine and record real outcomes;
- the system can operate across two physical laptops on a private LAN.

---

## 4. Important concepts

### Federated learning

Federated learning trains one shared model using data stored on multiple clients. The coordinator does not request the clients’ raw training images or labels. Instead, it distributes a model, receives trained model parameters, and aggregates them.

### Global model

The global model is the coordinator’s authoritative CNN. At the start of each round, all selected clients receive the same global parameters.

### Local training

Each selected client loads the global parameters and performs one or more local epochs using only its own MNIST partition.

### Federated round

A round is one complete distribute-train-collect-aggregate-evaluate-checkpoint cycle.

### Non-IID data

The client datasets have different digit distributions rather than identical random samples. For example:

- Client 1 is mainly digits 0, 1, 2, and 3.
- Client 2 is mainly digits 4, 5, and 6.
- Client 3 is mainly digits 7, 8, and 9.

The partitions are deterministic and disjoint. This better represents real federated environments, where different participants commonly have different data distributions.

### Heartbeat

A heartbeat is a periodic liveness message sent independently of model training. It tells the coordinator that the worker is still reachable and includes its latest round, epoch, batch, step, status, and PID.

### Telemetry

Telemetry is detailed batch-level training information. It records the exact last completed unit of work, including loss and accuracy.

### Checkpoint

A checkpoint is a durable snapshot of the latest valid global model and its metrics. It is written only after a successful aggregation.

---

## 5. System architecture

### Local three-client mode

```text
                         FL COORDINATOR
       global model · scheduler · fault detector · SQLite
                               |
               round command + global parameters
              /                |                 \
             v                 v                  v
      CLIENT 1 PROCESS  CLIENT 2 PROCESS   CLIENT 3 PROCESS
      local partition   local partition    local partition
      local training    local training     local training
      heartbeat         heartbeat          heartbeat
      telemetry         telemetry          telemetry
              \                |                 /
                 complete model updates only
                               |
                      weighted FedAvg
                               |
                 evaluation + checkpoint
```

Communication uses multiprocessing queues. The workers are still independent operating-system processes, so terminating one does not directly terminate the coordinator or the other workers.

### Two-laptop LAN mode

```text
                     COORDINATOR HTTP SERVICE
             global model · task API · monitoring
             scheduler · checkpoints · SQLite
                      192.168.1.10:8000
                         /              \
                        /                \
             model task                  model task
                      v                    v
          LAPTOP A / CLIENT 1       LAPTOP B / CLIENT 2
          hostname + IP + PID       hostname + IP + PID
          local MNIST data          local MNIST data
          heartbeat thread          heartbeat thread
          local PyTorch train       local PyTorch train
                      \                    /
                       complete updates only
```

With only two physical laptops, Laptop A can run the coordinator in one terminal and Client 1 in another terminal. Laptop B runs Client 2. A third dedicated coordinator computer is optional.

LAN workers communicate through authenticated HTTP endpoints:

- `GET /worker-config/{client_id}` — obtain coordinator-controlled training settings;
- `POST /register` — register hostname, IP, PID, platform, sample count, and digit distribution;
- `POST /heartbeat` — report liveness and current progress;
- `POST /telemetry` — persist the exact completed-batch state;
- `POST /event` — report structured lifecycle events;
- `GET /task/{client_id}` — poll for a round assignment and global model;
- `POST /update` — upload a complete trained model update;
- `GET /status` — inspect current system and worker state;
- `GET /events` — inspect recent coordinator events.

All endpoints except `/health` require the shared bearer token.

---

## 6. Main software components

| Component | Responsibility |
|---|---|
| `core/config.py` | Central configuration for clients, rounds, training, heartbeat, failure injection, paths, and reproducibility |
| `core/model.py` | Small PyTorch CNN used by every mode |
| `core/data.py` | MNIST loading and deterministic disjoint non-IID partitioning |
| `client/trainer.py` | Genuine local optimization, telemetry, metrics, and update creation |
| `client/heartbeat.py` | Background heartbeat emission for local workers |
| `client/failure_simulator.py` | Deterministic process crash, hang, or network-timeout injection |
| `coordinator/aggregator.py` | Sample-weighted Federated Averaging and update validation |
| `coordinator/registry.py` | Worker status, reliability, latency, availability, and progress |
| `coordinator/scheduler.py` | Dynamic healthy-client selection |
| `coordinator/fault_detector.py` | Missed-heartbeat timeout detection |
| `coordinator/checkpoint.py` | Atomic global-checkpoint save and restore |
| `coordinator/event_logger.py` | SQLite persistence for events, telemetry, workers, rounds, state, and HPO |
| `coordinator/server.py` | Local multiprocessing coordinator |
| `distributed/coordinator.py` | Authenticated HTTP coordinator for remote laptops |
| `distributed/worker.py` | Independent LAN worker that polls tasks and uploads updates |
| `distributed/codec.py` | Tensor-only model-state encoding and decoding |
| `dashboard/app.py` | Streamlit dashboard based entirely on SQLite runtime data |
| `hpo/` | Isolated federated hyperparameter trials |

---

## 7. Machine-learning model

The project uses a lightweight CNN suitable for CPU demonstrations:

```text
Input: 1 × 28 × 28 MNIST image
    ↓
Conv2d(1, 32, kernel=3)
ReLU
MaxPool2d
    ↓
Conv2d(32, 64, kernel=3)
ReLU
MaxPool2d
    ↓
Flatten
Linear(1600, 128)
ReLU
Linear(128, 10)
    ↓
Digit probabilities/classes 0–9
```

Each worker trains with Adam and cross-entropy loss. CUDA is used automatically when available, but the complete demonstration works on ordinary CPU laptops.

---

## 8. Normal federated-training flow

The following sequence happens during every healthy round.

### Step 1: Select workers

The coordinator asks the dynamic scheduler for available workers. Failed or unavailable workers are excluded.

### Step 2: Distribute the global model

Each selected worker receives the same global model state and the round configuration.

### Step 3: Train locally

Every worker:

1. loads the received global parameters;
2. trains using only its own local data loader;
3. sends heartbeats from a background thread;
4. emits telemetry after completed batches;
5. calculates local loss and accuracy;
6. creates a model update only after all local epochs finish.

### Step 4: Validate updates

The coordinator accepts an update only when:

- it belongs to the current assigned task and round;
- the client was selected;
- the update is explicitly marked complete;
- parameter keys and tensor shapes match the global model;
- floating-point tensors contain finite values;
- the task has not already been accepted.

Duplicate network submissions are handled idempotently using the task ID.

### Step 5: Perform weighted FedAvg

If client (k) trained on (n_k) samples and produced parameters (w_k), the new global parameters are:

```text
                 Σ (nₖ × wₖ)
global model =  ─────────────
                     Σ nₖ
```

Therefore, a client that trained on more samples has a proportionally larger contribution.

### Step 6: Evaluate the global model

The coordinator evaluates the aggregated global model on a bounded MNIST test set and records actual loss and accuracy.

### Step 7: Save the checkpoint

An atomic `round_NNN.pt` file is created. It contains:

- round number;
- global model state;
- optimizer-state placeholder;
- actual evaluation metrics;
- timestamp.

### Step 8: Persist the round

Round duration, selected clients, successful update count, active-client count, global loss, and accuracy are written to SQLite. The next round then begins.

---

## 9. Heartbeat and telemetry flow

Heartbeats and telemetry serve different purposes.

### Heartbeat message

```json
{
  "client_id": 2,
  "pid": 5604,
  "round": 4,
  "epoch": 2,
  "batch": 37,
  "global_step": 303,
  "status": "training",
  "timestamp": "actual runtime timestamp"
}
```

The heartbeat proves that the worker process is still responsive.

### Telemetry message

```json
{
  "client_id": 2,
  "round": 4,
  "epoch": 2,
  "batch": 37,
  "global_step": 303,
  "loss": 0.0711,
  "accuracy": 0.9782,
  "status": "training",
  "pid": 5604,
  "timestamp": "actual runtime timestamp"
}
```

Telemetry establishes the worker’s exact last successfully completed batch. The word “completed” matters: the optimizer step finishes first, then telemetry is delivered, and only then is the configured crash triggered.

---

## 10. Real worker-failure flow

The default failure is not a simulated red status on the dashboard. Client 2 actually calls `os._exit(42)`, immediately terminating its operating-system process.

```text
Client 2 completes configured batch
                ↓
Final completed-batch telemetry is sent
                ↓
Client 2 process exits with code 42
                ↓
No model update is produced
                ↓
Heartbeats stop
                ↓
Coordinator timeout expires
                ↓
Worker is marked FAILED
```

The coordinator uses this rule:

```text
current_time - last_heartbeat > heartbeat_timeout
```

The default timeout is three seconds. It is configurable so a fast classroom rehearsal can use a shorter interval.

---

## 11. Recovery flow

After detecting a failure, the coordinator performs the following actions in order:

1. **Mark the worker failed.** Its availability becomes false and reliability is reduced.
2. **Capture the final evidence.** The coordinator reads the last persisted telemetry and heartbeat.
3. **Create a `WORKER_FAILURE` event.** This includes client, host, IP, round, epoch, batch, step, loss, accuracy, timestamps, and detection latency.
4. **Discard incomplete work.** Because the crashed worker never sent a complete update, it cannot enter FedAvg.
5. **Restore the latest valid checkpoint.** For a default round-4 failure, this is `round_003.pt`.
6. **Reassign participation.** The scheduler selects only healthy workers.
7. **Aggregate valid updates.** Updates from healthy clients remain valid because they started from the same global checkpoint.
8. **Complete the current round.** Training does not return to round 1.
9. **Continue later rounds.** The failed worker stays excluded.
10. **Mark recovery complete.** Actual recovery duration is added to the original failure event.

The key continuity property is:

```text
Round 1 ✓
Round 2 ✓
Round 3 ✓  ← latest valid checkpoint
Round 4: Client 2 crashes, healthy updates continue
Round 5 ✓  ← no restart from round 1
...
Round 10 ✓
```

---

## 12. Dynamic worker allocation

Each worker has a simple, explainable score:

```text
latency_score = 1 / (1 + recent_latency_seconds)

worker_score = reliability × 0.75
             + latency_score × 0.25
```

Reliability has the larger weight because a fast worker is not useful if it frequently fails. A failed worker is marked unavailable and is not selected in future rounds.

This scheduler is intentionally simple for an MVP. Its purpose is to make allocation decisions transparent during a presentation.

---

## 13. Dashboard flow

The Streamlit dashboard is an observer, not the source of truth. It reads the coordinator’s SQLite database and never generates fake metrics.

### Global training

Shows current/total rounds, evaluated global accuracy and loss, active and failed clients, run status, and round duration.

### Live worker monitor

Shows every client’s status, PID, hostname, IP address in LAN mode, transport, heartbeat age, round, epoch, batch, global step, loss, accuracy, reliability, and availability.

### Failure investigation

Shows the exact failure type, round, epoch, completed batch, global step, final loss/accuracy, last heartbeat, detection timestamp, detection latency, checkpoint, recovery status, and recovery time.

### Event timeline

Shows ordered events such as registration, selection, local training, batch completion, update reception, worker failure, checkpoint restore, reassignment, and round completion.

### Training charts

Shows actual global accuracy/loss by round, active-client count, round duration, and local client loss.

### HPO results

Shows real trials with learning rate, batch size, local epochs, accuracy, duration, and status. The best completed trial is highlighted.

---

## 14. Persistent data and evidence

SQLite contains six main data groups:

| Table | Stored information |
|---|---|
| `events` | Ordered lifecycle, training, failure, recovery, and checkpoint events |
| `telemetry` | Per-client batch progress, loss, accuracy, status, and PID |
| `workers` | Latest worker health, reliability, latency, progress, host, and network identity |
| `rounds` | Global loss/accuracy, duration, participation, and successful updates |
| `state` | Current run configuration and overall training status |
| `hpo_trials` | Hyperparameters, outcomes, duration, result directory, and trial status |

Important event types include:

```text
SYSTEM_STARTED
CLIENT_CONNECTED
CLIENT_HEARTBEAT
ROUND_STARTED
CLIENT_SELECTED
LOCAL_TRAINING_STARTED
BATCH_COMPLETED
LOCAL_TRAINING_COMPLETED
MODEL_UPDATE_RECEIVED
WORKER_FAILURE
CHECKPOINT_RESTORED
WORKLOAD_REASSIGNED
CHECKPOINT_SAVED
ROUND_COMPLETED
TRAINING_COMPLETED
```

Client and coordinator log files provide a second evidence trail independent of the dashboard.

---

## 15. Hyperparameter-optimization flow

The HPO layer reuses the same federated engine instead of using a simplified fake objective.

```text
                         HPO COORDINATOR
                    /          |          \
                   v           v           v
               TRIAL 1     TRIAL 2     TRIAL 3
               own FL       own FL       own FL
               workers      workers      workers
               database     database     database
                    \          |          /
                       compare accuracy
                              ↓
                         best trial
```

The search covers combinations of:

- learning rate: `0.01`, `0.001`, `0.0001`;
- batch size: `32`, `64`;
- local epochs: `1`, `2`.

Each trial has isolated checkpoints, logs, and results. Accuracy and duration come from actual execution.

---

## 16. Why this is genuine federated learning

This project is not merely ordinary distributed data-parallel training.

| Federated property | How the project demonstrates it |
|---|---|
| Data locality | Each worker loads and trains on its own local MNIST partition |
| Non-IID clients | Each client has a different dominant digit distribution |
| Local optimization | Clients perform complete local epochs before communication |
| Model exchange | Global parameters and completed updates cross the boundary |
| No raw-data upload | Images and labels are never placed in coordinator messages |
| Federated aggregation | Client states are combined with sample-weighted FedAvg |
| Client selection | The coordinator dynamically chooses available participants |
| Partial participation | Training continues after one client becomes unavailable |

The local mode demonstrates process isolation. The LAN mode additionally demonstrates network isolation between worker applications on different machines.

---

## 17. Verified execution evidence

### Full local default run

One observed Windows CPU run produced:

```text
Rounds completed:           10 / 10
Failure client:             Client 2
Failure type:               PROCESS_CRASH
Failure round:              4
Failure epoch:              2
Last completed batch:       37
Client global step:         303
Process exit code:          42
Detection latency:          3095 ms
Checkpoint restored:        round_003.pt
Final global accuracy:      95.65%
Final global loss:          0.1895
```

Round 4 accepted updates only from Clients 1 and 3, and rounds 5–10 continued without Client 2.

### Network rehearsal

The LAN protocol was integration-tested with an HTTP coordinator and two independently launched worker processes:

```text
Round 1 updates:            Client 1 + Client 2
Client 2 crash point:       Round 2 / Epoch 1 / Batch 2 / Step 5
Detection latency:          2022 ms
Checkpoint restored:        round_001.pt
Round 2 updates:            Client 1 only
Round 3 updates:            Client 1 only
Final status:               COMPLETED
```

The rehearsal used only 96 training samples per worker and 128 test samples, so its accuracy was not intended as a model-quality benchmark. Its purpose was to verify network orchestration and fault recovery.

### Automated verification

The test suite currently contains 13 passing tests covering:

- CNN output shape;
- deterministic non-IID partitioning;
- disjoint client datasets;
- weighted FedAvg math;
- sample-count weighting;
- incomplete-update rejection;
- failed-client exclusion;
- reliability-based scheduling;
- checkpoint save/restore;
- heartbeat timeout detection;
- SQLite event and telemetry persistence;
- tensor state transport round-trip;
- LAN coordinator registration, worker metadata, and API routes.

---

## 18. Recommended live presentation flow

### Before the presentation

1. Connect both laptops to the same private Wi-Fi network.
2. Clone/pull the same repository on both machines.
3. Install `requirements.txt` in a virtual environment.
4. Note the coordinator laptop’s IPv4 address using `ipconfig`.
5. Confirm port 8000 is allowed on the private firewall profile.
6. Use the same `FL_SHARED_TOKEN` on both machines.
7. Run the fast rehearsal once before presenting.

### During the presentation

1. **Show the architecture diagram.** Explain that models move, while raw training data stays local.
2. **Start the coordinator.** Point out that it waits for the required workers.
3. **Start Client 1 on Laptop A.** Show its hostname and registration.
4. **Start Client 2 on Laptop B.** Show that the coordinator sees a second IP/PID.
5. **Open the dashboard.** Select the LAN run database.
6. **Watch round 1.** Both workers should send completed updates.
7. **Watch Client 2 terminate.** Its terminal genuinely closes/exits at the configured point.
8. **Wait for heartbeat detection.** Show the dashboard change from healthy to failed.
9. **Open failure investigation.** Point to the last epoch, batch, step, loss, heartbeat, and detection latency.
10. **Show checkpoint recovery.** Explain why the incomplete update is excluded.
11. **Watch the next round.** Only Client 1 is selected, proving dynamic continuation.
12. **Finish with the event timeline and charts.** Emphasize that all values are persisted runtime evidence.

### Fast rehearsal command

Coordinator laptop:

```powershell
$env:FL_SHARED_TOKEN = "replace-with-your-demo-token"

python run_network_coordinator.py `
  --host 0.0.0.0 `
  --port 8000 `
  --clients 2 `
  --rounds 3 `
  --local-epochs 1 `
  --max-train-samples 96 `
  --max-test-samples 128 `
  --failure-round 2 `
  --failure-epoch 1 `
  --failure-batch 2 `
  --heartbeat-timeout 2 `
  --run-id guide_demo
```

Client 1:

```powershell
$env:FL_SHARED_TOKEN = "replace-with-your-demo-token"
python run_network_worker.py --client-id 1 --coordinator http://COORDINATOR_IP:8000
```

Client 2:

```powershell
$env:FL_SHARED_TOKEN = "replace-with-your-demo-token"
python run_network_worker.py --client-id 2 --coordinator http://COORDINATOR_IP:8000
```

Dashboard on the coordinator laptop:

```powershell
python -m streamlit run dashboard/app.py
```

Select `guide_demo.db` in the sidebar.

---

## 19. Suggested presentation script

### Opening

“Our project combines federated learning with fault-tolerant distributed orchestration. The goal is to train one global model using data stored on separate workers while ensuring that a worker crash does not destroy the complete experiment.”

### Data locality

“Each client owns a disjoint, non-IID MNIST partition. The images and labels remain inside that worker. The coordinator only distributes model parameters and receives completed model updates, sample counts, metrics, heartbeats, and telemetry.”

### Normal round

“At the beginning of each round, the coordinator dynamically selects healthy clients and sends the same global model. Each client trains locally, sends live progress, and uploads a model update only after completing all local epochs. The coordinator validates the updates, applies weighted Federated Averaging, evaluates the global model, and saves a checkpoint.”

### Failure

“At a deterministic training point, Client 2 genuinely terminates its process. Because heartbeat and telemetry are separate from the final update, the coordinator knows its exact last completed batch but never receives an incomplete model update.”

### Recovery

“After the heartbeat timeout, the coordinator marks Client 2 failed, stores the complete failure event, restores the latest valid checkpoint, removes Client 2 from scheduling, aggregates only healthy completed updates, and continues without restarting from round 1.”

### Dashboard

“The dashboard is based entirely on persisted SQLite data. It displays live worker identity and progress, failure evidence, checkpoint recovery, the event timeline, global metrics, and HPO outcomes. No accuracy or failure value is hardcoded.”

### Closing

“The result is a small but credible distributed ML platform: data remains decentralized, workers are observable, failures are real, recovery is automatic, and the complete behavior can be demonstrated across two laptops.”

---

## 20. Common guide/viva questions

### Why is this federated learning and not just multiprocessing?

Multiprocessing is only one transport used by the local demo. The learning semantics are federated: every client owns a separate dataset, performs local optimization, and returns only model parameters. The LAN mode proves that the same design works across independent networked machines.

### Does the coordinator ever see raw client images?

No. Client loaders and raw images exist only inside worker processes. Coordinator messages contain model state, sample counts, metrics, heartbeat, telemetry, and events.

### Why use non-IID partitions?

Real federated clients usually do not have identical data. Label-skewed partitions demonstrate this challenge more realistically than identical random splits.

### Why is weighted averaging required?

Clients may have different sample counts. Weighting by local sample count prevents a small client from having the same influence as a much larger client.

### How do you know the crash is real?

The configured worker calls `os._exit(42)`. In local mode, the operating-system exit code is recorded. In LAN mode, the worker process disappears, HTTP communication stops, and the coordinator independently detects the missed heartbeat.

### Why is the failed client’s update discarded?

A mid-training model is not a complete round result. Aggregating it could corrupt the global update and violates the round protocol. Only explicitly completed, validated task updates are accepted.

### Why restore a checkpoint if healthy workers already returned updates?

All selected workers started from the same previous global checkpoint. Restoring it re-establishes the authoritative base state. Healthy complete updates remain valid relative to that state, while the failed client’s absent update is excluded.

### Does training restart after failure?

No. The last valid global checkpoint is used, the current round completes using available valid updates, and later rounds continue with healthy workers.

### How is a failure distinguished from a slow worker?

The MVP uses a configurable heartbeat and round timeout. In production, adaptive timeouts and more advanced failure detectors could reduce false positives in heterogeneous networks.

### What happens if every worker fails?

The round cannot meet its minimum complete-update requirement, so the coordinator marks training failed rather than creating an invalid aggregate.

### Can a worker submit the same update twice?

The LAN coordinator tracks unique task IDs. A repeated accepted task is treated idempotently and is not aggregated twice.

### Is the LAN communication production-secure?

No. It uses a bearer token over plain HTTP for a controlled private-network demonstration. Production deployment requires TLS, per-worker credentials, signed requests/updates, replay protection, secret management, and hardened serialization.

### Is the system privacy-preserving?

It demonstrates data locality, which is one privacy advantage of federated learning. It does not yet provide secure aggregation, differential privacy, or defenses against information leakage from model updates.

### Why use SQLite?

SQLite provides durable structured persistence without requiring an external database server. It is appropriate for a portable MVP and allows the dashboard to query real runtime history safely.

### Why use a simple scheduler?

The reliability/latency score is easy to explain and verify. The project prioritizes correct, observable fault tolerance over a complex scheduling algorithm.

### Why use MNIST?

MNIST is small, familiar, reproducible, and fast enough for a CPU demonstration. It keeps attention on federated orchestration rather than dataset preparation.

---

## 21. Limitations

This MVP intentionally does not implement:

- encrypted/TLS model transport;
- secure aggregation;
- differential privacy;
- protection against malicious or poisoned updates;
- per-worker certificates or production identity management;
- durable message queues;
- coordinator replication or leader election;
- automatic replacement of a failed physical worker;
- WAN optimization or compressed model deltas;
- a production-scale dataset or model;
- fairness-aware or energy-aware scheduling.

These are limitations of production readiness, not failures of the demonstrated objectives.

---

## 22. Future improvements

The most valuable next steps are:

1. add HTTPS and per-worker authentication;
2. add secure aggregation so the coordinator cannot inspect individual updates;
3. add differential privacy and privacy accounting;
4. sign model tasks and updates;
5. add update anomaly and poisoning detection;
6. compress model updates for slower networks;
7. support automatic worker restart/rejoin;
8. persist task delivery in a durable queue;
9. replicate coordinator state for high availability;
10. add adaptive timeout estimation using observed latency;
11. support more datasets and larger models;
12. deploy workers in containers or edge devices.

---

## 23. Final takeaway

The project demonstrates the full lifecycle of a fault-tolerant federated-learning round:

```text
distributed local data
        ↓
global model distribution
        ↓
independent local training
        ↓
heartbeat + exact telemetry
        ↓
real worker crash
        ↓
timeout detection
        ↓
incomplete update rejection
        ↓
checkpoint recovery
        ↓
dynamic healthy-worker selection
        ↓
continued FedAvg training
        ↓
persistent dashboard evidence
```

The central message for the guide is:

> The system does not merely display fault tolerance—it executes it. Workers train real models on local data, failures terminate real processes, heartbeat loss is actually detected, incomplete updates are excluded, checkpoints are genuinely restored, and training continues with healthy workers while the dashboard shows the complete evidence trail.
