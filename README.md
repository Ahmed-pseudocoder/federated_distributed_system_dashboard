# Adaptive Fault-Tolerant Federated Learning MVP

A working, process-isolated federated-learning demonstration for MNIST. Three clients keep their local non-IID training data inside independent Python processes. The coordinator distributes only model weights, receives structured telemetry and complete model updates, performs sample-weighted FedAvg, checkpoints every successful round, and continues after a real worker failure.

## What the demo proves

The default run starts three client processes. Each loads a reproducible, label-skewed MNIST partition and trains a small PyTorch CNN locally. In round 4, Client 2 finishes the configured batch, emits the exact completed-batch telemetry, and calls `os._exit(42)`. It does not send a model update. After the real heartbeat deadline expires, the coordinator:

1. marks Client 2 unavailable;
2. persists its last round, epoch, batch, step, loss, accuracy, PID, and heartbeat;
3. excludes the incomplete local work;
4. restores the latest valid global checkpoint (round 3 by default);
5. reassigns participation to healthy clients;
6. aggregates the complete updates that were trained from that checkpoint; and
7. continues later rounds without returning to round 1.

This is federated learning, rather than ordinary data-parallel training, because each client owns a distinct local dataset and performs multiple local optimization steps. Only model state, sample count, metrics, heartbeats, and telemetry cross the process boundary. Client training images and labels never enter coordinator messages.

## Architecture

```text
Coordinator (global model, scheduler, fault detector, SQLite)
        │ model snapshot / round command
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
 Client 1 process  Client 2 process  Client 3 process
 local MNIST       local MNIST       local MNIST
 heartbeat         heartbeat         heartbeat
 telemetry         telemetry         telemetry
        └──── complete model updates ────┘
                         │
                   weighted FedAvg
                         │
                 evaluation + checkpoint
```

The transport is deliberately small and transparent: multiprocessing command queues and a manager-backed event queue. The latter makes the final telemetry send synchronous before an injected hard exit. Modules are separated so a socket or HTTP transport can replace the local queues later.

## Repository map

- `core/`: configuration, CNN, MNIST partitioning, message schema, serialization
- `client/`: process entry point, local trainer, heartbeat, telemetry, failure injection
- `coordinator/`: server, registry, scheduling, FedAvg, detection, checkpoints, SQLite events
- `dashboard/`: Streamlit dashboard driven only by SQLite data
- `hpo/`: isolated parallel trials using the same FL engine
- `experiments/`: basic, dynamic-allocation, and fault-tolerance presets
- `tests/`: model, partition, aggregation, scheduler, checkpoint, timeout, and persistence tests

## Installation

Python 3.10 or 3.11 is recommended. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MNIST is downloaded automatically on the first run. GPU use is automatic inside clients when CUDA is available; CPU is fully supported.

## Run the fault-tolerant demo

```powershell
python run_fl.py
```

Default parameters are three clients, ten rounds, two local epochs, 1,200 examples per client, a 0.5-second heartbeat, a 3-second timeout, and a Client 2 process crash after round 4 / epoch 2 / batch 37 is completed. The batch number is a **completed-batch** coordinate: telemetry is committed after its optimizer step, then the process exits before producing a round update.

Useful variants:

```powershell
python run_fl.py --mode basic --rounds 5
python run_fl.py --clients 3 --rounds 10 --failure-client 2 --failure-round 4 --failure-epoch 2 --failure-batch 37
python run_fl.py --failure-mode none
python run_fl.py --rounds 2 --local-epochs 1 --max-train-samples 192 --max-test-samples 256
```

An observed default run on Windows CPU completed all ten rounds in 137 seconds:

```text
Round 3/10 accuracy=0.9500 loss=0.1588 updates=3
Worker 2 failed at round=4 epoch=2 batch=37 step=303
checkpoint=round_003.pt process_exitcode=42
Round 4/10 accuracy=0.9500 loss=0.1777 updates=2
...
Round 10/10 accuracy=0.9565 loss=0.1895 updates=2
```

These are recorded execution results, not dashboard defaults; every new run computes its own values.

For an injected batch to be reached, the local partition must contain at least `failure_batch × batch_size` samples. The defaults intentionally satisfy this.

## Dashboard

After starting a run, open another terminal:

```powershell
streamlit run dashboard/app.py
```

The dashboard never invents status or metrics. It reads the selected SQLite run database and displays:

- round progress, evaluated global loss/accuracy, worker counts, and duration;
- worker PID, health, heartbeat age, exact training coordinate, loss, and accuracy;
- the complete failure/recovery record;
- the persisted event timeline;
- global and client training charts; and
- measured HPO trials with the best result highlighted.

## Dynamic allocation

Eligible clients are sorted by an explainable score:

```text
score = reliability × 0.75 + (1 / (1 + recent_latency_seconds)) × 0.25
```

Every healthy client participates by default. A failure immediately makes a worker unavailable and lowers its reliability, so subsequent rounds select only healthy processes. The round does not wait indefinitely: heartbeat and round timeouts are both bounded.

## Fault detection, telemetry, and recovery

Clients emit heartbeat messages every 0.5 seconds from a dedicated thread. Batch telemetry contains client ID, round, epoch, completed batch, client-global step, timestamp, loss, accuracy, status, and PID. The coordinator retains every telemetry row plus a latest worker snapshot.

A missing heartbeat is a failure when `now - last_heartbeat > heartbeat_timeout`. The failure event records the real detection latency, last telemetry, chosen checkpoint, recovery interval, and whether incomplete work was discarded. Unexpected exceptions are logged and naturally enter the same missing-heartbeat decision path.

Checkpoints are atomic `round_NNN.pt` files containing round, model state, optimizer-state placeholder, global metrics, and timestamp. In the failure round, healthy updates remain valid because they began from the same restored checkpoint; only complete messages are eligible for FedAvg.

Supported injected modes are `none`, `process_crash`, `hang`, and `network_timeout`. `process_crash` is the default and genuinely terminates the OS process. Hang/timeout modes stop heartbeat communication and remain stuck until coordinator shutdown.

## Hyperparameter optimization

```powershell
python run_hpo.py
```

The default starts three actual FL trials in parallel. Each trial has three client processes, its own database, logs, and checkpoints under `results/trial_NNN/`. The search grid covers learning rates `0.01`, `0.001`, `0.0001`, batch sizes `32`, `64`, and local epochs `1`, `2`; the requested number of trials selects entries from that grid. Accuracy and duration are measured from real runs and summarized in `results/hpo.db`.

For a lighter check:

```powershell
python run_hpo.py --trials 3 --workers 1 --rounds 1 --max-train-samples 192
```

## Tests

```powershell
python -m pytest
```

The suite verifies model shape, deterministic non-IID skew, weighted averaging, rejection of incomplete updates, failed-client exclusion, scheduler priority, checkpoint restoration, heartbeat timeouts, and SQLite event/telemetry persistence.

## Persisted evidence

- `results/default.db`: events, telemetry, worker snapshots, round metrics, current state
- `checkpoints/round_NNN.pt`: latest valid global states
- `logs/coordinator.log`: orchestration and recovery decisions
- `logs/client_N.log`: exact completed local batches and the injected failure point

The key event types are `SYSTEM_STARTED`, `CLIENT_CONNECTED`, `CLIENT_HEARTBEAT`, `ROUND_STARTED`, `CLIENT_SELECTED`, `LOCAL_TRAINING_STARTED`, `BATCH_COMPLETED`, `LOCAL_TRAINING_COMPLETED`, `MODEL_UPDATE_RECEIVED`, `WORKER_FAILURE`, `CHECKPOINT_RESTORED`, `WORKLOAD_REASSIGNED`, `CHECKPOINT_SAVED`, `ROUND_COMPLETED`, and `TRAINING_COMPLETED`.

## Evaluation and limitations

Accuracy and loss are computed by evaluating the aggregated global model on a bounded MNIST test subset. Round duration, participation, update counts, detection latency, recovery time, and client utilization come from execution timestamps and messages. The small data bounds favor an observable laptop demo rather than maximum accuracy.

This MVP demonstrates data locality and crash recovery, but it is not a production privacy or security system. It does **not** implement secure aggregation, differential privacy, encrypted model updates, client authentication, Byzantine/adversarial-client defenses, WAN transport, durable queue delivery, or automatic worker replacement. A production evolution would add those controls, remote transport, replicated coordinator state, signed updates, and privacy accounting.
