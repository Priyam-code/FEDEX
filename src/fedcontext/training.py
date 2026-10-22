"""Centralized training and single-process, sample-weighted FedAvg simulation."""
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from .models import ThreeStream
from .utils import dump_json, seed_everything

KEYS = ["movement", "target", "timing", "context"]


def loader(data, ids, batch_size, shuffle=False):
    ds = TensorDataset(*(torch.from_numpy(data[k][ids]) for k in KEYS))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.inference_mode()
def score(model, data, ids=None, batch_size=512):
    model.eval()
    if ids is None:
        ids = np.arange(len(data["target"]))
    values = [model.errors(*batch).numpy() for batch in loader(data, ids, batch_size)]
    return np.concatenate(values) if values else np.empty((0, 3), dtype=np.float32)


def local_train(model, data, ids, cfg, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    result = []
    for _ in range(epochs):
        model.train()
        total = np.zeros(3)
        n = 0
        for batch in loader(data, ids, cfg["batch_size"], True):
            opt.zero_grad(set_to_none=True)
            errors = model.errors(*batch)
            loss = errors.mean(0).sum()
            if not torch.isfinite(loss):
                raise ValueError("Training diverged; inspect invalid units/outliers in input")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            total += errors.detach().sum(0).numpy()
            n += len(errors)
        result.append(total/n)
    return np.mean(result, axis=0)


def fedavg(states, counts):
    if not states or len(states) != len(counts) or min(counts) <= 0:
        raise ValueError("FedAvg requires nonempty client updates and positive sample counts")
    total = sum(counts)
    # Double-precision accumulation, then cast back to float32.
    return {k: sum(s[k].double()*(n/total) for s, n in zip(states, counts)).to(states[0][k].dtype)
            for k in states[0]}


def fit(data, meta, cfg, mode, seed, out):
    seed_everything(seed, cfg["threads"])
    model = ThreeStream()
    train = np.flatnonzero(meta.partition.eq("train"))
    select = np.flatnonzero(meta.partition.eq("selection"))
    if len(train) == 0 or len(select) == 0:
        raise ValueError("Need nonempty train and selection partitions. Add more receiver files.")
    if meta.iloc[train].label.ne(0).any():
        raise ValueError("Nonbenign training labels")
    groups = {int(c): train[meta.iloc[train].client.to_numpy()==c] for c in sorted(meta.iloc[train].client.unique())}
    if mode == "federated" and len(groups) < 2:
        raise ValueError("Federated experiment requires >=2 nonempty clients. Check client partition.")
    best, state, stale = float("inf"), None, 0
    history, communication = [], []
    param_bytes = sum(v.numel()*v.element_size() for v in model.state_dict().values())
    start = time.perf_counter()
    # Centralized optimizer persists across epochs; client Adam is restarted each round.
    central_opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    steps = cfg["rounds"] if mode == "federated" else cfg["epochs"]
    for step in range(1, steps+1):
        if mode == "federated":
            updates, sizes, losses = [], [], []
            for client, ids in groups.items():
                client_model = copy.deepcopy(model)
                losses.append(local_train(client_model, data, ids, cfg, cfg["local_epochs"]))
                sizes.append(len(ids))
                updates.append(copy.deepcopy(client_model.state_dict()))
            model.load_state_dict(fedavg(updates, sizes))
            train_loss = np.average(losses, axis=0, weights=sizes)
            communication.append(dict(round=step, participating_clients=len(groups),
                upload_bytes=param_bytes*len(groups), download_bytes=param_bytes*len(groups)))
        else:
            model.train()
            total = np.zeros(3)
            for batch in loader(data, train, cfg["batch_size"], True):
                central_opt.zero_grad(set_to_none=True)
                errors = model.errors(*batch)
                loss = errors.mean(0).sum()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                central_opt.step()
                total += errors.detach().sum(0).numpy()
            train_loss = total/len(train)
        val_loss = score(model, data, select).mean(0)
        value = float(val_loss.sum())
        history.append(dict(step=step, train_m=float(train_loss[0]), train_t=float(train_loss[1]), train_c=float(train_loss[2]),
                            selection_m=float(val_loss[0]), selection_t=float(val_loss[1]), selection_c=float(val_loss[2])))
        print(f"{mode} seed={seed} step={step}/{steps} selection_loss={value:.6f}", flush=True)
        if value < best-cfg["min_delta"]:
            best, state, best_step, stale = value, copy.deepcopy(model.state_dict()), step, 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    if state is None:
        raise ValueError("No finite model selected")
    model.load_state_dict(state)
    torch.save(model.state_dict(), out/"model.pt")
    pd.DataFrame(history).to_csv(out/"learning_curve.csv", index=False)
    pd.DataFrame(communication, columns=["round", "participating_clients", "upload_bytes", "download_bytes"]).to_csv(out/"communication.csv", index=False)
    stats = dict(mode=mode, seed=seed, best_step=best_step, executed_steps=len(history),
                 seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()),
                 parameter_bytes=param_bytes, artifact_bytes=(out/"model.pt").stat().st_size,
                 active_clients=len(groups), train_windows=len(train), selection_windows=len(select),
                 communication_bytes=sum(x["upload_bytes"]+x["download_bytes"] for x in communication),
                 communication_scope="float32 model payload only; excludes transport, setup, validation/calibration metadata and OBU distribution",
                 privacy="Simulation only. Plain FedAvg; no secure aggregation, differential privacy, networking or poisoning protection.")
    dump_json(out/"training.json", stats)
    return model, stats
