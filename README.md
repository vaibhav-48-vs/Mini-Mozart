# Mini-Mozart

**Mini-Mozart** is a deep-learning project that trains a multi-layer LSTM on classical MIDI files and uses it to generate new melodic sequences. The entire pipeline — data loading, model definition, training, and generation — lives in a single Jupyter notebook (`midi melody gen base model.ipynb`).

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [1. MIDI → Piano Roll Conversion](#1-midi--piano-roll-conversion)
  - [2. Dataset Construction](#2-dataset-construction)
  - [3. Model Architecture](#3-model-architecture)
  - [4. Training](#4-training)
  - [5. Music Generation](#5-music-generation)
  - [6. Piano Roll → MIDI Export](#6-piano-roll--midi-export)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)

---

## Overview

The project converts classical MIDI compositions into binary **piano rolls** (2-D arrays of shape `[time, 128]`), trains an LSTM to predict the next time-step of notes given a context window, and then samples new time-steps autoregressively to create a novel MIDI melody.

---

## How It Works

### 1. MIDI → Piano Roll Conversion

```python
midi_to_pianoroll(file_path, fs=16)
```

Each MIDI file is loaded with [`pretty_midi`](https://github.com/craffel/pretty-midi) and converted to a **binary piano roll** at a frame rate of `fs = 16` frames per second.

- Shape: `[time_steps, 128]`  — one column per MIDI pitch (0–127).
- Values: `1.0` if a note is active at that time step, `0.0` otherwise.

Corrupt or empty files are gracefully skipped by `load_valid_pianorolls`.

---

### 2. Dataset Construction

```python
class MIDIDataset(Dataset):
    def __init__(self, pieces, seq_len=64): ...
```

A PyTorch `Dataset` creates overlapping windows of length `seq_len = 64` frames from every piece using a **sliding window** approach:

| Item | Content |
|------|---------|
| `x` | frames `[t, t+seq_len)` — the input context |
| `y` | frames `[t+1, t+seq_len+1)` — the next-step targets |

A single dataset over 93 pieces yields **~449 k** training sequences.  
`DataLoader` shuffles them with `batch_size = 64` and `pin_memory = True` (CUDA).

---

### 3. Model Architecture

```python
class MusicLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256,
                 num_layers=2, dropout=0.2): ...
```

| Component | Details |
|-----------|---------|
| Input | Binary piano roll frame — 128-dimensional vector |
| LSTM | 2 stacked layers, hidden size 256, dropout 0.3 between layers |
| Output head | `nn.Linear(256 → 128)` producing raw logits |

The model returns `(logits, hidden_state)`, enabling stateful autoregressive generation.

---

### 4. Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 30 |
| Batch size | 64 |
| Sequence length | 64 |
| Optimizer | Adam, lr = 1e-3 |
| Loss | `BCEWithLogitsLoss` (binary cross-entropy on logits) |
| Gradient clipping | 1.0 |
| Device | CUDA if available, else CPU |

Each epoch iterates over all shuffled batches, computes the loss, back-propagates, clips gradients, and updates weights.  
Trained weights are saved to `music_lstm.pt`.

---

### 5. Music Generation

```python
generate_music(model, seed_seq, steps=512, temperature=1.0)
```

Generation is fully **autoregressive**:

1. A seed sequence of `seq_len` frames is taken from a real piece.
2. The model is run on the seed to warm up the LSTM hidden state.
3. At each subsequent step, the model predicts logits for the next frame.
4. Logits are divided by `temperature`, passed through `sigmoid`, and a binary frame is sampled via `torch.bernoulli`.
5. The new frame becomes the sole input for the next step (`inp` shape: `[1, 1, 128]`).
6. After `steps = 512` iterations the full generated piano roll is returned.

**Temperature** controls randomness:
- Low (< 1.0) → more conservative / repetitive output.
- High (> 1.0) → more chaotic / exploratory output.

---

### 6. Piano Roll → MIDI Export

```python
pianoroll_to_pretty_midi(piano_roll, fs=16, program=0, min_note_len=1)
```

The generated binary piano roll is converted back to a standard MIDI file:

- Each active "run" of `1`s along a pitch row becomes a single `pretty_midi.Note`.
- Notes shorter than `min_note_len` frames are discarded to suppress noise.
- The result is written to `out.mid`.

---

## Project Structure

```
Mini-Mozart/
├── midi melody gen base model.ipynb   # Full pipeline notebook
├── midi_data/                         # (Optional) place MIDI training files here
└── README.md
```

---

## Requirements

```
python >= 3.9
torch
pretty_midi
numpy
```

Install dependencies:

```bash
pip install torch pretty_midi numpy
```

GPU support is strongly recommended for training (CUDA).

---

## Usage

1. **Prepare data** – place your `.mid` files in a folder and update `fpath` in the notebook.
2. **Open the notebook** – `jupyter notebook "midi melody gen base model.ipynb"`.
3. **Run all cells** – the notebook will:
   - Load and convert MIDI files to piano rolls.
   - Build the `MIDIDataset` and `DataLoader`.
   - Instantiate and train `MusicLSTM` for 30 epochs.
   - Generate a 512-step melody seeded from the first piece.
   - Save the output to `out.mid` and the model weights to `music_lstm.pt`.

---

## Configuration

Key variables near the top of the training cells that you can adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `fs` | `16` | Piano-roll frame rate (frames per second) |
| `seq_len` | `64` | Context window length for training / seeding |
| `batch_size` | `64` | Mini-batch size |
| `hidden_size` | `256` | LSTM hidden dimension |
| `num_layers` | `2` | Number of LSTM layers |
| `dropout` | `0.3` | Dropout between LSTM layers |
| `epochs` | `30` | Training epochs |
| `lr` | `1e-3` | Adam learning rate |
| `grad_clip` | `1.0` | Gradient-norm clipping threshold |
| `steps` | `512` | Generation steps (output frames) |
| `temperature` | `1.0` | Sampling temperature |