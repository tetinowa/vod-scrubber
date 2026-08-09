# highlight-panel

Premiere Pro CEP extension that scans a downloaded Twitch VOD, scores clip-worthy
moments, and drops markers onto the active sequence timeline. Does not cut, export,
or edit — it just tells you where to look.

## engine/ — standalone analysis CLI

```
py -3.12 -m venv engine\.venv
engine\.venv\Scripts\python.exe -m pip install -r engine\requirements.txt
engine\.venv\Scripts\python.exe engine\analyze.py <vod.mp4> --out candidates.json
```

Fuses audio-energy (librosa RMS) and speech signals (faster-whisper large-v3,
CUDA) over a 5s/2s rolling window into one score per candidate, with non-max
suppression and configurable pad. Weights live as constants at the top of
`analyze.py`. Chat is stubbed at weight 0 pending a future chat-log signal.

### Setup gotchas

- **`python`/`python3` may not be on PATH** even if Python is installed — Windows'
  Store alias stub shadows it. Use the `py` launcher (`py -0p` lists installed
  versions) or the interpreter's full path instead.
- **`ffmpeg` must be on PATH.** `analyze.py` checks and exits with a clear error
  if it isn't; install it (e.g. `winget install Gyan.FFmpeg`) and restart your
  shell, or add it to PATH manually.
- **faster-whisper on CUDA needs `cublas64_12.dll` / cuDNN DLLs that ctranslate2
  does not bundle on Windows.** `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
  (already in `requirements.txt`) puts them in the venv's `site-packages\nvidia\*\bin`,
  but that directory is not on PATH by default — add it before running:
  ```
  $env:PATH += ";<path-to-venv>\Lib\site-packages\nvidia\cublas\bin;<path-to-venv>\Lib\site-packages\nvidia\cudnn\bin"
  ```
  Without this, transcription fails at model inference with
  `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded` —
  it gets past model load and download fine, so the failure only shows up once
  transcription actually starts.
