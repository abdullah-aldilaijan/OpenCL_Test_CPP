# OpenCL ZIP prototype (C++)

A native C++ port of the Python prototype [OpenCL_Test](https://github.com/abdullah-aldilaijan/OpenCL_Test). The original Python files are unchanged. This version needs no Python, numpy or pyopencl.

## Build

Requires Visual Studio 2022 or newer with the **Desktop development with C++** workload. Then run:

```bat
build.cmd
```

The build makes two programs:

- `build\oclzip-gui.exe` is the desktop window. It replaces `desktop_launcher.py`.
- `build\oclzip.exe` is the command-line tool. It replaces `gpu_zip.py`, `hardware.py` and the decompress side of `opencl_lz.py`, `opencl_compress.py` and `gpu_folder.py`.

Both are statically linked, so they run on other Windows PCs without the Visual C++ runtime. OpenCL is loaded at runtime from `OpenCL.dll`, which GPU drivers install. No OpenCL SDK is needed. If no OpenCL runtime is found, CPU mode still works.

## Use

`Run.cmd` opens the window. It builds first if needed. `Run.cmd <args>` passes the arguments to `oclzip.exe`:

```bat
oclzip devices
oclzip compress "C:\path\My Folder" "C:\elsewhere\My Folder.zip" --device NVIDIA
oclzip compress file.bin file.zip --device CPU
oclzip compress "C:\path\My Folder" "C:\elsewhere\all.zip" --device ALL
oclzip decompress archive.zip "C:\path\new-folder"
oclzip decompress old.folder.oclr "C:\path\new-folder" --device Intel
oclzip decompress old.oclr restored.bin --device NVIDIA
```

`--device` matches part of a GPU name, or takes `CPU`. If you leave it out, the best GPU is used (a discrete GPU first, then the one with the most memory).

`--device ALL`, or **"All — CPU + all GPUs together"** in the window, makes the CPU and every GPU compress the same archive at the same time. It applies to compression only. Restoring an old `.oclr` with `ALL` uses the single best device.

## What runs where

| Step | Device |
|---|---|
| DEFLATE match search and Huffman bit encoding | Selected GPU (optimised `deflate_blocks` kernel; same output as `deflate.cl` in the `OpenCL Test` folder) |
| Packing compressed chunks end to end before download | Selected GPU (`compact` kernel) |
| The same encoder when **CPU** is selected | C++ port of the original kernel, run on all CPU threads |
| File I/O, CRC-32, ZIP/ZIP64 headers | CPU |
| ZIP extraction (stored + DEFLATE, including dynamic Huffman ZIPs from Windows, 7-Zip and Python) | CPU (built-in inflater) |
| Legacy `.oclr` / `.folder.oclr` decoding | Selected device (OpenCL `lz_decode` / `rle_decode`), validated on CPU first |

**Change from the Python version:** in Python, CPU mode used zlib (dynamic Huffman, level 6). Here, CPU mode runs the same fixed-Huffman encoder as the GPU. So CPU and GPU ZIPs are byte-for-byte identical, but CPU-mode ZIPs may be somewhat larger than Python's zlib ZIPs were.

## Verification done

- ZIPs from the RTX 5060, the Iris Xe and CPU mode are byte-identical to each other and to the original `gpu_zip.py` output.
- Test data:
  - random, repeated and all-byte-value data
  - empty files and files of 1–4 bytes
  - Unicode (Arabic) names, nested and empty folders
  - 3,000 small files, and files of 235 MB and 244 MB
  - an edge-case set that hits every match length (3–258) and every distance code
- The optimised GPU kernel is checked against the CPU encoder, which still uses the original loop-based algorithm. That makes the CPU encoder an independent reference for the kernel's lookup tables.
- Python's `zipfile`/zlib and Windows `Expand-Archive` extract the ZIPs correctly.
- `oclzip decompress` restores ZIPs made by this tool, by Python (`ZIP_DEFLATED` level 9) and by Windows `Compress-Archive`, all identical to the source.
- Legacy `OCLLZ001`, `OCLRLE01` and `.folder.oclr` archives made by the Python scripts restore correctly on NVIDIA, Intel and CPU.
- These are refused, and leave no temporary folders behind: an existing output, an archive saved inside its source folder, `../` paths, reserved Windows names (`CON` etc.), and corrupted DEFLATE data.

Files larger than 4 GiB have not been tested yet. ZIP64 headers are always written.

## How compression runs in parallel

Compression is a three-stage pipeline. Three batches circulate between the stages, so all three run at the same time:

1. **Reader thread.** It fills batches of 32 KiB chunk slots with file data. Chunks from **many files are packed into the same batch**: 1,024 chunks (32 MiB) on the GPU, 256 on the CPU. Each file still starts at a fresh chunk, so the ZIP bytes don't change.
2. **Encoder thread.** It uploads the batch and runs the kernel with **one work-item per chunk**. A second kernel then packs the compressed chunks end to end, so only the real bytes are downloaded rather than a 64 KiB slot per chunk.
3. **Main thread.** It writes the ZIP headers and compressed chunks in order, then patches each file's CRC and sizes.

**Reading in parallel.** Every file, large or small, is split into pieces of up to 8 MiB:
- 8 read threads read and checksum pieces at the same time, up to 512 MiB ahead. The disk gets many requests at once, and the CRC work is spread over several cores.
- The reader takes the pieces strictly in archive order and only copies them into batches.
- Each file's CRC-32 is assembled from its piece CRCs with zlib's `crc32_combine` method.
- Pieces are multiples of 32 KiB, so the chunks, and therefore the ZIP bytes, are unchanged.

**Small files.** For a file that fits entirely in one batch (most files), the writer already knows the CRC and sizes. It writes the complete header straight away, instead of seeking back afterwards to patch it.

Smaller batches turned out faster than large ones. With large batches, a set of small files fits into a single batch, so the read, compress and write stages can't overlap.

**Safeguard against dropped launches.** The Intel driver once silently dropped a very long kernel launch and left stale data in the buffers. So every launch is checked:
- Before each run, the size slots are filled with a marker value.
- A slot the kernel did not overwrite shows that the launch was dropped.
- Dropped chunks are retried once with smaller launches. If that still fails, the rest of the archive is encoded on the CPU, with the same output bytes.

An archive can never be silently corrupt.

## Using all devices at once (`ALL`)

How it works:
- **One encoder thread per device.** The CPU and every usable GPU each get their own thread.
- **Shared batches.** The reader still packs files into 512-chunk batches. Each device takes the next batch as soon as it finishes its previous one, so faster devices automatically do more of the work.
- **Same output.** Batches are numbered, and the writer puts them back in order before writing. The ZIP is byte-identical to single-device mode no matter which device compressed which batch.
- **Fewer CPU threads.** The CPU encoder leaves a few threads free for reading, writing and feeding the GPUs: 16 of 20 on this machine.
- **Work split report.** At the end, each device's share of the work is shown:

```
Elapsed after setup: 0.83s. Work split:
  NVIDIA GeForce RTX 5060 Laptop GPU     19 batches,  24.9% of chunks
  Intel(R) Iris(R) Xe Graphics           12 batches,  16.4% of chunks
  CPU (16 threads)                       43 batches,  58.7% of chunks
```

Testing: 27 `ALL` runs, including 16-chunk batches that force heavy out-of-order completion. Every ZIP was byte-identical to the single-device reference, and Python's `zipfile` validated them.

## Where the time goes

Every compression run ends with a timing report for the stages. Because each stage has its own thread, the times overlap:

```
Reader: working 0.21s, waiting for disk 0.81s, waiting for free buffers 0.00s
Writer: writing 0.04s, waiting for compressed data 0.95s
Limited by: reading the source files from disk (the devices were waiting for data)
```

How to read it:
- **"Waiting for disk"** is high: the source drive is the limit. More GPUs won't help. Try faster storage, or write the ZIP to a different drive than the one you are reading from.
- **"Waiting for free buffers"** is high: the encoders, or the writer, are the limit.
- In `ALL` mode, devices that are "busy" for much less time than the elapsed time spent the rest of it waiting for data.

## GPU kernel optimisations

The optimised `deflate_blocks` produces exactly the same bits as the original kernel. It changes how they are produced:

| Original | Optimised |
|---|---|
| Zeroes the whole 64 KiB output slot first | No zeroing needed |
| Writes every single bit with a global-memory read-modify-write | Collects bits in a 64-bit register and stores 32 bits at a time |
| Loops to find Huffman, length and distance codes | Uses lookup tables (generated on the host and prepended to the kernel as `__constant` arrays) |
| 32-bit hash table, filled one entry at a time | 16-bit hash table, filled with 16-byte vector stores |
| Compares matches one byte at a time | Compares matches 4 bytes at a time (`vload4`) |

GPU busy time (upload + compress + download), same pipeline, best of 3:

| Test | Original kernel | Optimised kernel |
|---|---|---|
| 235 MB file, RTX 5060 | 0.48 s | **0.22 s** |
| 235 MB file, Iris Xe | 1.88 s | **0.65 s** |
| 3,000 small files, RTX 5060 | 0.30 s | **0.14 s** |
| 3,000 small files, Iris Xe | 1.07 s | **0.42 s** |

Optional tuning through environment variables:
- `OCLZIP_BATCH_CHUNKS`: chunks per batch (default 1024 on the GPU, 256 on the CPU, 512 for `ALL`)
- `OCLZIP_LAUNCH_CHUNKS`: chunks per kernel launch (default 4096 on discrete GPUs, 512 on integrated)
- `OCLZIP_READ_THREADS`: number of parallel read threads (default 8)
- `OCLZIP_KERNEL=original`: use the original kernel, for comparison

## Performance

All runs used an i9-13900H with an RTX 5060 Laptop GPU and Iris Xe. Times are total seconds, including OpenCL start-up.

| Test | First C++ port (one file at a time) | Now |
|---|---|---|
| 3,000 small files (92 MB), RTX 5060 | 12.0 s | **0.55 s** |
| 3,000 small files, Iris Xe | – | **0.67 s** |
| 235 MB file, RTX 5060 | 2.2 s | **0.49 s** |
| 235 MB file, Iris Xe | – | **0.83 s** |
| 3,000 small files, CPU (20 threads) | 1.3 s | **0.18 s** |
| 235 MB file, CPU (20 threads) | 0.58 s | **0.17 s** |

`ALL` compared with single devices (best of 3, total seconds):

| Test | CPU | RTX 5060 | Iris Xe | **ALL** |
|---|---|---|---|---|
| 3,000 small files (92 MB) | 0.18 | 0.57 | 0.66 | 0.48 |
| 235 MB file | 0.16 | 0.47 | 0.84 | 0.43 |
| 1.2 GB file | 0.72 | 1.30 | 3.41 | 1.01 |

With the parallel reader (best of 2, total seconds; "cold" means the files were dropped from the Windows cache first):

| Test | CPU | RTX 5060 | **ALL** |
|---|---|---|---|
| bench folder (3,014 files, 400 MB), warm | 0.21 | 0.76 | 0.52 |
| bench folder, cold | 0.30 | 0.77 | 0.64 |
| 1.2 GB file, warm | 0.46 | 1.36 | 0.75 |
| 1.2 GB file, cold | 0.63 | 1.40 | 0.88 |

`ALL` is faster than either GPU alone, but **slower than CPU-only on this laptop**. Three reasons:
- Starting OpenCL on two GPUs adds about 0.2 s.
- The Iris Xe shares the CPU's power budget and memory, so it slows the CPU down.
- Every GPU batch has to be copied to the GPU and back, while the CPU alone already uses most of the memory bandwidth.

`ALL` should help more on a desktop with a strong discrete GPU and fewer CPU cores, or with a heavier compressor.

**The CPU is still faster on this machine.** Two reasons:
- About 0.19 s of every GPU run is OpenCL start-up: loading the GPU drivers and compiling the kernel.
- Fixed-Huffman DEFLATE with this simple match finder is very cheap work: 20 CPU threads can compress the 235 MB file in about 0.05 s. That leaves little for the GPU to win back after copying the data to it and back.

The GPU would pay off with a heavier compressor, such as dynamic Huffman or deeper match search, where the work per byte is much higher.

## Files

- `src/ocl.*` – loads OpenCL dynamically; handles devices, context, program, buffers
- `src/kernels.h` – embedded OpenCL kernels
- `src/zip_create.cpp` – ZIP/ZIP64 writer with GPU and CPU encoders (`gpu_zip.py`)
- `src/zip_restore.cpp` – safe ZIP extraction and the inflater (`folder_archive.restore`)
- `src/legacy.cpp` – `.oclr` decoders (`opencl_lz.py`, `opencl_compress.py`, `gpu_folder.py`)
- `src/hardware.cpp` – device discovery (`hardware.py`)
- `src/cli.cpp`, `src/gui.cpp` – command line and Win32 window
