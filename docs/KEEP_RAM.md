# Keep the experts in RAM (opt-in)

Unloading a model gives the graphics card to a game or another server. The next request starts a new engine, and
that engine reads the experts off the disk again. This option keeps the experts in system RAM while the card is
free, so a reload skips that read. What goes back is the graphics card; what stays is the memory.

**Measured on:** two RTX PRO 4500 cards, engine 0.1.36, the expert load of a ~40 GB pack read cold
([bench results](../bench/results/2026-10-02-community-2x-rtx-pro-4500-engine-0.1.36/kaltstart/kalt-out.txt)), and
a Ryzen 7 9800X3D with an RTX 5070 Ti, same kind of load
([DETAILS.md](DETAILS.md#running-it-at-startup-task-scheduler)):

| The expert read | |
| --- | --- |
| Cold, from the NVMe | 6.40 GiB/s — 19-20 s for 39.97 GiB |
| Ordinary start on a 9800X3D | 1.42-1.52 GiB/s — about 35 s |

The engine already has one protection: when the RAM can hold the experts, it reads them buffered on purpose, so a
restart often finds them in the OS file cache and loads in seconds
([DETAILS.md](DETAILS.md#sharing-the-gpu-with-other-programs-optional)). That cache is what the PC is willing to
give back: another program's big file read can push it out. This option is for the case where the cache cannot be
counted on and the RAM can. **The end-to-end saving on your PC is not measured here** - it is the difference
between the table above and whatever your file cache already gave you. Time the first answer after an unload with
the option on and off to see what it is worth where you run it.

## Turning it on

One key in `strata-<model>.json`, the Model settings card, or the server's command line:

```json
"keep_experts_in_ram": true
```

```sh
python serve/server.py --config strata-iq3_s.json --keep-experts-in-ram     # or STRATA_KEEP_EXPERTS_IN_RAM=1
```

It pairs with `"idle_unload_s"` and `POST /unload`
([sharing the GPU](DETAILS.md#sharing-the-gpu-with-other-programs-optional)): the idle unload frees the card on
the server's terms, and the reload that follows keeps the experts where they are.

At the server's start it prints what it decided; the last line below is the unload:

```
[strata] the experts stay in RAM between unloads (50.3 GiB, one memory object of this server, not limited by the size of /dev/shm): a reload skips the read
[strata] the experts are held in system RAM (54009213747 bytes); an unloaded model loads without reading them again
[strata] model unloaded after 600 s idle; the experts stay in system RAM, the next request starts the engine without reading them
```

The read happens at the server's start, before it answers requests, and it costs what the table above shows -
also when `"lazy_load"` is on, because that read is the point of the option. Closing the server gives the memory
back.

## How it works

One ordinary engine process is started with `--hold-expert-arena` beside `--shared-expert-arena FILE`. It reads
the experts into that one memory object (a `memfd` of the server on Linux, a named section on Windows - the same
kind of object [split serving](SPLIT_SERVE.md) shares), marks the arena's header filled, prints `ARENA READY`,
and then holds nothing but the mapping: no GPU context, no weights, no byte of graphics memory. The holder starts
before any device query, which is what keeps that promise. The server waits for the line before it lets a request
start an engine.

A serving engine that starts later opens the same object. The header's state word says it is filled, so the engine
skips the read - the same protocol the second engine of split serving already uses
([expert_source.cpp](../src/core/expert_source.cpp)). It still registers the pages with its own graphics driver,
which is the part that belongs to the process holding a card.

If the holder dies - the OOM killer, a crash - the memory goes back with it; the server starts it again and says
so, at most once every 20 seconds, and gives up after three failed starts in a row.

## What it refuses

The option never forces anything and never fails a start; it prints one line and unloads as before. It refuses at
the server's start, where a refusal costs nothing, and never at an unload:

| Situation | The line says |
| --- | --- |
| `--mmap-experts`, `--resident-experts`, `--resident-budget-gib` | the model has no resident expert arena to keep |
| DeepSeek-V4.1 | the DeepSeek engine has no Qwen expert arena to keep |
| Intel (SYCL) or Mac (Metal) build | that build has no arena-holding mode |
| Free RAM below the experts plus 12 GiB | `the experts are 50.3 GiB and this PC has 30.0 GiB free; the arena plus 12 GiB for the OS and the engine needs 62.3 GiB` |
| No room on `/dev/shm` on a kernel without `memfd` | how much the mount has and how much the experts are |
| A pack whose experts' size is not known here | says so; the engine's own reserve check still applies |

The 12 GiB is the operating system, the server and the engine's own RAM kept above the arena. The memory needed
per quant is the measured figure in [strata_control/replica_profile.py](../strata_control/replica_profile.py):
50.3 GiB IQ3_S, 42.9 GiB IQ3_XXS, 34.0 GiB Q2_0.

## What still costs time on every reload

The experts are the biggest part of the read, not the whole start. A reload still makes its CUDA context, reads
the dense weights and the PLE table, and registers the arena's pages with the graphics driver. The engine's own
note in [expert_source.cpp](../src/core/expert_source.cpp) puts whole-arena registration at about 2.5 s for
33 GiB of 4 KiB pages on Windows (the code's number, marked as read from code, not measured here). The K/V, the
drafter and the session are re-made as before too.

Two engine settings interact with the shared arena, read from the code and not run here: `STRATA_PIN_BY_NEED`
is inert when a shared arena is in use (one layout has to serve every process, the packed one), and the holder
itself neither registers nor mlocks the arena (the driver alias belongs to the process with the card; an mlock of
tens of GiB would fail the default `RLIMIT_MEMLOCK`). Both are default-off paths, so nothing changes for a normal
setup.

## Limits

- Linux and Windows. Intel (SYCL) and Mac (Metal) builds are refused (read from the code, not run: the SYCL tree
  lacks the no-pin form the holder needs, and the Metal engine has no shared arena at all).
- Qwen packs with a resident arena. Not the mapped or budgeted RAM modes, not DeepSeek.
- It is not `--vram-elastic`: that shrinks an expert cache while the model keeps serving
  ([VRAM elasticity](VRAM_ELASTIC.md)); this ends the engine and keeps its experts.
- One model per server, as always: the arena header carries a hash of the pack, and an engine refuses an arena
  written for different pack bytes (`shared arena ... was written for pack hash ...`). Replacing the pack's files
  under a running server hits this; restarting the server makes a fresh arena.
- The memory is not free: one copy of the pages, shared by the holder and any serving engine, and it stays until
  the option is off and the server restarts. A per-process monitor shows it in the holder, not twice.
