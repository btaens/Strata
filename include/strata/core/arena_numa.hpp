// include/strata/core/arena_numa.hpp - which CPU's memory the held expert arena's pages sit in (opt-in;
// docs/KEEP_RAM.md, "two sockets").
//
// On a two-socket PC the kernel spreads first touches over both sockets' memory. The process that holds the experts
// in RAM (--hold-expert-arena, serve/arena_keep.py) is the one that touches every page of the arena first, and a
// tmpfs page takes the memory policy of the thread that faults it: set this thread's policy to one node before the
// arena is read and the whole arena lands on that node, so the card wired to that CPU DMAs it without crossing the
// link between the processors. set_mempolicy needs no privilege (it steers this process's own faults), and the read
// threads the fill starts inherit the policy at creation. The serving engine is deliberately NOT given the policy:
// attaching maps pages that already exist, and a policy only steers NEW faults - registering does not move a page.
//
// Unset: none of this runs and the arena is placed by the kernel exactly as before.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core {

/// STRATA_ARENA_NUMA_NODE as a node number, or -1 when the name is unset, empty, not a plain number, or out of
/// 0..63. The server resolves "auto" (the card's node) to a number before it starts the holder, so the engine only
/// ever sees a number here.
inline int arena_numa_node_from_env(const char* value) {
    if (value == nullptr || *value == '\0') return -1;
    char* end = nullptr;
    const long v = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || v < 0 || v > 63) return -1;
    return (int) v;
}

#if defined(__linux__)
/// Ask this thread - and the read threads it starts - to fault the arena's pages on `node`. MPOL_PREFERRED is 1,
/// the value in mempolicy.h spelled out as ds4_os.hpp spells its modes, and the kernel wants maxnode strictly
/// greater than node. PREFERRED, not BIND, on purpose: BIND makes the node strict, and a node that cannot hold a
/// 48 GiB tmpfs arena fails the faulting read with SIGBUS - a holder dying mid-read is the one failure mode the
/// option's "never fails a start" rule forbids. PREFERRED puts every page on the node while the node has the room,
/// and falls back to the faulting CPU's node when it does not; the sample below then reports which happened.
/// "" when the kernel accepted it, the reason when not.
inline std::string arena_numa_bind(int node) {
    if (node < 0 || node > 63) return "the node number is out of 0..63";
    unsigned long mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    mask[(size_t) node / 64] = 1UL << (node % 64);
    if (syscall(SYS_set_mempolicy, 1 /* MPOL_PREFERRED */, mask, (unsigned long) (node + 1), 0) != 0)
        return std::string("set_mempolicy: ") + std::strerror(errno);
    return {};
}

/// Back to the kernel's default placement (a test's cleanup; the engine process just ends).
inline void arena_numa_reset() {
    (void) syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, 1UL, 0);
}

/// Ask the kernel where each page of `pages` sits (pid 0: this process), in batches of 4096; a page the kernel
/// will not answer keeps -1. 0 when every batch answered, else the first errno - the way expert_source.cpp's
/// NUMA report handles it: a refused syscall (a seccomp sandbox, a hardened kernel) is "not verified", never a
/// failed start.
inline int arena_numa_sample(const std::vector<void*>& pages, std::vector<int>& status) {
    status.assign(pages.size(), -1);
    int sys_errno = 0;
    for (size_t b = 0; b < pages.size(); b += 4096) {
        const size_t cnt = pages.size() - b < 4096 ? pages.size() - b : 4096;
        if (syscall(SYS_move_pages, 0, (unsigned long) cnt, const_cast<void**>(pages.data() + b), nullptr,
                    status.data() + b, 0) != 0) {
            if (sys_errno == 0) sys_errno = errno;
            for (size_t i = b; i < b + cnt; ++i) status[i] = -1;
        }
    }
    return sys_errno;
}

/// The line from the statuses (pure, so the arithmetic has a test even where move_pages is refused). The sample says
/// the SHARE that landed right; `arena_bytes` turns that share into the GiB the card actually reaches locally - the
/// sample is ~96 pages of a tens-of-GiB arena, so its own page count would mean nothing. The bind states the intent,
/// this states the fact: a kernel with NUMA balancing on can still move pages later, and the caller reads it after
/// the whole arena is resident.
inline std::string arena_numa_report_text(const std::vector<int>& status, int node, int sys_errno,
                                          long long arena_bytes = 0) {
    long long on = 0, total = 0;
    for (const int s : status) {
        if (s < 0) continue;
        ++total;
        if (s == node) ++on;
    }
    if (total == 0)
        return sys_errno != 0 ? "NUMA placement: not verified (move_pages failed: " +
                                    std::string(std::strerror(sys_errno)) + ")"
                              : "NUMA placement: nothing to sample";
    const double share = (double) on / (double) total;
    char buf[224];
    std::snprintf(buf, sizeof buf,
                  "NUMA placement: %lld of %lld sampled pages on node %d (%.1f%%; ~%.2f of %.2f GiB beside that "
                  "CPU%s)", (long long) on, (long long) total, node, 100.0 * share,
                  (double) arena_bytes * share / (1024.0 * 1024.0 * 1024.0),
                  (double) arena_bytes / (1024.0 * 1024.0 * 1024.0),
                  on == total ? "" : " - the rest sits by the other CPU and crosses the link to the card");
    return buf;
}
#endif

}  // namespace strata::core
