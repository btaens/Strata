// src/core/arena_numa_test.cpp - the held arena's NUMA placement helpers (include/strata/core/arena_numa.hpp):
// the environment's number, the bind's accept/refuse, and the report's arithmetic. Pure + one real mmap: no GPU,
// no engine. A kernel that refuses set_mempolicy/move_pages (a seccomp sandbox) is a PASS with a note: the engine
// treats the same refusal as "not verified" and places the arena by the kernel's default - that behavior is what
// is pinned here, alongside the arithmetic on statuses fed by hand.
#include "strata/core/arena_numa.hpp"

#include <cstdio>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace {

int g_fail = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "arena_numa_test: FAIL: %s\n", what);
        ++g_fail;
    }
}

}  // namespace

int main() {
    // the environment's number: a plain 0..63, anything else off
    expect(strata::core::arena_numa_node_from_env("0") == 0, "0 is node 0, not unset");
    expect(strata::core::arena_numa_node_from_env("1") == 1, "1 is node 1");
    expect(strata::core::arena_numa_node_from_env("63") == 63, "63 is the top");
    expect(strata::core::arena_numa_node_from_env(nullptr) == -1, "unset: off");
    expect(strata::core::arena_numa_node_from_env("") == -1, "empty: off");
    expect(strata::core::arena_numa_node_from_env("auto") == -1, "auto is the server's, never the engine's");
    expect(strata::core::arena_numa_node_from_env("0,1") == -1, "a list is not one node");
    expect(strata::core::arena_numa_node_from_env("64") == -1, "64 is out of range");
    expect(strata::core::arena_numa_node_from_env("-1") == -1, "-1 is off, not a node");

    // the report's arithmetic, statuses fed by hand (no syscall, runs anywhere): node 0 gets pages 0 and 2
    {
        std::vector<int> st = {0, 1, 0, -1, 1};              // 2 of 4 answered on node 0
        const std::string s = strata::core::arena_numa_report_text(st, 0, 0, 48LL * 1024 * 1024 * 1024);
        expect(s.find("2 of 4 sampled pages on node 0") != std::string::npos, s.c_str());
        expect(s.find("50.0%") != std::string::npos, "the share prints");
        // the GiB figure is the SHARE applied to the arena, not the sample's own bytes: 50% of 48 GiB is 24 GiB,
        // and 96 sampled 4 KiB pages would be 0.0004 GiB, which says nothing to the person reading the line
        expect(s.find("24.00 of 48.00 GiB") != std::string::npos, s.c_str());
        expect(s.find("crosses the link") != std::string::npos, "a share under 100% names the far pages");
        std::vector<int> all = {0, 0, 0};
        expect(strata::core::arena_numa_report_text(all, 0, 0, 48LL * 1024 * 1024 * 1024).find("crosses the link") ==
                   std::string::npos,
               "all pages local: no far-pages tail");
        std::vector<int> none = {-1, -1};
        expect(strata::core::arena_numa_report_text(none, 0, 0) == "NUMA placement: nothing to sample",
               "no answered page, no errno: nothing to sample");
        expect(strata::core::arena_numa_report_text(none, 0, 1).find("not verified") != std::string::npos,
               "a refused move_pages reads as not verified, not as 0%");
    }

#if defined(__linux__)
    // the bind on this machine: accepted (node 0 exists here), or refused by the sandbox - the engine's own path
    // prints the reason and places by default, so a refusal is a pass with the reason shown
    const std::string b0 = strata::core::arena_numa_bind(0);
    if (b0.empty()) {
        std::printf("arena_numa_test: bound node 0\n");
        strata::core::arena_numa_reset();
    } else {
        std::printf("arena_numa_test: set_mempolicy refused (%s): the engine prints this and places by default\n",
                    b0.c_str());
    }
    expect(!strata::core::arena_numa_bind(-1).empty(), "-1: refused before any syscall");
    expect(!strata::core::arena_numa_bind(64).empty(), "64: refused before any syscall");

    // a real mapping: sample it (refused in a sandbox) and report - either outcome must read right
    constexpr size_t kPages = 64;
    void* m = mmap(nullptr, kPages * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(m != MAP_FAILED, "the test mapping");
    if (m != MAP_FAILED) {
        std::vector<void*> pages;
        for (size_t i = 0; i < kPages; ++i) {
            ((volatile char*) m)[i * 4096] = 1;             // fault it: now the kernel knows its node
            pages.push_back((char*) m + i * 4096);
        }
        std::vector<int> status;
        const int err = strata::core::arena_numa_sample(pages, status);
        const std::string line = strata::core::arena_numa_report_text(status, 0, err, 48LL * 1024 * 1024 * 1024);
        std::printf("arena_numa_test: %s\n", line.c_str());
        if (err != 0) {
            expect(line.find("not verified") != std::string::npos, "a refused sample reads as not verified");
        } else {
            long long answered = 0, on0 = 0;
            for (const int s : status) {
                if (s < 0) continue;
                ++answered;
                if (s == 0) ++on0;
            }
            expect(answered == (long long) kPages, "every faulted page of this process has a node");
            expect(line.find(std::to_string(on0) + " of " + std::to_string(answered)) != std::string::npos,
                   "the line counts what the sample found");
        }
        munmap(m, kPages * 4096);
    }
#endif

    if (g_fail) {
        std::fprintf(stderr, "arena_numa_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("arena_numa_test: OK\n");
    return 0;
}
