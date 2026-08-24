from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


backend_path = Path("ggml/src/ggml-backend.cpp")
backend = backend_path.read_text(encoding="utf-8")

backend = replace_once(
    backend,
    """        // an async CPU split may still be computing; join before anything that
// depends on it: another CPU split, or a split reading a CPU tensor.
// When the immediately preceding CPU split is independent, preserve the
// intended cold-CPU/hot-GPU overlap and do not apply the generic
// zero-input backend-transition synchronization below.
bool cpu_async_overlap = false;
if (sched->cpu_async && sched->cpu_async->pending) {
    bool must_join = split_backend_id == sched->n_backends - 1;
    for (int input_id = 0; !must_join && input_id < split->n_inputs; input_id++) {
        ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
        must_join = input_backend == sched->backends[sched->n_backends - 1];
    }
    if (must_join) {
        enum ggml_status ec = sched->cpu_async->join();
        if (ec != GGML_STATUS_SUCCESS) {
            return ec;
        }
    } else {
        cpu_async_overlap = prev_backend_id == sched->n_backends - 1;
    }
}

// ensure the previous split's async work has completed before we start
// this split, the allocator may have reused buffer regions across splits
if (!cpu_async_overlap && split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
    if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
    } else {
        ggml_backend_synchronize(sched->backends[prev_backend_id]);
    }
}
""",
    """        // An async CPU split may still be computing. Join before another CPU
        // split or any split that reads a CPU-resident input.
        if (sched->cpu_async && sched->cpu_async->pending) {
            bool must_join = split_backend_id == sched->n_backends - 1;
            for (int input_id = 0; !must_join && input_id < split->n_inputs; input_id++) {
                ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
                must_join = input_backend == sched->backends[sched->n_backends - 1];
            }
            if (must_join) {
                enum ggml_status ec = sched->cpu_async->join();
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
            }
        }

        // Ensure the previous main-thread backend's asynchronous work has
        // completed before a zero-input transition. An asynchronously launched
        // CPU split deliberately does not replace prev_backend_id, allowing the
        // following independent hot GPU split to overlap it.
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }
""",
    "async CPU/current scheduler reconciliation",
)

backend = replace_once(
    backend,
    """                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
""",
    """                    int id = 0;
                    while (id < n_expert && !ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    if (id == n_expert) {
                        // Every routed slot belongs to the opposite hot/cold pack. The
                        // MUL_MAT_ID skip path emits zeros and does not read this tensor.
                        continue;
                    }
                    int32_t first_id = id;
""",
    "all-skipped selective expert copy guard",
)

backend = replace_once(
    backend,
    """                sched->cpu_async->launch(split_backend, &split->graph);
                continue;
""",
    """                sched->cpu_async->launch(split_backend, &split->graph);
                // Keep prev_backend_id at the last main-thread backend. The next
                // independent GPU split can then launch without serializing on the
                // CPU worker; any actual CPU dependency is joined above.
                continue;
""",
    "async CPU launch comment",
)

backend = backend.replace(
    "// async execution of CPU splits (GGML_SCHED_ASYNC_CPU): a persistent worker",
    "// Async execution of CPU splits: a persistent worker",
)
backend = backend.replace(
    "// async CPU split execution (GGML_SCHED_ASYNC_CPU); NULL when disabled",
    "// Async CPU split execution; NULL when disabled.",
)
backend = backend.replace("\n\n\n    ggml_backend_sched_reset(sched);", "\n\n    ggml_backend_sched_reset(sched);")
backend_path.write_text(backend, encoding="utf-8")

graph_path = Path("src/llama-graph.cpp")
graph = graph_path.read_text(encoding="utf-8")
graph = replace_once(
    graph,
    """        selected_experts_in
    ,
        moe_cache);""",
    """        selected_experts_in,
        moe_cache);""",
    "build_moe_ffn forwarding call formatting",
)
graph_path.write_text(graph, encoding="utf-8")

trace_path = Path("tools/moe-trace/moe-trace.cpp")
trace = trace_path.read_text(encoding="utf-8")
trace = replace_once(
    trace,
    """//   MOE_TRACE_OUT=trace.csv llama-moe-trace -m model.gguf -ngl 99 -ncmoe 26 -fa on \\
//       -p \"prompt text\" -n 512""",
    """//   MOE_TRACE_OUT=trace.csv llama-moe-trace -m model.gguf -ngl 99 -ncmoe 26
//       -fa on -p \"prompt text\" -n 512""",
    "moe-trace usage comment",
)
trace_path.write_text(trace, encoding="utf-8")

tests_path = Path("tests/test-backend-ops.cpp")
tests = tests_path.read_text(encoding="utf-8")
tests = replace_once(
    tests,
    "static void init_mul_mat_id_tensors(ggml_context * ctx, int n_mats, bool skip_ids = false) {",
    "static void init_mul_mat_id_tensors(ggml_context * ctx, int n_mats, bool skip_ids = false, bool all_skip_ids = false) {",
    "MUL_MAT_ID initializer signature",
)
tests = replace_once(
    tests,
    """                if (skip_ids) {
                    // id == -1 marks \"expert not owned by this pack\" (hot/cold expert
                    // split); keep slot 0 valid so every row computes something
                    for (int i = 1; i < t->ne[0]; i += 2) {
                        data[i] = -1;
                    }
                }
""",
    """                if (all_skip_ids) {
                    std::fill(data.begin(), data.end(), -1);
                } else if (skip_ids) {
                    // id == -1 marks \"expert not owned by this pack\" (hot/cold
                    // expert split); keep slot 0 valid for mixed-id coverage.
                    for (int i = 1; i < t->ne[0]; i += 2) {
                        data[i] = -1;
                    }
                }
""",
    "MUL_MAT_ID all-skipped initialization",
)
tests = replace_once(
    tests,
    """    const bool skip_ids; // some ids are -1 (hot/cold expert-pack split)

    std::string vars() override {
        return VARS_TO_STR9(type_a, type_b, n_mats, n_used, b, m, n, k, skip_ids);
    }
""",
    """    const bool skip_ids;     // some ids are -1 (hot/cold expert-pack split)
    const bool all_skip_ids; // every id is -1

    std::string vars() override {
        return VARS_TO_STR10(type_a, type_b, n_mats, n_used, b, m, n, k, skip_ids, all_skip_ids);
    }
""",
    "MUL_MAT_ID test fields",
)
tests = replace_once(
    tests,
    """    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32, bool skip_ids = false)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k), skip_ids(skip_ids) {
            GGML_ASSERT(n_used <= n_mats);
        }
""",
    """    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            bool skip_ids = false, bool all_skip_ids = false)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k), skip_ids(skip_ids), all_skip_ids(all_skip_ids) {
            GGML_ASSERT(n_used <= n_mats);
            GGML_ASSERT(!all_skip_ids || skip_ids);
        }
""",
    "MUL_MAT_ID test constructor",
)
tests = replace_once(
    tests,
    """        if (skip_ids) {
            // announce that ids may contain -1 so backends route around
            // kernels without skip support (mirrors llama's pack nodes)
            out->op_params[0] = 1;
        }
""",
    """        if (skip_ids) {
            // Announce that ids may contain -1 so backends route around
            // kernels without skip support (mirrors llama's pack nodes).
            out->op_params[0] = 1;
        }
""",
    "MUL_MAT_ID skip flag comment",
)
tests = replace_once(
    tests,
    "        init_mul_mat_id_tensors(ctx, n_mats, skip_ids);",
    "        init_mul_mat_id_tensors(ctx, n_mats, skip_ids, all_skip_ids);",
    "MUL_MAT_ID initializer call",
)
tests = replace_once(
    tests,
    """    for (ggml_type ta : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, 16, 8, false, 256, 1, 256, /*skip_ids=*/true));
        test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, 16, 8, false, 256, 4, 256, /*skip_ids=*/true));
        test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, 16, 8, false, 256, 64, 256, /*skip_ids=*/true));
    }
""",
    """    for (ggml_type ta : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, 16, 8, false, 256, 1, 256, /*skip_ids=*/true));
        test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, 16, 8, false, 256, 4, 256, /*skip_ids=*/true));
        test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, 16, 8, false, 256, 64, 256, /*skip_ids=*/true));
        // A batch can route exclusively to the opposite pack. Exercise the
        // all--1 edge case in the vector and general CUDA dispatch shapes.
        test_cases.emplace_back(new test_mul_mat_id(
                ta, GGML_TYPE_F32, 16, 8, false, 256, 1, 256,
                /*skip_ids=*/true, /*all_skip_ids=*/true));
        test_cases.emplace_back(new test_mul_mat_id(
                ta, GGML_TYPE_F32, 16, 8, false, 256, 64, 256,
                /*skip_ids=*/true, /*all_skip_ids=*/true));
    }
""",
    "MUL_MAT_ID all-skipped test cases",
)
tests_path.write_text(tests, encoding="utf-8")

print("perf MoE cache cleanup applied")
