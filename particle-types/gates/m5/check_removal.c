/*
 * Check an M5 removal run against the crafted IC.
 *
 *   check_removal <crafted_ic.h5> <run_output.h5> [--step N] [--max-nonfluid-drop N]
 *
 * Re-derives the selection from the crafted IC (same shared select_fluid),
 * verifies that the IC's ptype marks exactly that selection (drift guard), then
 * reads id/ptype from the last (or --step) output step and asserts:
 *  - every cluster id survives with ptype 0;
 *  - every doomed id is gone;
 *  - every surviving id exists in the IC and keeps its IC ptype;
 *  - the only losses are the 3 doomed particles plus at most
 *    --max-nonfluid-drop non-fluid particles (default 0; the 2-rank gates pass
 *    1 for the pre-existing one-particle sync drop).  Any fluid loss is fatal.
 *
 * Exit 0 on success, 1 with diagnostics otherwise.
 */
#include "removal_ic.h"

typedef struct
{
    uint64_t id;
    size_t   idx;
} IdEntry;

static int cmp_entry(const void* pa, const void* pb)
{
    const IdEntry* a = pa;
    const IdEntry* b = pb;
    return (a->id < b->id) ? -1 : (a->id > b->id) ? 1 : 0;
}

static const IdEntry* find_entry(const IdEntry* entries, size_t n, uint64_t id)
{
    IdEntry key = { id, 0 };
    return bsearch(&key, entries, n, sizeof(*entries), cmp_entry);
}

/* Return 1 and fill step_name when a step group exists; prefer the highest N. */
static int last_step(hid_t file, char* step_name, size_t cap, unsigned* step_num)
{
    hsize_t links = 0;
    H5G_info_t info;
    if (H5Gget_info(file, &info) < 0) die("could not inspect the output file");
    links = info.nlinks;

    int      found = 0;
    unsigned best  = 0;
    for (hsize_t i = 0; i < links; ++i)
    {
        char name[256];
        ssize_t len = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i, name, sizeof(name),
                                         H5P_DEFAULT);
        if (len < 0) die("could not enumerate the output file");
        if (len < 5 || strncmp(name, "Step#", 5) != 0) continue;
        char* end = NULL;
        unsigned long v = strtoul(name + 5, &end, 10);
        if (!name[5] || !end || *end) continue;
        if (!found || v > best)
        {
            found = 1;
            best  = (unsigned)v;
        }
    }
    if (!found) return 0;
    if (snprintf(step_name, cap, "/Step#%u", best) >= (int)cap) die("step name buffer too small");
    *step_num = best;
    return 1;
}

static void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s <crafted_ic.h5> <run_output.h5> [--step N] [--max-nonfluid-drop N]\n", argv0);
}

static int parse_int(const char* s, const char* what)
{
    char* end = NULL;
    long  v   = strtol(s, &end, 10);
    if (!s[0] || !end || *end) dief("invalid %s '%s'", what, s);
    return (int)v;
}

int main(int argc, char** argv)
{
    if (argc < 3) { usage(argv[0]); return EXIT_FAILURE; }
    const char* ic_path = argv[1];
    const char* out_path = argv[2];
    int         want_step = -1;
    int         max_nonfluid_drop = 0;

    for (int i = 3; i < argc; ++i)
    {
        if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) { want_step = parse_int(argv[++i], "step"); }
        else if (strcmp(argv[i], "--max-nonfluid-drop") == 0 && i + 1 < argc)
        {
            max_nonfluid_drop = parse_int(argv[++i], "max-nonfluid-drop");
        }
        else
        {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (max_nonfluid_drop < 0) die("--max-nonfluid-drop must be non-negative");

    hid_t ic_file = open_rd(ic_path);
    Ic    ic;
    ic_load(ic_file, "/Step#0", 1, &ic);
    Selection sel;
    select_fluid(&ic, &sel);
    if (sel.count != M5_NCLUSTER + M5_NDOOMED)
        dief("selection produced %zu particles, expected %u", sel.count, M5_NCLUSTER + M5_NDOOMED);

    int errors = 0;
#define FAIL(...)                                                                                              \
    do                                                                                                         \
    {                                                                                                          \
        fputs("FAIL: ", stderr);                                                                               \
        fprintf(stderr, __VA_ARGS__);                                                                          \
        fputc('\n', stderr);                                                                                   \
        ++errors;                                                                                              \
    } while (0)

    /* 1. the crafted IC must mark exactly the re-derived selection */
    size_t mismatches = 0;
    for (size_t i = 0; i < (size_t)ic.n; ++i)
    {
        uint8_t expected = sel.selected[i] ? 0 : 1;
        if (ic.ptype[i] != expected)
        {
            if (mismatches < 10)
                FAIL("IC ptype drift at index %zu (id %llu): got %u, expected %u", i,
                     (unsigned long long)ic.id[i], (unsigned)ic.ptype[i], (unsigned)expected);
            ++mismatches;
        }
    }
    if (mismatches >= 10) FAIL("IC ptype drift: %zu mismatching particles in total", mismatches);

    /* 2. output step */
    char     out_step[64];
    unsigned out_step_num = 0;
    hid_t    out_file     = open_rd(out_path);
    if (want_step >= 0)
    {
        snprintf(out_step, sizeof(out_step), "/Step#%d", want_step);
        out_step_num = (unsigned)want_step;
        if (H5Lexists(out_file, out_step, H5P_DEFAULT) <= 0) dief("output has no step group '%s'", out_step);
    }
    else if (!last_step(out_file, out_step, sizeof(out_step), &out_step_num))
    {
        dief("output '%s' contains no step group", out_path);
    }

    hid_t out_group = open_step(out_file, out_step);
    hsize_t nout    = dataset_size(out_group, "id");
    if (dataset_size(out_group, "ptype") != nout) die("output id/ptype lengths differ");
    uint64_t* out_ids   = malloc((size_t)nout * sizeof(*out_ids));
    uint8_t*  out_ptype = malloc((size_t)nout * sizeof(*out_ptype));
    if (!out_ids || !out_ptype) die("out of memory");
    read_u64(out_group, "id", out_ids, nout);
    read_u8(out_group, "ptype", out_ptype, nout);
    H5Gclose(out_group);

    IdEntry* ic_entries = malloc((size_t)ic.n * sizeof(*ic_entries));
    IdEntry* out_entries = malloc((size_t)nout * sizeof(*out_entries));
    if (!ic_entries || !out_entries) die("out of memory");
    for (size_t i = 0; i < (size_t)ic.n; ++i) ic_entries[i] = (IdEntry){ ic.id[i], i };
    for (size_t i = 0; i < (size_t)nout; ++i) out_entries[i] = (IdEntry){ out_ids[i], i };
    qsort(ic_entries, (size_t)ic.n, sizeof(*ic_entries), cmp_entry);
    qsort(out_entries, (size_t)nout, sizeof(*out_entries), cmp_entry);

    /* duplicate output ids would make the survivor set ambiguous */
    for (size_t i = 1; i < (size_t)nout; ++i)
        if (out_entries[i].id == out_entries[i - 1].id)
        {
            FAIL("duplicate id %llu in the output", (unsigned long long)out_entries[i].id);
            break;
        }

    /* 3a. cluster members survive with ptype 0 */
    size_t cluster_ok = 0;
    for (size_t k = 0; k < sel.ncluster; ++k)
    {
        size_t           i  = sel.cluster[k];
        const IdEntry*   oe = find_entry(out_entries, (size_t)nout, ic.id[i]);
        if (!oe) { FAIL("cluster fluid particle id %llu is missing", (unsigned long long)ic.id[i]); continue; }
        if (out_ptype[oe->idx] != 0)
            FAIL("cluster particle id %llu has ptype %u (expected 0)", (unsigned long long)ic.id[i],
                 (unsigned)out_ptype[oe->idx]);
        else
            ++cluster_ok;
    }

    /* 3b. doomed particles are gone */
    size_t doomed_removed = 0;
    size_t doomed_missing = 0;
    for (size_t k = 0; k < sel.ndoomed; ++k)
    {
        size_t         i  = sel.doomed[k];
        const IdEntry* oe = find_entry(out_entries, (size_t)nout, ic.id[i]);
        if (oe)
            FAIL("doomed particle id %llu survived in the output", (unsigned long long)ic.id[i]);
        else
        {
            ++doomed_removed;
            ++doomed_missing;
        }
    }
    if (doomed_removed != M5_NDOOMED)
        FAIL("only %zu/%u doomed particles were removed", doomed_removed, M5_NDOOMED);

    /* 3c. every surviving id exists in the IC and keeps its ptype */
    for (size_t j = 0; j < (size_t)nout; ++j)
    {
        const IdEntry* ie = find_entry(ic_entries, (size_t)ic.n, out_ids[j]);
        if (!ie) { FAIL("output id %llu is not in the crafted IC", (unsigned long long)out_ids[j]); continue; }
        if (out_ptype[j] != ic.ptype[ie->idx])
            FAIL("id %llu changed ptype: IC %u, output %u", (unsigned long long)out_ids[j],
                 (unsigned)ic.ptype[ie->idx], (unsigned)out_ptype[j]);
    }

    /* 3d. classify every missing id: doomed, or a (bounded) non-fluid drop */
    size_t   nonfluid_drop = 0;
    uint64_t missing_doomed_ids[M5_NDOOMED];
    for (size_t k = 0; k < M5_NDOOMED; ++k) missing_doomed_ids[k] = ic.id[sel.doomed[k]];

    for (size_t i = 0; i < (size_t)ic.n; ++i)
    {
        if (find_entry(out_entries, (size_t)nout, ic.id[i])) continue;
        int is_doomed = 0;
        for (size_t k = 0; k < M5_NDOOMED; ++k)
            if (missing_doomed_ids[k] == ic.id[i]) { is_doomed = 1; break; }
        if (is_doomed) continue;

        ++nonfluid_drop;
        if (ic.ptype[i] == 0)
            FAIL("fluid particle id %llu was lost (not a doomed particle)", (unsigned long long)ic.id[i]);
        else
            fprintf(stderr, "note: non-fluid particle id %llu dropped (IC ptype %u)\n",
                    (unsigned long long)ic.id[i], (unsigned)ic.ptype[i]);
    }

    if (nonfluid_drop > (size_t)max_nonfluid_drop)
        FAIL("%zu non-fluid particles dropped, limit is %d", nonfluid_drop, max_nonfluid_drop);

    size_t n_missing = (size_t)ic.n - (size_t)nout;
    if (n_missing != doomed_missing + nonfluid_drop)
        FAIL("missing count %zu != doomed %zu + non-fluid drop %zu", n_missing, doomed_missing, nonfluid_drop);
    if ((size_t)nout != (size_t)ic.n - M5_NDOOMED - nonfluid_drop)
        FAIL("output has %llu particles, expected %llu", (unsigned long long)nout,
             (unsigned long long)((size_t)ic.n - M5_NDOOMED - nonfluid_drop));

    if (errors == 0)
    {
        printf("check_removal: OK\n");
        printf("  crafted IC : %s (n=%llu, fluid=%zu)\n", ic_path, (unsigned long long)ic.n, sel.count);
        printf("  run output : %s %s (n=%llu)\n", out_path, out_step, (unsigned long long)nout);
        printf("  cluster    : %zu/%u survived with ptype 0\n", cluster_ok, M5_NCLUSTER);
        printf("  doomed     : %zu/%u removed (ids", doomed_removed, M5_NDOOMED);
        for (size_t k = 0; k < sel.ndoomed; ++k)
            printf(" %llu", (unsigned long long)ic.id[sel.doomed[k]]);
        printf(")\n");
        printf("  non-fluid  : %zu dropped (limit %d)\n", nonfluid_drop, max_nonfluid_drop);
    }
    else
    {
        fprintf(stderr, "check_removal: FAILED with %d error(s)\n", errors);
    }

    free(ic_entries);
    free(out_entries);
    free(out_ids);
    free(out_ptype);
    select_free(&sel);
    ic_free(&ic);
    H5Fclose(out_file);
    H5Fclose(ic_file);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
#undef FAIL
}
