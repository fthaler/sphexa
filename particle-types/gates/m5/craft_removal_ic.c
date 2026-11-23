/*
 * Craft an M5 gate initial condition from a full restartable sphexa dump.
 *
 *   craft_removal_ic <in.h5> <out.h5> [--mode removal|allfluid] [--flag 0|1] [--step N]
 *
 * Copies /Step#<N> to /Step#0 (H5Ocopy) and the root file attributes, then:
 *   removal (default)  rewrite ptype (fluid 0 on 2 x 100 cluster members + 3
 *                      isolated doomed particles, 1 elsewhere), shrink the
 *                      doomed h to 1e-6, zero their v* and x_m1..z_m1 (x_m1 is
 *                      the displacement positionUpdate uses to rebuild v), and
 *                      set removeUnconvergedParticles / iteration on both the
 *                      step (i32/u64) and root (f64) groups.
 *   allfluid           set removeUnconvergedParticles only; leave iteration at
 *                      the source value (10) so a -s 20 -w 10 run matches M4.
 *
 * The output is created with H5F_ACC_EXCL, so remove stale outputs first.
 */
#include "removal_ic.h"

static void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s <in.h5> <out.h5> [--mode removal|allfluid] [--flag 0|1] [--step N]\n", argv0);
}

static int parse_int(const char* s, const char* what)
{
    char*  end = NULL;
    long   v   = strtol(s, &end, 10);
    if (!s[0] || !end || *end) dief("invalid %s '%s'", what, s);
    return (int)v;
}

int main(int argc, char** argv)
{
    const char* in_path  = NULL;
    const char* out_path = NULL;
    const char* mode     = "removal";
    int         flag     = 1;
    int         src_step = 0;

    if (argc < 3) { usage(argv[0]); return EXIT_FAILURE; }
    in_path  = argv[1];
    out_path = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) { mode = argv[++i]; }
        else if (strcmp(argv[i], "--flag") == 0 && i + 1 < argc) { flag = parse_int(argv[++i], "flag"); }
        else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) { src_step = parse_int(argv[++i], "step"); }
        else
        {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (strcmp(mode, "removal") != 0 && strcmp(mode, "allfluid") != 0) dief("unknown mode '%s'", mode);
    if (flag != 0 && flag != 1) dief("--flag must be 0 or 1 (got %d)", flag);
    if (src_step < 0) dief("--step must be non-negative (got %d)", src_step);

    char src_name[64];
    snprintf(src_name, sizeof(src_name), "/Step#%d", src_step);

    hid_t src = open_rd(in_path);
    if (H5Lexists(src, src_name, H5P_DEFAULT) <= 0) dief("input '%s' has no step group '%s'", in_path, src_name);

    hid_t dst = H5Fcreate(out_path, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (dst < 0) dief("could not create '%s' (remove an existing file first)", out_path);

    if (H5Ocopy(src, src_name, dst, "/Step#0", H5P_DEFAULT, H5P_DEFAULT) < 0)
        die("H5Ocopy of the step group failed");
    copy_root_attrs(src, dst);

    hid_t step = open_step(dst, "/Step#0");

    if (strcmp(mode, "removal") == 0)
    {
        Ic        ic;
        Selection sel;
        ic_load(src, src_name, 0, &ic);
        select_fluid(&ic, &sel);
        if (sel.count != M5_NCLUSTER + M5_NDOOMED)
            dief("selection produced %zu particles, expected %u", sel.count, M5_NCLUSTER + M5_NDOOMED);

        size_t   n  = (size_t)ic.n;
        uint8_t* pt = malloc(n * sizeof(*pt));
        if (!pt) die("out of memory");
        memset(pt, 1, n);
        for (size_t k = 0; k < sel.ncluster; ++k) pt[sel.cluster[k]] = 0;
        for (size_t k = 0; k < sel.ndoomed; ++k) pt[sel.doomed[k]] = 0;
        write_u8(step, "ptype", pt, ic.n);

        for (size_t k = 0; k < sel.ndoomed; ++k)
        {
            size_t i   = sel.doomed[k];
            ic.h[i]    = 1e-6f;
            ic.vx[i]   = 0.0f;
            ic.vy[i]   = 0.0f;
            ic.vz[i]   = 0.0f;
            ic.x_m1[i] = 0.0f;
            ic.y_m1[i] = 0.0f;
            ic.z_m1[i] = 0.0f;
        }
        write_floats(step, "h", ic.h, ic.n);
        write_floats(step, "vx", ic.vx, ic.n);
        write_floats(step, "vy", ic.vy, ic.n);
        write_floats(step, "vz", ic.vz, ic.n);
        write_floats(step, "x_m1", ic.x_m1, ic.n);
        write_floats(step, "y_m1", ic.y_m1, ic.n);
        write_floats(step, "z_m1", ic.z_m1, ic.n);

        write_attr_i32(step, "removeUnconvergedParticles", (int32_t)flag);
        write_attr_u64(step, "iteration", 0);
        write_attr_f64(dst, "removeUnconvergedParticles", (double)flag);
        write_attr_f64(dst, "iteration", 0.0);

        uint64_t cmin[2], cmax[2];
        for (int c = 0; c < 2; ++c)
        {
            cmin[c] = UINT64_MAX;
            cmax[c] = 0;
            for (size_t k = 0; k < M5_K; ++k)
            {
                uint64_t id = ic.id[sel.cluster[c * M5_K + k]];
                if (id < cmin[c]) cmin[c] = id;
                if (id > cmax[c]) cmax[c] = id;
            }
        }
        printf("craft_removal_ic: %s -> %s (mode=removal, flag=%d, source=%s)\n", in_path, out_path, flag,
               src_name);
        printf("  particles                : %llu\n", (unsigned long long)ic.n);
        printf("  cluster 0                : %zu particles, id %llu..%llu\n", (size_t)M5_K,
               (unsigned long long)cmin[0], (unsigned long long)cmax[0]);
        printf("  cluster 1                : %zu particles, id %llu..%llu\n", (size_t)M5_K,
               (unsigned long long)cmin[1], (unsigned long long)cmax[1]);
        printf("  doomed                   :");
        for (size_t k = 0; k < sel.ndoomed; ++k) printf(" %llu", (unsigned long long)ic.id[sel.doomed[k]]);
        printf("\n  step removeUnconvergedParticles=%d iteration=0\n", flag);
        printf("  root removeUnconvergedParticles=%d iteration=0\n", flag);

        free(pt);
        select_free(&sel);
        ic_free(&ic);
    }
    else
    {
        write_attr_i32(step, "removeUnconvergedParticles", (int32_t)flag);
        write_attr_f64(dst, "removeUnconvergedParticles", (double)flag);
        printf("craft_removal_ic: %s -> %s (mode=allfluid, flag=%d, source=%s)\n", in_path, out_path, flag,
               src_name);
        printf("  step removeUnconvergedParticles=%d, iteration unchanged\n", flag);
        printf("  root removeUnconvergedParticles=%d, iteration unchanged\n", flag);
    }

    H5Gclose(step);
    H5Fclose(dst);
    H5Fclose(src);
    return EXIT_SUCCESS;
}
