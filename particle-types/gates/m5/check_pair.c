/*
 * Bitwise pair comparison for the M5 all-fluid no-op gate (G5).
 *
 *   check_pair <a.h5> <b.h5> [--vept]
 *
 * Mirrors particle-types/check_gate.py's M1 criterion without h5py:
 *  - same step groups at the root, same dataset names inside each step;
 *  - with --vept, b may carry an extra `ptype` dataset (a must not have one)
 *    and b's ptype must be all zero;
 *  - every common dataset is read with its native type and compared with
 *    memcmp (bitwise, no floating-point tolerance).
 *
 * Tolerance mode is deliberately out of scope; the M3 GPU tolerance gate stays
 * on check_gate.py.
 */
#include <hdf5.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char* msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(EXIT_FAILURE);
}

typedef struct
{
    char** items;
    size_t n;
    size_t cap;
} StrList;

static void list_push(StrList* l, const char* s)
{
    if (l->n == l->cap)
    {
        size_t cap = l->cap ? 2 * l->cap : 32;
        char** items = realloc(l->items, cap * sizeof(*items));
        if (!items) die("out of memory");
        l->items = items;
        l->cap   = cap;
    }
    size_t len = strlen(s) + 1;
    l->items[l->n] = malloc(len);
    if (!l->items[l->n]) die("out of memory");
    memcpy(l->items[l->n], s, len);
    ++l->n;
}

static int cmp_str(const void* pa, const void* pb)
{
    const char* const* a = pa;
    const char* const* b = pb;
    return strcmp(*a, *b);
}

static void list_free(StrList* l)
{
    for (size_t i = 0; i < l->n; ++i) free(l->items[i]);
    free(l->items);
    memset(l, 0, sizeof(*l));
}

static int list_has(const StrList* l, const char* s)
{
    return bsearch(&s, l->items, l->n, sizeof(*l->items), cmp_str) != NULL;
}

static void list_links(hid_t loc, H5O_type_t want, StrList* out)
{
    H5G_info_t info;
    if (H5Gget_info(loc, &info) < 0) die("could not inspect a group");
    for (hsize_t i = 0; i < info.nlinks; ++i)
    {
        char name[256];
        if (H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_INC, i, name, sizeof(name), H5P_DEFAULT) < 0)
            die("could not enumerate a group");
        H5O_info2_t oinfo;
        if (H5Oget_info_by_name3(loc, name, &oinfo, H5O_INFO_BASIC, H5P_DEFAULT) < 0)
            die("could not inspect a link");
        if (oinfo.type == want) list_push(out, name);
    }
    qsort(out->items, out->n, sizeof(*out->items), cmp_str);
}

static void print_list(const char* indent, const StrList* l)
{
    for (size_t i = 0; i < l->n; ++i) fprintf(stderr, "%s%s\n", indent, l->items[i]);
}

static int compare_lists(const char* what, const char* a_path, const char* b_path, const StrList* a,
                         const StrList* b)
{
    if (a->n == b->n)
    {
        int same = 1;
        for (size_t i = 0; i < a->n; ++i)
            if (strcmp(a->items[i], b->items[i]) != 0) { same = 0; break; }
        if (same) return 1;
    }
    fprintf(stderr, "FAIL: %s differ\n  %s:\n", what, a_path);
    print_list("    ", a);
    fprintf(stderr, "  %s:\n", b_path);
    print_list("    ", b);
    return 0;
}

static int datasets_equal(hid_t ga, hid_t gb, const char* a_path, const char* name)
{
    hid_t da = H5Dopen2(ga, name, H5P_DEFAULT);
    hid_t db = H5Dopen2(gb, name, H5P_DEFAULT);
    if (da < 0 || db < 0) die("could not open a dataset");

    hid_t ta = H5Dget_type(da);
    hid_t tb = H5Dget_type(db);
    if (H5Tequal(ta, tb) <= 0)
    {
        fprintf(stderr, "FAIL: %s/%s datatypes differ\n", a_path, name);
        H5Tclose(ta);
        H5Tclose(tb);
        H5Dclose(da);
        H5Dclose(db);
        return 0;
    }

    hid_t sa = H5Dget_space(da);
    hid_t sb = H5Dget_space(db);
    int   rank_a = H5Sget_simple_extent_ndims(sa);
    int   rank_b = H5Sget_simple_extent_ndims(sb);
    if (rank_a != rank_b || rank_a < 0)
    {
        fprintf(stderr, "FAIL: %s/%s dataspaces differ\n", a_path, name);
        H5Sclose(sa);
        H5Sclose(sb);
        H5Tclose(ta);
        H5Tclose(tb);
        H5Dclose(da);
        H5Dclose(db);
        return 0;
    }
    hsize_t dims_a[8], dims_b[8];
    if (rank_a > 8) die("dataset rank > 8");
    if (H5Sget_simple_extent_dims(sa, dims_a, NULL) < 0 || H5Sget_simple_extent_dims(sb, dims_b, NULL) < 0)
        die("could not read a dataspace");
    for (int d = 0; d < rank_a; ++d)
        if (dims_a[d] != dims_b[d])
        {
            fprintf(stderr, "FAIL: %s/%s shapes differ\n", a_path, name);
            H5Sclose(sa);
            H5Sclose(sb);
            H5Tclose(ta);
            H5Tclose(tb);
            H5Dclose(da);
            H5Dclose(db);
            return 0;
        }

    hid_t  native = H5Tget_native_type(ta, H5T_DIR_DEFAULT);
    size_t dsize  = H5Tget_size(native);
    size_t nelm   = 1;
    for (int d = 0; d < rank_a; ++d) nelm *= (size_t)dims_a[d];
    size_t nbytes = nelm * dsize;
    void*  buf_a  = malloc(nbytes ? nbytes : 1);
    void*  buf_b  = malloc(nbytes ? nbytes : 1);
    if (!buf_a || !buf_b) die("out of memory");

    int ok = 1;
    if (H5Dread(da, native, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf_a) < 0 ||
        H5Dread(db, native, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf_b) < 0)
    {
        fprintf(stderr, "FAIL: could not read %s/%s\n", a_path, name);
        ok = 0;
    }
    else if (memcmp(buf_a, buf_b, nbytes) != 0)
    {
        fprintf(stderr, "FAIL: %s/%s differs bitwise\n", a_path, name);
        ok = 0;
    }

    free(buf_a);
    free(buf_b);
    H5Tclose(native);
    H5Sclose(sa);
    H5Sclose(sb);
    H5Tclose(ta);
    H5Tclose(tb);
    H5Dclose(da);
    H5Dclose(db);
    return ok;
}

static int ptype_all_zero(hid_t g, const char* path)
{
    if (H5Lexists(g, "ptype", H5P_DEFAULT) <= 0) return 1;
    hid_t d = H5Dopen2(g, "ptype", H5P_DEFAULT);
    hid_t s = H5Dget_space(d);
    if (H5Sget_simple_extent_ndims(s) != 1)
    {
        fprintf(stderr, "FAIL: %s/ptype is not 1-D\n", path);
        H5Sclose(s);
        H5Dclose(d);
        return 0;
    }
    hsize_t n = 0;
    H5Sget_simple_extent_dims(s, &n, NULL);
    H5Sclose(s);
    uint8_t* pt = malloc(n ? (size_t)n : 1);
    if (!pt) die("out of memory");
    if (H5Dread(d, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, pt) < 0)
    {
        fprintf(stderr, "FAIL: could not read %s/ptype\n", path);
        free(pt);
        H5Dclose(d);
        return 0;
    }
    H5Dclose(d);
    int ok = 1;
    for (size_t i = 0; i < (size_t)n; ++i)
        if (pt[i] != 0)
        {
            fprintf(stderr, "FAIL: %s/ptype[%zu] = %u (expected 0)\n", path, i, (unsigned)pt[i]);
            ok = 0;
            break;
        }
    free(pt);
    return ok;
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) die("usage: check_pair <a.h5> <b.h5> [--vept]");
    const char* a_path = argv[1];
    const char* b_path = argv[2];
    int         vept   = 0;
    if (argc == 4)
    {
        if (strcmp(argv[3], "--vept") != 0) die("usage: check_pair <a.h5> <b.h5> [--vept]");
        vept = 1;
    }

    hid_t a = H5Fopen(a_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t b = H5Fopen(b_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (a < 0 || b < 0) die("could not open an input file");

    StrList groups_a = { 0 }, groups_b = { 0 };
    list_links(a, H5O_TYPE_GROUP, &groups_a);
    list_links(b, H5O_TYPE_GROUP, &groups_b);
    int errors = 0;
    if (!compare_lists("step groups", a_path, b_path, &groups_a, &groups_b))
    {
        list_free(&groups_a);
        list_free(&groups_b);
        H5Fclose(a);
        H5Fclose(b);
        return EXIT_FAILURE;
    }

    size_t n_checked = 0;
    for (size_t gi = 0; gi < groups_a.n; ++gi)
    {
        const char* step = groups_a.items[gi];
        hid_t       ga   = H5Gopen2(a, step, H5P_DEFAULT);
        hid_t       gb   = H5Gopen2(b, step, H5P_DEFAULT);
        if (ga < 0 || gb < 0) die("could not open a step group");

        StrList da = { 0 }, db = { 0 };
        list_links(ga, H5O_TYPE_DATASET, &da);
        list_links(gb, H5O_TYPE_DATASET, &db);

        if (vept)
        {
            if (list_has(&da, "ptype"))
            {
                fprintf(stderr, "FAIL: %s/%s has a ptype dataset (only b may)\n", a_path, step);
                ++errors;
            }
            for (size_t i = 0; i < da.n; ++i)
                if (!list_has(&db, da.items[i]))
                {
                    fprintf(stderr, "FAIL: %s/%s/%s is missing from b\n", a_path, step, da.items[i]);
                    ++errors;
                }
            for (size_t i = 0; i < db.n; ++i)
                if (!list_has(&da, db.items[i]) && strcmp(db.items[i], "ptype") != 0)
                {
                    fprintf(stderr, "FAIL: %s/%s/%s is an unexpected extra dataset\n", b_path, step,
                            db.items[i]);
                    ++errors;
                }
            if (!list_has(&db, "ptype"))
            {
                fprintf(stderr, "FAIL: %s/%s is missing the ptype dataset\n", b_path, step);
                ++errors;
            }
            if (!ptype_all_zero(gb, step)) ++errors;
        }
        else if (!compare_lists("dataset names", a_path, b_path, &da, &db))
        {
            ++errors;
        }

        for (size_t i = 0; i < da.n; ++i)
        {
            if (!list_has(&db, da.items[i])) continue;
            if (!datasets_equal(ga, gb, a_path, da.items[i])) ++errors;
            ++n_checked;
        }

        list_free(&da);
        list_free(&db);
        H5Gclose(ga);
        H5Gclose(gb);
    }

    if (errors == 0)
    {
        printf("check_pair: OK (%zu datasets bitwise-identical across %zu steps%s)\n", n_checked, groups_a.n,
               vept ? "; ptype all zero in b" : "");
    }
    else
    {
        fprintf(stderr, "check_pair: FAILED with %d error(s)\n", errors);
    }

    list_free(&groups_a);
    list_free(&groups_b);
    H5Fclose(a);
    H5Fclose(b);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
