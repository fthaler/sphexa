/*
 * Shared HDF5 helpers and the deterministic fluid selection used by the M5
 * acceptance-gate tools (craft_removal_ic, check_removal).
 *
 * The M5 tooling was originally written under /tmp and lost; this is a
 * re-derivation (particle-types/milestone-5.md section 3).  Keep the selection
 * rule identical in the crafter and the checker -- both include this header.
 *
 * Selection rationale (ng0 = 100, ngmax = 150, ngmin = 25 for the fixtures):
 *  - The two clusters are the 100 particles nearest to C+- = center +-
 *    0.25 * extent.x.  Their 100-nearest radius is ~0.115, below 2h for the
 *    glass h >= 0.05; members with a smaller h are grown by updateH until
 *    nc >= ngmin.  With only 200 fluid cluster members, nc stays <= ngmax+1,
 *    so every cluster member is converged and survives.
 *  - The doomed sites S1,2 = center +- 0.65 * extent.y and S3 = center +
 *    0.65 * extent.z lie outside the ~unit sphere of the Evrard fixture.  The
 *    nearest particle to each is shrunk to h = 1e-6; the capped h growth over
 *    10 iterations reaches only ~1e-4 (2h ~ 2e-4), far below the distance to
 *    the nearest fluid cluster (~1), so nc = 1 -> marked and removed.
 */
#ifndef REMOVAL_IC_H
#define REMOVAL_IC_H

#include <hdf5.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define M5_K        100u
#define M5_NCLUSTER (2u * M5_K)
#define M5_NDOOMED  3u

typedef struct
{
    hsize_t   n;
    double*   x;
    double*   y;
    double*   z;
    uint64_t* id;
    uint8_t*  ptype; /* NULL when the file has no ptype dataset */
    float*    h;
    float*    vx;
    float*    vy;
    float*    vz;
    float*    x_m1;
    float*    y_m1;
    float*    z_m1;
    double    box[6];
} Ic;

typedef struct
{
    size_t   count;
    size_t   ncluster;
    size_t   ndoomed;
    uint8_t* selected; /* n booleans */
    size_t   cluster[M5_NCLUSTER];
    size_t   doomed[M5_NDOOMED];
} Selection;

static inline void die(const char* msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(EXIT_FAILURE);
}

static inline void dief(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("error: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(EXIT_FAILURE);
}

static inline hid_t open_rd(const char* path)
{
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) dief("could not open '%s' for reading", path);
    return f;
}

static inline hid_t open_step(hid_t file, const char* step)
{
    if (H5Lexists(file, step, H5P_DEFAULT) <= 0) dief("missing step group '%s'", step);
    hid_t g = H5Gopen2(file, step, H5P_DEFAULT);
    if (g < 0) dief("could not open step group '%s'", step);
    return g;
}

static inline hsize_t dataset_size(hid_t loc, const char* name)
{
    hid_t d = H5Dopen2(loc, name, H5P_DEFAULT);
    if (d < 0) dief("missing dataset '%s'", name);
    hid_t s = H5Dget_space(d);
    if (H5Sget_simple_extent_ndims(s) != 1) dief("[%s] dataset is not 1-D", name);
    hsize_t n = 0;
    if (H5Sget_simple_extent_dims(s, &n, NULL) < 0) dief("could not read dataset shape [%s]", name);
    H5Sclose(s);
    H5Dclose(d);
    return n;
}

static inline void read_vec(hid_t loc, const char* name, hid_t mem_type, void* out, hsize_t n)
{
    if (dataset_size(loc, name) != n) dief("dataset '%s' has an unexpected length", name);
    hid_t d = H5Dopen2(loc, name, H5P_DEFAULT);
    if (d < 0) dief("missing dataset '%s'", name);
    if (H5Dread(d, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) < 0) dief("could not read dataset '%s'", name);
    H5Dclose(d);
}

static inline void write_vec(hid_t loc, const char* name, hid_t mem_type, const void* data, hsize_t n)
{
    if (dataset_size(loc, name) != n) dief("dataset '%s' has an unexpected length", name);
    hid_t d = H5Dopen2(loc, name, H5P_DEFAULT);
    if (d < 0) dief("missing dataset '%s'", name);
    if (H5Dwrite(d, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) dief("could not write dataset '%s'", name);
    H5Dclose(d);
}

static inline void read_doubles(hid_t loc, const char* name, double* out, hsize_t n)
{
    read_vec(loc, name, H5T_NATIVE_DOUBLE, out, n);
}

static inline void read_floats(hid_t loc, const char* name, float* out, hsize_t n)
{
    read_vec(loc, name, H5T_NATIVE_FLOAT, out, n);
}

static inline void read_u64(hid_t loc, const char* name, uint64_t* out, hsize_t n)
{
    read_vec(loc, name, H5T_NATIVE_UINT64, out, n);
}

static inline void read_u8(hid_t loc, const char* name, uint8_t* out, hsize_t n)
{
    read_vec(loc, name, H5T_NATIVE_UINT8, out, n);
}

static inline void write_floats(hid_t loc, const char* name, const float* data, hsize_t n)
{
    write_vec(loc, name, H5T_NATIVE_FLOAT, data, n);
}

static inline void write_u8(hid_t loc, const char* name, const uint8_t* data, hsize_t n)
{
    write_vec(loc, name, H5T_NATIVE_UINT8, data, n);
}

static inline int space_is_scalar(hid_t space)
{
    int nd = H5Sget_simple_extent_ndims(space);
    if (nd == 0) return 1;
    if (nd == 1)
    {
        hsize_t dim = 0;
        if (H5Sget_simple_extent_dims(space, &dim, NULL) < 0) return 0;
        return dim == 1;
    }
    return 0;
}

static inline void read_scalar_attr(hid_t loc, const char* name, hid_t mem_type, hid_t expect_class,
                                    size_t expect_size, void* out)
{
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    if (a < 0) dief("missing attribute '%s'", name);
    hid_t t = H5Aget_type(a);
    if (H5Tget_class(t) != expect_class || H5Tget_size(t) != expect_size)
        dief("attribute '%s' has an unexpected datatype", name);
    hid_t s = H5Aget_space(a);
    if (!space_is_scalar(s)) dief("attribute '%s' is not scalar", name);
    H5Sclose(s);
    if (H5Aread(a, mem_type, out) < 0) dief("could not read attribute '%s'", name);
    H5Tclose(t);
    H5Aclose(a);
}

static inline void read_attr_f64(hid_t loc, const char* name, double* out)
{
    read_scalar_attr(loc, name, H5T_NATIVE_DOUBLE, H5T_FLOAT, 8, out);
}

static inline void write_scalar_attr(hid_t loc, const char* name, hid_t mem_type, hid_t expect_class,
                                     size_t expect_size, const void* data)
{
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    if (a < 0) dief("missing attribute '%s'", name);
    hid_t t = H5Aget_type(a);
    if (H5Tget_class(t) != expect_class || H5Tget_size(t) != expect_size)
        dief("attribute '%s' has an unexpected datatype", name);
    hid_t s = H5Aget_space(a);
    if (!space_is_scalar(s)) dief("attribute '%s' is not scalar", name);
    H5Sclose(s);
    if (H5Awrite(a, mem_type, data) < 0) dief("could not write attribute '%s'", name);
    H5Tclose(t);
    H5Aclose(a);
}

static inline void write_attr_f64(hid_t loc, const char* name, double value)
{
    write_scalar_attr(loc, name, H5T_NATIVE_DOUBLE, H5T_FLOAT, 8, &value);
}

static inline void write_attr_i32(hid_t loc, const char* name, int32_t value)
{
    write_scalar_attr(loc, name, H5T_NATIVE_INT32, H5T_INTEGER, 4, &value);
}

static inline void write_attr_u64(hid_t loc, const char* name, uint64_t value)
{
    write_scalar_attr(loc, name, H5T_NATIVE_UINT64, H5T_INTEGER, 8, &value);
}

static inline void read_box_attr(hid_t loc, double box[6])
{
    hid_t a = H5Aopen(loc, "box", H5P_DEFAULT);
    if (a < 0) die("missing attribute 'box'");
    hid_t t = H5Aget_type(a);
    if (H5Tget_class(t) != H5T_FLOAT || H5Tget_size(t) != 8) die("attribute 'box' is not f64");
    hid_t s = H5Aget_space(a);
    hsize_t dims = 0;
    if (H5Sget_simple_extent_ndims(s) != 1 || H5Sget_simple_extent_dims(s, &dims, NULL) < 0 || dims != 6)
        die("attribute 'box' is not a 6-element vector");
    if (H5Aread(a, H5T_NATIVE_DOUBLE, box) < 0) die("could not read attribute 'box'");
    H5Sclose(s);
    H5Tclose(t);
    H5Aclose(a);
}

/* 1 when the file/group holds a link of the given name, 0 otherwise. */
static inline int has_link(hid_t loc, const char* name)
{
    htri_t r = H5Lexists(loc, name, H5P_DEFAULT);
    if (r < 0) dief("could not query link '%s'", name);
    return r > 0;
}

static inline void ic_load(hid_t file, const char* step, int require_ptype, Ic* ic)
{
    memset(ic, 0, sizeof(*ic));
    hid_t g  = open_step(file, step);
    ic->n    = dataset_size(g, "x");
    size_t n = (size_t)ic->n;

    ic->x  = malloc(n * sizeof(*ic->x));
    ic->y  = malloc(n * sizeof(*ic->y));
    ic->z  = malloc(n * sizeof(*ic->z));
    ic->id = malloc(n * sizeof(*ic->id));
    ic->h  = malloc(n * sizeof(*ic->h));
    ic->vx = malloc(n * sizeof(*ic->vx));
    ic->vy = malloc(n * sizeof(*ic->vy));
    ic->vz = malloc(n * sizeof(*ic->vz));
    ic->x_m1 = malloc(n * sizeof(*ic->x_m1));
    ic->y_m1 = malloc(n * sizeof(*ic->y_m1));
    ic->z_m1 = malloc(n * sizeof(*ic->z_m1));
    if (!ic->x || !ic->y || !ic->z || !ic->id || !ic->h || !ic->vx || !ic->vy || !ic->vz || !ic->x_m1 ||
        !ic->y_m1 || !ic->z_m1)
        die("out of memory");

    read_doubles(g, "x", ic->x, ic->n);
    read_doubles(g, "y", ic->y, ic->n);
    read_doubles(g, "z", ic->z, ic->n);
    read_u64(g, "id", ic->id, ic->n);
    read_floats(g, "h", ic->h, ic->n);
    read_floats(g, "vx", ic->vx, ic->n);
    read_floats(g, "vy", ic->vy, ic->n);
    read_floats(g, "vz", ic->vz, ic->n);
    read_floats(g, "x_m1", ic->x_m1, ic->n);
    read_floats(g, "y_m1", ic->y_m1, ic->n);
    read_floats(g, "z_m1", ic->z_m1, ic->n);

    if (has_link(g, "ptype"))
    {
        ic->ptype = malloc(n * sizeof(*ic->ptype));
        if (!ic->ptype) die("out of memory");
        read_u8(g, "ptype", ic->ptype, ic->n);
    }
    else if (require_ptype)
    {
        dief("step '%s' has no ptype dataset", step);
    }

    read_box_attr(g, ic->box);
    H5Gclose(g);
}

static inline void ic_free(Ic* ic)
{
    free(ic->x);
    free(ic->y);
    free(ic->z);
    free(ic->id);
    free(ic->ptype);
    free(ic->h);
    free(ic->vx);
    free(ic->vy);
    free(ic->vz);
    free(ic->x_m1);
    free(ic->y_m1);
    free(ic->z_m1);
    memset(ic, 0, sizeof(*ic));
}

typedef struct
{
    double   d2;
    uint64_t id;
    size_t   idx;
} Cand;

static inline int cand_better(const Cand* a, const Cand* b)
{
    if (a->d2 != b->d2) return a->d2 < b->d2;
    if (a->id != b->id) return a->id < b->id;
    return a->idx < b->idx;
}

/* Insertion-select into a fixed K-element array sorted by (dist^2, id). */
static inline void cand_insert(Cand* best, size_t* count, Cand c)
{
    if (*count < M5_K)
    {
        size_t pos = *count;
        while (pos > 0 && cand_better(&c, &best[pos - 1]))
        {
            best[pos] = best[pos - 1];
            --pos;
        }
        best[pos] = c;
        ++*count;
    }
    else if (cand_better(&c, &best[M5_K - 1]))
    {
        size_t pos = M5_K - 1;
        while (pos > 0 && cand_better(&c, &best[pos - 1]))
        {
            best[pos] = best[pos - 1];
            --pos;
        }
        best[pos] = c;
    }
}

static inline void select_fluid(const Ic* ic, Selection* sel)
{
    memset(sel, 0, sizeof(*sel));
    size_t n = (size_t)ic->n;
    sel->selected = calloc(n, 1);
    if (!sel->selected) die("out of memory");

    double center[3], extent[3];
    for (int d = 0; d < 3; ++d)
    {
        center[d] = 0.5 * (ic->box[2 * d] + ic->box[2 * d + 1]);
        extent[d] = ic->box[2 * d + 1] - ic->box[2 * d];
    }

    const double cx[2] = { center[0] - 0.25 * extent[0], center[0] + 0.25 * extent[0] };
    for (int c = 0; c < 2; ++c)
    {
        Cand   best[M5_K];
        size_t count = 0;
        for (size_t i = 0; i < n; ++i)
        {
            double dx = ic->x[i] - cx[c];
            double dy = ic->y[i] - center[1];
            double dz = ic->z[i] - center[2];
            Cand   cand = { dx * dx + dy * dy + dz * dz, ic->id[i], i };
            cand_insert(best, &count, cand);
        }
        if (count != M5_K) dief("cluster %d has only %zu candidates", c, count);
        for (size_t k = 0; k < M5_K; ++k)
        {
            size_t i = best[k].idx;
            if (sel->selected[i]) dief("clusters overlap at particle id %llu", (unsigned long long)ic->id[i]);
            sel->selected[i] = 1;
            sel->cluster[sel->ncluster++] = i;
        }
    }
    if (sel->ncluster != M5_NCLUSTER) dief("expected %u cluster members, got %zu", M5_NCLUSTER, sel->ncluster);

    const double sites[M5_NDOOMED][3] = {
        { center[0], center[1] - 0.65 * extent[1], center[2] },
        { center[0], center[1] + 0.65 * extent[1], center[2] },
        { center[0], center[1], center[2] + 0.65 * extent[2] },
    };
    for (int s = 0; s < (int)M5_NDOOMED; ++s)
    {
        Cand best   = { 0.0, 0, 0 };
        int  have   = 0;
        for (size_t i = 0; i < n; ++i)
        {
            if (sel->selected[i]) continue;
            double dx = ic->x[i] - sites[s][0];
            double dy = ic->y[i] - sites[s][1];
            double dz = ic->z[i] - sites[s][2];
            Cand   cand = { dx * dx + dy * dy + dz * dz, ic->id[i], i };
            if (!have || cand_better(&cand, &best))
            {
                best = cand;
                have = 1;
            }
        }
        if (!have) dief("no particle available for doomed site %d", s);
        if (sel->selected[best.idx])
            dief("doomed sites resolve to a duplicate particle id %llu", (unsigned long long)ic->id[best.idx]);
        sel->selected[best.idx]          = 1;
        sel->doomed[sel->ndoomed++]      = best.idx;
    }
    if (sel->ndoomed != M5_NDOOMED) dief("expected %u doomed particles, got %zu", M5_NDOOMED, sel->ndoomed);

    sel->count = sel->ncluster + sel->ndoomed;
}

static inline void select_free(Selection* sel)
{
    free(sel->selected);
    memset(sel, 0, sizeof(*sel));
}

/* Copy every scalar f64 root attribute.  The fixtures store the file-level
 * settings (and a duplicated iteration / removeUnconvergedParticles) here; the
 * step group stores the same settings with their in-memory C types. */
typedef struct
{
    hid_t dst;
} RootAttrCtx;

static inline herr_t copy_root_attr_cb(hid_t obj, const char* name, const H5A_info_t* ainfo, void* op_data)
{
    (void)ainfo;
    RootAttrCtx* ctx = op_data;

    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) return -1;
    hid_t t = H5Aget_type(a);
    if (H5Tget_class(t) != H5T_FLOAT || H5Tget_size(t) != 8)
    {
        fprintf(stderr, "error: root attribute '%s' is not scalar f64\n", name);
        H5Tclose(t);
        H5Aclose(a);
        return -1;
    }
    hid_t s = H5Aget_space(a);
    if (!space_is_scalar(s))
    {
        fprintf(stderr, "error: root attribute '%s' is not scalar\n", name);
        H5Sclose(s);
        H5Tclose(t);
        H5Aclose(a);
        return -1;
    }
    double value = 0.0;
    if (H5Aread(a, H5T_NATIVE_DOUBLE, &value) < 0)
    {
        H5Sclose(s);
        H5Tclose(t);
        H5Aclose(a);
        return -1;
    }
    H5Aclose(a);

    if (H5Aexists(ctx->dst, name) > 0) H5Adelete(ctx->dst, name);
    hid_t da = H5Acreate2(ctx->dst, name, t, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(s);
    H5Tclose(t);
    if (da < 0) return -1;
    if (H5Awrite(da, H5T_NATIVE_DOUBLE, &value) < 0)
    {
        H5Aclose(da);
        return -1;
    }
    return H5Aclose(da) < 0 ? -1 : 0;
}

static inline void copy_root_attrs(hid_t src, hid_t dst)
{
    hsize_t     idx = 0;
    RootAttrCtx ctx = { dst };
    if (H5Aiterate2(src, H5_INDEX_NAME, H5_ITER_INC, &idx, copy_root_attr_cb, &ctx) < 0)
        die("failed to copy root file attributes");
}

#endif /* REMOVAL_IC_H */
