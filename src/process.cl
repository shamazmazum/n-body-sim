#define G 100.0f
#define EPS 1.0f

float2 accel (float2 r_me, __local float *m, __local float2 *r)
{
    size_t grp_size = get_local_size(0);
    size_t i;

    float2 res = (float2) (0.0f);

    for (i=0; i<grp_size; i++) {
        float2 dir = r[i] - r_me;
        float dist_sq = pown(dir.x, 2) + pown(dir.y, 2);
        res += G*m[i]/powr(dist_sq + EPS, 3.0f/2) * dir;
    }

    return res;
}

float2 collect_accel (float2 r_me,
                      __constant float *m,
                      __local float *loc_m,
                      __global float2 *r,
                      __local float2 *loc_r)
{
    size_t num_groups = get_num_groups(0);
    size_t grp_size = get_local_size(0);
    size_t loc_id = get_local_id(0);

    float2 res = (float2) (0.0f);
    size_t i, glob_id;

    for (i=0; i< num_groups; i++) {
        size_t glob_id = i*grp_size + loc_id;
        loc_r[loc_id] = r[glob_id];
        loc_m[loc_id] = m[glob_id];
        barrier (CLK_LOCAL_MEM_FENCE);

        res += accel (r_me, loc_m, loc_r);
        barrier (CLK_LOCAL_MEM_FENCE);
    }

    return res;
}

__kernel void take_step_euler (__constant float *m,
                               __global float2 *r,
                               __global float2 *v,
                               float delta,
                               __local float *loc_m,
                               __local float2 *loc_r)
{
    size_t glob_id = get_global_id(0);

    size_t i;
    float2 r_me = r[glob_id];
    float2 v_me = v[glob_id];

    float2 nr = v_me;
    float2 nv = collect_accel (r_me, m, loc_m, r, loc_r);

    r[glob_id] = r_me + nr * delta;
    v[glob_id] = v_me + nv * delta;
}

__kernel void take_step_rk2 (__constant float *m,
                             __global float2 *r,
                             __global float2 *v,
                             float delta,
                             __local float *loc_m,
                             __local float2 *loc_r)
{
    size_t glob_id = get_global_id(0);

    size_t i;
    float2 r_me = r[glob_id];
    float2 v_me = v[glob_id];

    float2 mv = collect_accel (r_me, m, loc_m, r, loc_r);
    float2 mr = v_me;

    float2 vpred = v_me + delta * mv;
    float2 rpred = r_me + delta * mr;

    r[glob_id] = rpred;
    v[glob_id] = vpred;
    barrier (CLK_GLOBAL_MEM_FENCE);

    float2 nv = collect_accel (rpred, m, loc_m, r, loc_r);
    float2 nr = vpred;

    r[glob_id] = r_me + (nr + mr) * delta / 2;
    v[glob_id] = v_me + (nv + mv) * delta / 2;
}

__kernel void take_step_rk4 (__constant float *m,
                             __global float2 *r,
                             __global float2 *v,
                             float delta,
                             __local float *loc_m,
                             __local float2 *loc_r)
{
    size_t glob_id = get_global_id(0);

    size_t i;
    float2 r_me = r[glob_id];
    float2 v_me = v[glob_id];
    float2 vpred;
    float2 rpred;

    float2 mv = collect_accel (r_me, m, loc_m, r, loc_r);
    float2 mr = v_me;

    vpred = v_me + delta * mv / 2;
    rpred = r_me + delta * mr / 2;

    r[glob_id] = rpred;
    v[glob_id] = vpred;
    barrier (CLK_GLOBAL_MEM_FENCE);

    float2 nv = collect_accel (rpred, m, loc_m, r, loc_r);
    float2 nr = vpred;

    vpred = v_me + delta * nv / 2;
    rpred = r_me + delta * nr / 2;

    r[glob_id] = rpred;
    v[glob_id] = vpred;
    barrier (CLK_GLOBAL_MEM_FENCE);

    float2 pv = collect_accel (rpred, m, loc_m, r, loc_r);
    float2 pr = vpred;

    vpred = v_me + delta * pv;
    rpred = r_me + delta * pr;

    r[glob_id] = rpred;
    v[glob_id] = vpred;
    barrier (CLK_GLOBAL_MEM_FENCE);

    float2 qv = collect_accel (rpred, m, loc_m, r, loc_r);
    float2 qr = vpred;

    r[glob_id] = r_me + (mr + 2*nr + 2*pr + qr) * delta / 6;
    v[glob_id] = v_me + (mv + 2*nv + 2*pv + qv) * delta / 6;
}

__kernel void kinetic_energy (__constant float *m,
                              __constant float2 *velocity,
                              __global float *out)
{
    size_t idx = get_global_id(0);

    float norm = length (velocity[idx]);
    out[idx] = m[idx] * pown(norm, 2) / 2;
}

__kernel void angular_momentum (__constant float *m,
                                __constant float2 *position,
                                __constant float2 *velocity,
                                __global float *out)
{
    size_t idx = get_global_id(0);
    float2 r = position[idx];
    float2 v = velocity[idx];

    out[idx] = m[idx] * (r.x * v.y - r.y * v.x);
}

float pe_group (float m_me, float2 r_me, __local float *m, __local float2 *r)
{
    size_t grp_size = get_local_size(0);
    size_t i;

    float res = 0.0f;

    for (i=0; i<grp_size; i++) {
        float2 dir = r[i] - r_me;
        float dist_sq = pown(dir.x, 2) + pown(dir.y, 2);
        res += -G*m_me*m[i]/sqrt(dist_sq + EPS);
    }

    return res;
}

/*
 * This function does more than twice the amount of computations than
 * needed, because, given the bodies i and j, it computes interaction
 * between pairs ii, jj, ij and ji while we need only ij or
 * ji. Because this function is not called often, I leave it as it
 * is.
 */
__kernel void potential_energy (__constant float *m,
                                __constant float2 *r,
                                __global float *out,
                                __local float *loc_m,
                                __local float2 *loc_r)
{
    size_t num_groups = get_num_groups(0);
    size_t grp_size = get_local_size(0);
    size_t loc_id = get_local_id(0);
    size_t glob_id = get_global_id(0);
    size_t i, idx;

    float res = 0.0f;
    float m_me = m[glob_id];
    float2 r_me = r[glob_id];

    for (i=0; i<num_groups; i++) {
        size_t idx = i*grp_size + loc_id;
        loc_r[loc_id] = r[idx];
        loc_m[loc_id] = m[idx];
        barrier (CLK_LOCAL_MEM_FENCE);

        res += pe_group (m_me, r_me, loc_m, loc_r);
        barrier (CLK_LOCAL_MEM_FENCE);
    }

    out[glob_id] = (res + G*pown(m_me, 2)/sqrt(EPS))/ 2;
}

__kernel void reduce (__global float *array,
                      __local float *tmp,
                      unsigned long length)
{
    size_t global_size = get_global_size(0);
    size_t global_idx = get_global_id(0);
    size_t local_idx = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t group_num = get_group_id(0);
    size_t i;
    float acc = 0.0f;

    for (i=0; i<length; i+=global_size) {
        size_t idx = global_idx + i;
        if (idx < length) acc += array[idx];
    }

    tmp[local_idx] = acc;
    barrier (CLK_LOCAL_MEM_FENCE);

    for (i=local_size>>1; i>0; i>>=1) {
        if (local_idx < i) tmp[local_idx] += tmp[local_idx + i];
        barrier (CLK_LOCAL_MEM_FENCE);
    }

    if (local_idx == 0) array[group_num] = tmp[0];
}
