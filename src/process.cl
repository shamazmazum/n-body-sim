#define G 100.0f
#define EPS 1.0f

float2 accel (float2 r_me, __local float *m, __local float2 *r)
{
    size_t grp_size = get_local_size(0);
    size_t i;

    float2 res = (float2) (0.0f);

    for (i=0; i<grp_size; i++) {
        float2 dir = r[i] - r_me;
        float dist = length(dir) + EPS;
        res += G*m[i]/pown(dist, 3) * dir;
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

    float2 nv = collect_accel (rpred, m, loc_m, r, loc_r);
    float2 nr = vpred;

    r[glob_id] = r_me + (nr + mr) * delta / 2;
    v[glob_id] = v_me + (nv + mv) * delta / 2;
}

__kernel void kinetic_energy (__constant float *m,
                              __constant float2 *velocity,
                              __global float *out)
{
    size_t idx = get_global_id(0);

    float norm = length (velocity[idx]);
    out[idx] = m[idx] * pown(norm, 2) / 2;
}

float pe_group (float m_me, float2 r_me, __local float *m, __local float2 *r)
{
    size_t grp_size = get_local_size(0);
    size_t i;

    float res = 0.0f;

    for (i=0; i<grp_size; i++) {
        float2 dir = r[i] - r_me;
        float dist = length(dir);
        // KLUDGE
        res += (dist != 0)? -G*m_me*m[i]/(dist + EPS): 0.0f;
    }

    return res;
}

__kernel void potential_energy (__constant float *m,
                                __constant float2 *r,
                                __local float *loc_m,
                                __local float2 *loc_r,
                                __global float *out)
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

    out[glob_id] = res;
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
