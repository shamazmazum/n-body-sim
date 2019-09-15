#define G 100.0f
#define EPS 1.0f

float2 accel (float2 r_me, __local float *m, __local float2 *r)
{
    size_t grp_size = get_local_size(0);
    size_t i;

    float2 res = (0.0f, 0.0f);

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

    float2 res = (0.0f, 0.0f);
    size_t i, glob_id;

    for (i=0; i< num_groups; i++) {
        size_t glob_id = i*grp_size + loc_id;
        loc_r[loc_id] = r[glob_id];
        loc_m[loc_id] = m[glob_id];
        barrier (CLK_LOCAL_MEM_FENCE);

        res += accel (r_me, loc_m, loc_r);
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
