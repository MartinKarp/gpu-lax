
#define CP 1000.0
#define GAMMA 1.4
#define CV (CP/GAMMA)
#define R  (CP - CV)
#define TAU_C (0.0000009/(3.0*340.0))
#define T_A (2000 * 6.0)
#define DH_F0 (-700 * CP)
#define CV_ (1/CV)
__kernel void lax(const float dt, const float dx, const int n, __global float * restrict state, const __global float * state_old,const __global float * f,const __global float *w){
  int j = get_global_id(0) + 1;
  if(j < n - 1){
      state[j] = 0.5 * (state_old[j + 1] + state_old[j - 1]) - 0.5 * dt/dx * (f[j + 1] - f[j - 1]);
      state[j + 1 * n] = 0.5 * (state_old[j + 1 + n * 1] + state_old[j - 1 + n * 1]) - 0.5 * dt/dx * (f[j + 1 + 1 * n] - f[j - 1 + n * 1]);
      state[j + 2 * n] = 0.5 * (state_old[j + 1 + n * 2] + state_old[j - 1 + n * 2]) - 0.5 * dt/dx * (f[j + 1 + 2 * n] - f[j - 1 + n * 2]);
      state[j + 3 * n] = 0.5 * (state_old[j + 1 + n * 3] + state_old[j - 1 + n * 3]) - 0.5 * dt/dx * (f[j + 1 + 3 * n] - f[j - 1 + n * 3]) + 0.5 * dt * w[j + 3 * n];
      if( state[j + 3 * n] > 1) state[j + 3 * n]  = 1;
      if( state[j + 3 * n] < 0) state[j + 3 * n]  = 0;
  }
}

__kernel void get_fw(const int n, const  __global float * restrict state, __global float * restrict state_old, __global float * f, __global float * restrict w){
    int j = get_global_id(0);
    int j2 = j;
    float u, e, temp1, temp2, h, p, rhou;
    if(j == 0) j2++;
    if(j == (n - 1)) j2--;
    if(j < n){
        //printf("fw state %d %f %f %f %f \n", j, state[j2],state[j2 + 1 * n],state[j2 + 2 * n], state[j2 + 3 * n]);
        rhou = state[j2 + 1 * n];
        u = rhou/state[j2];
        e = state[j2 + 2 * n]/state[j2];

        temp1 = (e - state[j2 + 3 * n]*DH_F0 - 0.5 * u * u) * CV_;
        h = e + R*temp1;
        p = temp1 * state[j2] * R;

        f[j] = rhou;
        f[j + 1 * n] = rhou * u + p;
        f[j + 2 * n] = rhou * h;
        f[j + 3 * n] = rhou * state[j2 + 3 * n];

        w[j] = u;
        w[j + 1 * n] = temp1;

        state_old[j] = state[j2];
        state_old[j + 1 * n] = rhou;
        state_old[j + 2 * n] = state[j2 + 2 * n];
        state_old[j + 3 * n] = state[j2 + 3 * n];

        u = state[j2 - 1 + 1 * n]/state[j2 - 1];

        e = state[j2 - 1 + 2 * n]/state[j2 - 1];
        temp1 = (e - state[j2 - 1 + 3 * n]*DH_F0 - 0.5 * u * u) * CV_;

        u = state[j2 + 1 + 1 * n]/state[j2 + 1];
        e = state[j2 + 1 + 2 * n]/state[j2 + 1];
        temp2 = (e - state[j2 + 1 + 3 * n]*DH_F0 - 0.5 * u * u) * CV_;


        w[j + 3 * n] = state[j2] / TAU_C * (1.0 - 0.5 * (state[j2 - 1 + 3 * n] + state[j2 + 1 + 3 * n]))*exp(-2 * T_A/(temp1 + temp2));

        //printf("f %d %f %f %f %f\n",j,f[j],f[j + 1 * n],f[j + 2 * n], f[j + 3 * n]);
        //printf("%f %f %f %f\n", w[j],w[j + 1 * n], w[j + 2 * n], w[j + 3 * n]);
    }
}
