#define CP 1000.0
#define GAMMA 1.4
#define CV (CP/GAMMA)
#define R  (CP - CV)
#define TAU_C (0.0000009/(3.0*340.0))
#define T_A (2000 * 6.0)
#define DH_F0 (-700 * CP)
__kernel void lax(const float dt, const float dx, const int n, __global float * state, __global float * state_old,__global float * f, __global float *w){
  int j = get_global_id(0) + 1;
  if(j < n - 1){
      state[4 * j] = 0.5 * (state_old[4* (j + 1)] + state_old[4 * (j - 1)]) - 0.5 * dt/dx * (f[4 * (j + 1)] - f[4 * (j - 1)]) + 0.5 * dt * w[4 * j];
      state[4 * j + 1] = 0.5 * (state_old[4 * (j + 1) + 1] + state_old[4 * (j - 1) + 1]) - 0.5 * dt/dx * (f[4 * (j + 1) + 1] - f[4 * (j - 1) + 1]) + 0.5 * dt * w[4 * j + 1];
      state[4 * j + 2] = 0.5 * (state_old[4 * (j + 1) + 2] + state_old[4 * (j - 1) + 2]) - 0.5 * dt/dx * (f[4 * (j + 1) + 2] - f[4 * (j - 1) + 2]) + 0.5 * dt * w[4 * j + 2];
      state[4 * j + 3] = 0.5 * (state_old[4 * (j + 1) + 3] + state_old[4 * (j - 1) + 3]) - 0.5 * dt/dx * (f[4 * (j + 1) + 3] - f[4 * (j - 1) + 3]) + 0.5 * dt * w[4 * j + 3];
  }
}

__kernel void get_fw(const int n, __global float * state, __global float * state_old, __global float * f, __global float *w){
    int j = get_global_id(0);
    float u, e, temp, h, p;
    if(j == 0){
        state[0] = state[4];
        state[1] = state[5];
        state[2] = state[6];
        state[3] = state[7];
    }
    if(j == (n - 1)){
        state[4 * j] = state[4 * (j - 1)];
        state[4 * j + 1] = state[4 * (j - 1) + 1];
        state[4 * j + 2] = state[4 * (j - 1) + 2];
        state[4 * j + 3] = state[4 * (j - 1) + 3];
    }
    if(j < n){
        //printf("fw state %d %f %f %f %f \n", j, state[4 * j],state[4 * j + 1],state[4 * j + 2], state[4 * j + 3]);
        u = state[4 * j + 1]/state[4 * j];
        e = state[4 * j + 2]/state[4 * j];

        temp = (e - state[4 * j + 3]*DH_F0 - 0.5 * u * u)/CV;
        h = e + R*temp;
        p = temp * state[4 * j] * R;

        f[4 * j] = state[4 * j + 1];
        f[4 * j + 1] = state[4 * j + 1] * u + p;
        f[4 * j + 2] = state[4 * j + 1] * h;
        f[4 * j + 3] = state[4 * j + 1] * state[4 * j + 3];
        w[4 * j + 3] = state[4 * j] / TAU_C * (1.0 - state[4 * j + 3])*exp(-T_A/temp);

        state_old[4 * j] = state[4 * j];
        state_old[4 * j + 1] = state[4 * j + 1];
        state_old[4 * j + 2] = state[4 * j + 2];
        state_old[4 * j + 3] = state[4 * j + 3];
        //printf("f %d %f %f %f %f\n",j,f[4 * j],f[4 * j + 1],f[4 * j + 2], f[4 * j + 3]);

        //printf("%f %f %f %f\n", w[4 * j],w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]);

    }

}
