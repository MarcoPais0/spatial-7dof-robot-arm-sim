# Jacobian and Inverse Differential Kinematics

## Geometric Jacobian

The base-frame geometric Jacobian maps joint rates to end-effector spatial velocity:

```text
x_dot = [v; omega] = J(q) q_dot
v_i = z_(i-1) x (p_e - p_(i-1))
omega_i = z_(i-1)
J_i = [v_i; omega_i]
```

The position Jacobian `J_v` is the upper 3x7 block of `J`.

With the default tool alignment aligned to the final wrist axis, the terminal roll joint contributes orientation but not tool-origin translation, so the last column of `J_v` is zero.

## Singularity Analysis

The position Jacobian is analyzed through its singular values:

- numerical rank
- smallest singular value
- condition number
- manipulability

These quantities support both documentation and damping logic.

## Inverse Differential Kinematics

The resolved-rate position controller uses damped least squares:

```text
v_des_raw = K_x (x_d - x)
v_des = sat_v(v_des_raw, v_max)
sigma_min_value = min_singular_value(J_v)
lambda = lambda_0 + k_lambda / (sigma_min_value + epsilon)
J_v^+ = J_v^T (J_v J_v^T + lambda^2 I_3)^(-1)
q_dot_cmd = J_v^+ v_des
q_ref_next = q_ref + dt q_dot_cmd
q_dot_ref = q_dot_cmd
```

## Design Notes

- Damping grows as the smallest singular value decreases.
- The method prioritizes stability near singularities over perfect tracking accuracy.
- The inverse-kinematics loop is position-first; orientation is not the primary task.
