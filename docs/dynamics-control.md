# Dynamics and Control

## Simplified Dynamics

The project uses a reduced joint-space model:

```text
M q_ddot = tau - D q_dot - g(q)
M approx diag(I_1, I_2, I_3, I_4, I_5, I_6, I_7)
D approx diag(d_1, d_2, d_3, d_4, d_5, d_6, d_7)
```

This keeps the simulation simple and consistent with the educational scope of the arm model.

## Joint-Space Control

The controller is a per-joint PD tracker with direct gravity compensation and torque saturation:

```text
e = q_ref - q
e_dot = q_dot_ref - q_dot
tau_cmd = K_p e + K_d e_dot + g(q)
tau = sat(tau_cmd, -tau_limit, tau_limit)
```

The gravity term is included directly in the controller so it approximately cancels the gravity term in the dynamics. The default torque limit is `tau_limit = 50 N m` per joint.

For the 7DOF arm, the gravity approximation uses a lightweight segment-length profile rather than a full rigid-body model, which keeps the demo stable without introducing a much heavier dynamics stack.

## Numerical Integration

The simulation uses semi-implicit Euler stepping:

```text
q_ddot = M^(-1) (tau - D q_dot - g(q))
q_dot_next = q_dot + dt q_ddot
q_next = q + dt q_dot_next
```

## Design Notes

- The inertia matrix is diagonal and configuration-independent.
- The controller uses velocity error rather than finite-difference differentiation.
- The integral term is not used; the controller is intentionally PD-only to avoid integral windup and zero-crossing behavior from accumulation.
- The integration step is fixed so the runtime loop stays predictable.
