## Inspiration

Our project started with a question: Could the buffer's ringing help us read a cat qubit?

A cat qubit stores information in two separated oscillator states. The challenge is identifying the state while preserving the protection that made the cat useful in the first place. We wanted to understand that tradeoff from the underlying physics.

## What We Built

We built a Jupyter notebook that follows the introductory track from photon loss to cat preparation, then extends into standard-track error characterization and readout design.

Our final strategy combines three steps: briefly expanding the cat, tracking the buffer's changing frequency, and matching the receiver to the emitted waveform. The buffer can ring; the receiver needs to know the tune.

The creative step was connecting the state motion, control phases, and receiver. Expanding the cat changes its interaction with the buffer, so we compensate for the estimated cross-Kerr frequency shift as the cat expands:

$$
\epsilon_B(t)=g_2\alpha^2[1+0.45r(t)],
\qquad
\delta_b(t)=-\chi\alpha^2[1+0.45r(t)].
$$

Here, the buffer drive and frequency correction share a smooth sine-squared ramp over a 1.5-microsecond pulse. The ramp returns to zero; the static correction remains during recovery. We derived how opposing pump and buffer-drive chirps produce that correction, with the homodyne reference following the same phase.

## How We Built It

We used Python, Dynamiqs, JAX, and QuTiP to simulate the Lindblad master equation and noisy measurement trajectories.

First came single-photon loss, pair loss, stabilization, Wigner functions, and logical-state preparation. We varied the memory and buffer Hilbert-space truncations independently and checked transient peaks and populations near the truncation boundaries. Before optimizing the readout, we needed to trust the simulation.

Then we compared flat, resonant, closed, and robust pulses, along with different receivers, recovery corrections, and temporary cat expansion. Expansion helped much more in the ideal model than under nonlinear stress. That gap led us to the frequency-tracking control. Testing each added control separately helped us check whether coordinating them actually mattered.

## Results, With the Receipts

In the strong nonlinear model, our joint pulse reduced endpoint damage by about **20%** relative to a closely matched flat baseline, at nearly equal mean-signal information and the same observation time. Damage includes wrong-codeword population and leakage. The joint pulse does require extra buffer amplitude and phase control.

With ideal detection, the final receiver made **14 errors in 1,024 records after 21 microseconds**: an error rate of 1.37%, with a 95% confidence interval of 0.82–2.28%. That takes seven cycles without resets. The flat baseline made nine errors, so improved accuracy is not established.

Preserving idle protection remains an unmet goal: the joint pulse's contrast decays at about **1.67 times the matched idle rate**, failing our preservation test. Numerical checks support that finding. The receiver sample with a smaller time step also leaves our strict statistical equivalence criterion unresolved.

## Challenges and What We Learned

One stochastic integration scheme produced artificial heating despite producing normalized states. Comparing its trajectory ensemble with the master equation exposed the error; we rejected the affected scores and corrected the numerical map. A normalized cat can still be a numerically suspicious cat.

The entire 105-cell notebook executed successfully in a fresh environment with pinned dependencies and the supplied artifacts. We also checked the saved error counts and confidence intervals independently.

Our biggest lesson: less damage, accurate assignment, and preserved protection each require separate evidence. The next control search should optimize total active decay relative to idle alongside readout error. We now have a benchmark that can tell us when that next idea actually works.
