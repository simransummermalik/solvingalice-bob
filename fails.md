# stuff that did not work

this is just a record of the ideas we tried and then left out.

## short pulse

the short pulse sounded nice because it lets the buffer ring after the drive.
the problem is that it gives too little information under the same amplitude
limit. its information was about `0.243`, while the target was `0.79`.

so it is not a fair winner. lower damage here mostly means less measurement.

## very short flat pulse

the `0.75 us` flat pulse could not reach the information target without going
over the shared drive limit. it stays in the comparison table as an infeasible
candidate so we do not hide failed attempts.

## longer flat pulse

the `2.25 us` flat pulse lowered the wrong-codeword population a little, but it
left more leakage at the common three microsecond endpoint. the `1.5 us` flat
pulse was the cleaner baseline.

## robust closed pulse

the extra derivative constraints made the linear endpoint much less sensitive
to a five percent exchange-frequency error. that part worked.

but the full quantum training score was worse:

- closed nominal: about `2.59e-4`
- closed robust: about `2.71e-4`

we kept the robust version as a real alternative, but did not call it the
winner.

## plain averaging

integrating the whole noisy record throws away the ringing information. for
the closed pulse it gave about fifty percent assignment error. using only the
drive window was also close to guessing. the matched filter was much better.

## sign-following receiver

flipping the receiver sign whenever the predicted signal changes sign helped,
but it still gave about thirty percent error on the same records. the detailed
matched filter gave about eighteen percent.

## geometric recovery lobe

we tried adding a phase-controlled recovery lobe after the main pulse. it did
not improve the equal-information wrong-codeword plus leakage score. the best
small test was still slightly worse than the existing closed pulse.

## counterdiabatic derivative correction

we tried an orthogonal drive proportional to the time derivative of the main
pulse. this was meant to cancel nonadiabatic ringing. it increased logical
damage for every tested nonzero strength, so it is not in the final method.

## random closed-loop perturbations

we also perturbed the coefficients while preserving the linearized endpoint
closure. some perturbations produced much more leakage in the full quantum
model. returning the linearized mean motion is therefore not enough to make a
pulse safe.

the main lesson is that the full nonlinear Lindblad calculation has the final
vote. a pulse can look better in a simplified design model and still lose
once the actual cat-state damage is calculated.

## linear damage-aware pulse

we also maximized a simple linearized information divided by memory-motion
proxy. it reached the same information target, but the full quantum score was
about `3.76e-4`, worse than the nominal closed pulse at about `2.59e-4`.

this is another warning that a nice quadratic proxy is not the same thing as
logical bit protection.

## orthogonal buffer quadrature correction

after the nonlinear stress test, we added a small imaginary buffer-drive
quadrature during the temporary cat expansion. across `-0.06` to `+0.06` of
the idle buffer-drive amplitude, the information changed, but the damage
stayed around `2.54e-3` and did not beat the real buffer-drive ramp. it is
kept as a failed correction rather than quietly folded into the winning pulse.

## the fast trajectory step on long nonlinear records

the fast ket-normalized trajectory scheme looked fine in the earlier ideal
model, but using it for 21 us with the added Kerr terms produced about 8.05
photons where the master equation predicts 4.406. its apparent 34.6% assignment
error is therefore not a physical result. we retained the failed check and
rejected those scores.

normalizing the Kraus map as well as the state fixes the pilot's spurious
heating. the final audit also requires the trajectory ensemble to agree with
the deterministic photon number, then checks a smaller step and larger cutoff.
normalizing a ket is not enough to certify a stochastic integration.

## less damage did not mean idle protection

the final nonlinear joint pulse reads the initial bit with 14 errors in 1024
records after 21 us. that part works in the ideal-detector model. flat readout
gets 9 errors, so there is no accuracy win to claim here.

the protection comparison is the bigger problem. joint active contrast decays
about 1.67 times faster than its matched idle, even though it is much gentler
than our original active pulse. larger cutoffs and tighter tolerances do not
remove that difference. our 10% preservation test fails.

so the next control objective has to include total active decay relative to
idle. optimizing endpoint damage and subtracting the idle cost let us improve
the wrong score for this particular claim. the cat is readable, but it is
still paying rent.
