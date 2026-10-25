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
