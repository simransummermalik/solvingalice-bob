# %% [markdown]
# cat qubits, from losing photons to reading a bit
#
# this follows the introductory track, then the standard track of the supplied
# duquantum 2026 prompt. the small comments are there so the calculations can be
# walked through, not just run. all times below are in microseconds.
#
# the questions i want to keep asking: is this a real cat or just two blobs?
# is a small error physical, or did the fock-space ceiling manufacture it?
# and can the buffer's ringing help a measurement instead of just delaying it?
#
# this notebook was developed with substantial codex assistance; see the readme.
# numerical results are computed below, not copied from the prompt's expectations.

# %%
import os  # keep caches local when this runs on another computer
from pathlib import Path  # portable paths for the saved tables and figures
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import time  # records the cost of the numerical checks
import json  # saves a small machine-readable summary at the end
import hashlib  # ties cached full-model runs to the actual matrices and solver settings
import importlib.metadata as metadata  # records exactly which package versions ran
import numpy as np  # ordinary arrays for analysis and plotting
import pandas as pd  # puts the convergence results in readable tables
import matplotlib.pyplot as plt  # makes the figures
import scipy.linalg as la  # dense linear algebra for small verification problems
from scipy.optimize import curve_fit  # fits decay rates rather than eyeballing them
from scipy.sparse.linalg import eigs  # finds slow modes without evolving for seconds
from scipy.integrate import cumulative_trapezoid  # accumulates readout information
from scipy.special import erfc  # gaussian discrimination error, with caveats below
import qutip as qt  # independent sparse solver and wigner-function calculation
import dynamiqs as dq  # the master-equation solver requested in the challenge
import jax  # dynamiqs uses jax underneath
import jax.numpy as jnp  # arrays which the compiled solver understands
from IPython.display import display  # keeps tables readable inside the notebook

dq.set_precision("double")  # tiny decay rates need 64-bit, not the tutorial's 32-bit
dq.set_progress_meter(False)  # a separate progress bar for every solve gets noisy
plt.rcParams.update({"figure.dpi": 115, "axes.spines.top": False, "axes.spines.right": False})
artifacts = Path("artifacts")  # these are generated results, not input data
artifacts.mkdir(exist_ok=True)
g2 = 2 * np.pi * 0.763  # mhz -> angular frequency per microsecond
kb = 2 * np.pi * 2.6  # intentionally leaky buffer
ka = 2 * np.pi * 0.0093  # unwanted single-photon memory loss
alpha = 2.1  # target separation in coherent-state amplitude
k2 = 4 * g2**2 / kb  # eliminated-buffer prediction, not automatically valid
eps_d = g2 * alpha**2  # buffer drive with the sign convention of this prompt
versions = {name: metadata.version(name) for name in ["dynamiqs", "jax", "qutip", "numpy", "scipy", "matplotlib", "pandas"]}
print(versions)
print(f"jax double precision: {jax.config.x64_enabled}; devices: {jax.devices()}")
display(pd.Series({"g2": g2, "kb": kb, "ka": ka, "k2": k2, "eps_d": eps_d}, name="per microsecond"))

# %% [markdown]
# a few reusable pieces, so changing a cutoff really changes the same model.
#
# a density matrix contains populations on its diagonal and coherences off its
# diagonal. the two-mode matrix is indexed as `(memory, buffer)` on each side.
# tracing out the buffer means summing over its matching row/column indices.
# i save the reduced memory state during a solve, rather than hundreds of large
# two-mode matrices. this changes storage, not the equation being integrated.

# %%
def cat_ket(n, amplitude, sign=1):
    plus = dq.coherent(n, amplitude)  # one well
    minus = dq.coherent(n, -amplitude)  # the other well
    return dq.unit(plus + sign * minus)  # normalization includes their overlap


def two_mode(na, nb, amplitude=alpha, loss=ka, drive=0.0, coupling=g2, buffer_loss=kb):
    a, b = dq.destroy(na, nb)  # tensor-product operators, not two unrelated matrices
    h = coupling * (a @ a @ b.dag() + a.dag() @ a.dag() @ b)
    h = h - coupling * amplitude**2 * (b + b.dag())  # same target wells at every cutoff
    h = h + drive * (a + a.dag())  # optional real resonant memory drive
    jumps = [np.sqrt(buffer_loss) * b, np.sqrt(loss) * a]  # rates belong inside square roots
    tail_a = dq.tensor(dq.fock_dm(na, na - 2) + dq.fock_dm(na, na - 1), dq.eye(nb))
    tail_b = dq.tensor(dq.eye(na), dq.fock_dm(nb, nb - 1))  # include the transient, not only the end
    parity = dq.tensor(dq.parity(na), dq.eye(nb))  # only memory parity is the cat observable
    observables = [a.dag() @ a, b.dag() @ b, a, b, parity, tail_a, tail_b]
    return h, jumps, observables, a, b


def evolve(h, jumps, initial, times, observables, dims=None, rtol=2e-8, atol=2e-10):
    extra = None  # single-mode calculations can simply save their small matrices
    if dims is not None:
        extra = lambda t, rho: dq.to_jax(dq.ptrace(rho, 0))  # keep memory, discard buffer
    result = dq.mesolve(h, jumps, initial, jnp.asarray(times), exp_ops=observables,
                       method=dq.method.Tsit5(rtol=rtol, atol=atol, max_steps=2_000_000),
                       save_states=dims is None, save_extra=extra)  # sparse operators accelerate the rhs
    values = np.asarray(result.expects)  # this also waits for the actual computation to finish
    states = np.asarray(result.extra if dims is not None else dq.to_numpy(result.states))
    return {"t": np.asarray(times), "e": values, "rho": states, "final": result.final_state}


def run_key(h, jumps, initial, times, ops, rtol, atol):
    key = hashlib.sha256(f"memory-ptrace-v1:Tsit5:{rtol}:{atol}:{versions}".encode())
    for value in [h, *jumps, initial, *ops]:
        array = np.asarray(dq.to_numpy(value))  # hash actual operators, not just a descriptive filename
        key.update(str((array.shape, array.dtype)).encode())
        key.update(array.tobytes())
    key.update(np.asarray(times, dtype=np.float64).tobytes())
    folder = artifacts / "checkpoints"
    folder.mkdir(exist_ok=True)
    return folder / f"{key.hexdigest()}.npz"


def save_run(path, run):
    np.savez_compressed(path, t=run["t"], e=run["e"], rho=run["rho"], final=dq.to_numpy(run["final"]))


def full_run(na, nb, times, amplitude=alpha, seed=0.0, even=False, loss=ka, drive=0.0,
             rtol=2e-8, atol=2e-10):
    h, jumps, ops, a, b = two_mode(na, nb, amplitude, loss, drive)  # one model for every experiment
    memory = cat_ket(na, amplitude) if even else dq.coherent(na, seed)
    initial = dq.tensor(memory, dq.fock(nb, 0))  # buffer starts empty
    path = run_key(h, jumps, initial, times, ops, rtol, atol)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:  # optional speed-up; no pickle or hidden dependencies
            run = {name: saved[name] for name in ["t", "e", "rho"]}
            run["final"] = dq.asqarray(saved["final"], dims=(na, nb))
            return run
    run = evolve(h, jumps, initial, times, ops, (na, nb), rtol, atol)
    save_run(path, run)  # delete artifacts/checkpoints to force every solve to run again
    return run


def checks(run):
    rho = run["rho"]  # reduced states still must be valid density matrices
    trace_error = np.max(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1))
    herm_error = np.max(np.abs(rho - rho.swapaxes(-1, -2).conj()))
    minimum_eigenvalue = np.linalg.eigvalsh(rho).min()  # negative beyond solver error is a warning
    return {"trace error": trace_error, "hermiticity error": herm_error, "min eigenvalue": minimum_eigenvalue}


def plot_wigners(states, titles, filename):
    grid = np.linspace(-3.8, 3.8, 151)  # axes are real and imaginary coherent amplitude
    fig, axes = plt.subplots(1, len(states), figsize=(3.1 * len(states), 2.8), constrained_layout=True)
    for axis, state, title in zip(np.atleast_1d(axes), states, titles):
        w = qt.wigner(qt.Qobj(state), grid, grid, g=2)  # g=2 means x=real(alpha), p=imag(alpha)
        axis.contourf(grid, grid, w, levels=np.linspace(-2 / np.pi, 2 / np.pi, 65), cmap="RdBu_r", extend="both")
        axis.set(title=title, xlabel="real amplitude", ylabel="imaginary amplitude", aspect="equal")
    fig.savefig(artifacts / filename, bbox_inches="tight")  # the notebook also retains the picture
    plt.show()


def table(frame, name):
    frame.to_csv(artifacts / f"{name}.csv", index=False)  # enough precision for independent replotting
    display(frame)


def finish_figure(fig, name):
    fig.tight_layout()  # don't cut off axis labels
    fig.savefig(artifacts / f"{name}.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# 1.0a — one photon at a time
#
# here the whole state stays coherent: its amplitude is
# $2\exp[-(\kappa/2+i\Delta)t]$. squaring its magnitude gives the predicted
# photon number. this is a useful check on units, normalization, and the solver.

# %%
n = 30  # the warm-up cutoff requested in the prompt
c = dq.destroy(n)  # a single memory mode for now
number = c.dag() @ c  # counts photons
warm_t = np.linspace(0, 1, 151)  # one microsecond
warm = evolve(2 * np.pi * number, [np.sqrt(2 * np.pi) * c], dq.coherent(n, 2), warm_t, [number, c])
analytic_n = 4 * np.exp(-2 * np.pi * warm_t)  # the loss rate is angular, as specified
analytic_a = 2 * np.exp(-(np.pi + 2j * np.pi) * warm_t)  # rotating and shrinking
warm_error = np.max(np.abs(warm["e"][0].real - analytic_n))
print(f"maximum photon-number error: {warm_error:.3e}")
assert warm_error < 2e-6  # fail visibly if an elementary case is wrong
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(warm_t, warm["e"][0].real, label="master equation")
axes[0].plot(warm_t, analytic_n, "--", label="4 exp(-κt)")
axes[0].set(xlabel="time (μs)", ylabel="mean photons")
axes[0].legend()
axes[1].plot(warm["e"][1].real, warm["e"][1].imag, label="simulation")
axes[1].plot(analytic_a.real, analytic_a.imag, "--", label="analytic amplitude")
axes[1].set(xlabel="real ⟨a⟩", ylabel="imaginary ⟨a⟩", aspect="equal")
finish_figure(fig, "01_single_photon_loss")
plot_wigners(warm["rho"][[0, 30, 75, 150]], [f"t = {warm_t[i]:.2f} μs" for i in [0, 30, 75, 150]], "02_shrinking_coherent.png")

# %% [markdown]
# 1.0b — losing pairs leaves an odd photon behind
#
# $a^2$ annihilates both $|0\rangle$ and $|1\rangle$. more importantly, it never
# changes even/odd parity. every initially odd fock population can cascade down
# to $|1\rangle$, not just the population that started there. so the limiting
# photon number is the *total initial odd probability*,
# $(1-e^{-2|\alpha_0|^2})/2$, not $P(n=1)$.

# %%
pair_t = np.linspace(0, 2, 201)  # enough time for all higher fock levels to empty
pair = evolve(0 * number, [np.sqrt(k2) * c @ c], dq.coherent(n, 2), pair_t, [number, dq.parity(n)])
odd_probability = (1 - np.exp(-8)) / 2  # coherent-state poisson distribution, summed over odd n
print(f"final n = {pair['e'][0, -1].real:.9f}; total odd probability = {odd_probability:.9f}")
assert abs(pair["e"][0, -1].real - odd_probability) < 2e-6
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(pair_t, pair["e"][0].real, label="two-photon loss")
ax.axhline(odd_probability, ls="--", color="black", label="initial odd probability")
ax.set(xlabel="time (μs)", ylabel="mean photons")
ax.legend()
finish_figure(fig, "03_pair_loss")

# %% [markdown]
# 1.0c — add the drive inside the two-photon jump
#
# if $\kappa_a=0$, multiplying $\kappa_2$ multiplies the *entire* generator, so
# changing $\kappa_2$ is exactly rescaling time. with fixed nonzero $\kappa_a$,
# that statement is only approximate, and the final photon numbers can differ
# slightly too. the steady state is not exactly a pure even cat under loss.

# %%
inflate_t = np.linspace(0, 2, 401)  # resolves the early overshoot in the next task too
effective = {}  # keep all three runs for later comparisons
for factor in [0.25, 1.0, 4.0]:
    l2 = np.sqrt(factor * k2) * (c @ c - alpha**2 * dq.eye(n))
    effective[factor] = evolve(0 * number, [l2, np.sqrt(ka) * c], dq.fock(n, 0), inflate_t, [number, dq.parity(n)])
fig, ax = plt.subplots(figsize=(6, 3))
for factor, run in effective.items():
    ax.plot(inflate_t, run["e"][0].real, label=f"κ₂ × {factor:g}")
ax.axhline(alpha**2, ls=":", color="black", label="α²")
ax.set(xlabel="time (μs)", ylabel="memory photons")
ax.legend()
finish_figure(fig, "04_effective_stabilization")
plot_wigners(effective[1.0]["rho"][[0, 20, 80, 400]], [f"t = {inflate_t[i]:.2f} μs" for i in [0, 20, 80, 400]], "05_effective_cat.png")

# %% [markdown]
# 1.1 and 1.2 — let the buffer be a real mode, and move both ceilings
#
# the workshop hint was to vary the memory hilbert-space truncation too.
# here each axis is swept separately. a tiny final buffer occupation does not
# excuse cutting off its earlier transient. the reference is the largest tested
# pair, not a claim to have simulated an infinite oscillator.
#
# the declared observable tolerance is 0.2% relative to that reference, for both
# final photon number and transient peak. tails are a separate diagnostic:
# a small tail alone is not a convergence proof. a 0.005 μs output grid locates
# the peak; a finer-grid check below tests that choice.

# %%
cutoff_pairs = [(na, 12) for na in [16, 20, 24, 32, 40]]  # memory sweep with buffer held fixed
cutoff_pairs += [(40, nb) for nb in [4, 6, 8, 10, 16]]  # buffer sweep; (40,12) already exists
cutoff_pairs += [(24, 6)]  # the suggested starting point deserves a direct comparison
inflations = {}  # keep reduced states, not all the large joint density matrices
cutoff_rows = []
for na, nb in cutoff_pairs:
    start = time.perf_counter()  # includes compilation when this dimension is new
    run = full_run(na, nb, inflate_t)
    inflations[na, nb] = run
    cutoff_rows.append({"na": na, "nb": nb, "final n": run["e"][0, -1].real,
                        "peak n": run["e"][0].real.max(), "peak buffer n": run["e"][1].real.max(),
                        "max memory tail": run["e"][5].real.max(), "max buffer tail": run["e"][6].real.max(),
                        "seconds": time.perf_counter() - start})
    print(f"cutoffs ({na}, {nb}) finished in {cutoff_rows[-1]['seconds']:.1f}s", flush=True)
convergence = pd.DataFrame(cutoff_rows)
reference_row = convergence.query("na == 40 and nb == 16").iloc[0]
for quantity in ["final n", "peak n"]:
    convergence[f"relative error: {quantity}"] = abs(convergence[quantity] / reference_row[quantity] - 1)
table(convergence, "cutoff_convergence")
observable_tolerance = 0.002  # 0.2%, stated before selecting a cutoff
for column, fixed_column, fixed_value in [("na", "nb", 12), ("nb", "na", 40)]:
    subset = convergence[convergence[fixed_column] == fixed_value].sort_values(column)
    for quantity in ["final n", "peak n"]:
        passing = subset[f"relative error: {quantity}"].to_numpy() < observable_tolerance
        stable = np.logical_and.accumulate(passing[::-1])[::-1]  # every larger tested value must pass too
        print(f"smallest tested {column} passing {quantity}: {subset.loc[stable, column].min()}")

# %%
inflation = inflations[40, 16]  # safest tested run for figures and cat-state diagnostics
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
axes[0, 0].plot(inflate_t, inflation["e"][0].real, label="full, 40 × 16")
axes[0, 0].plot(inflate_t, inflations[24, 6]["e"][0].real, ":", label="starting cutoffs, 24 × 6")
axes[0, 0].plot(inflate_t, effective[1.0]["e"][0].real, "--", label="eliminated buffer")
axes[0, 0].axhline(alpha**2, ls=":", color="black")
axes[0, 0].set(ylabel="memory photons")
axes[0, 0].legend(fontsize=8)
axes[0, 1].plot(inflate_t, inflation["e"][1].real)
axes[0, 1].set(ylabel="buffer photons")
for pair, style in [((24, 6), ":"), ((40, 16), "-")]:
    run = inflations[pair]
    axes[1, 0].semilogy(inflate_t, np.maximum(run["e"][5].real, 1e-16), style, label=str(pair))
    axes[1, 1].semilogy(inflate_t, np.maximum(run["e"][6].real, 1e-16), style, label=str(pair))
axes[1, 0].set(ylabel="top two memory populations")
axes[1, 1].set(ylabel="top buffer population")
for axis in axes.flat:
    axis.set(xlabel="time (μs)")
axes[1, 0].legend()
finish_figure(fig, "06_full_inflation_and_tails")
plot_wigners(inflation["rho"][[0, 30, 65, 200, 400]], [f"t = {inflate_t[i]:.3g} μs" for i in [0, 30, 65, 200, 400]], "07_full_cat.png")
print(f"g2/kb = {g2/kb:.4f}; eps_d/kb = {eps_d/kb:.4f}")
print(f"final n differs from alpha² by {100*(inflation['e'][0,-1].real/alpha**2-1):.3f}%")
print("reference reduced-state checks:", checks(inflation))
assert abs(inflation["e"][0, -1].real / alpha**2 - 1) < 0.02
assert checks(inflation)["trace error"] < 1e-7

# %% [markdown]
# the eliminated-buffer assumptions are not satisfied: $g_2/\kappa_b\simeq0.293$
# and $\epsilon_d/\kappa_b\simeq1.294$. the buffer stores excitation and feeds it
# back before it leaks away. linearizing near either well gives poles
#
# $$\lambda_\pm=-\kappa_b/4\pm\sqrt{(\kappa_b/4)^2-(2g_2\alpha)^2}.$$
#
# these are complex here: ringing is physical. also, with memory loss the
# semiclassical stationary value is $|a|^2=\alpha^2-\kappa_a\kappa_b/(8g_2^2)$,
# not precisely $\alpha^2$. settling of photon number is not settling of parity.

# %%
fine = full_run(32, 12, np.linspace(0, 2, 801), rtol=4e-9, atol=4e-11)  # tighter tolerance and twice the output resolution
coarse = inflations[32, 12]
print(f"peak change with tighter integration/finer output: {abs(fine['e'][0].real.max()-coarse['e'][0].real.max()):.3e}")
print(f"maximum matched-time n change: {np.max(abs(fine['e'][0, ::2]-coarse['e'][0])):.3e}")
late = inflate_t >= 1.5  # check an interval, not just the last point
print(f"late-time n variation / alpha²: {np.ptp(inflation['e'][0, late].real)/alpha**2:.3e}")
print(f"semiclassical loss-corrected n: {alpha**2-ka*kb/(8*g2**2):.6f}")

# %% [markdown]
# 1.3 — what state did inflation actually make?
#
# $[P,H]=0$ and the buffer jump commutes with memory parity, but $Pa=-aP$.
# without memory loss, vacuum's even parity stays even. with memory loss,
# $d\langle P\rangle/dt=-2\kappa_a\langle n_aP\rangle$; it is not conserved.
#
# i use squared pure-state fidelity $F_+=\langle C_+|\rho_a|C_+\rangle$.
# the equal incoherent mixture of the two coherent states has parity
# $e^{-2\alpha^2}$ and $F_+=(1+e^{-2\alpha^2})/2$. exceeding those baselines,
# together with the negative central wigner fringes, distinguishes coherence
# from merely seeing two lobes. there is no sharp universal 'cat lifetime'.

# %%
even40 = np.asarray(dq.to_numpy(cat_ket(40, alpha))).ravel()
cat_fidelity = np.einsum("i,tij,j->t", even40.conj(), inflation["rho"], even40).real
purity = np.einsum("tij,tji->t", inflation["rho"], inflation["rho"]).real
mixture_fidelity = (1 + np.exp(-2 * alpha**2)) / 2
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(inflate_t, inflation["e"][4].real, label="parity")
ax.plot(inflate_t, cat_fidelity, label="even-cat fidelity")
ax.plot(inflate_t, purity, label="memory purity")
ax.axhline(mixture_fidelity, ls=":", color="black", label="incoherent-mixture fidelity")
ax.set(xlabel="time (μs)", ylabel="dimensionless")
ax.legend(fontsize=8)
finish_figure(fig, "08_which_cat")
best_cat_index = int(np.argmax(cat_fidelity))
print(f"best even-cat fidelity = {cat_fidelity[best_cat_index]:.6f} at t = {inflate_t[best_cat_index]:.3f} μs")
coherence_t = np.linspace(0, 8, 401)  # long enough to cross an explicit 1/e coherence threshold
long_cat = full_run(28, 6, coherence_t, even=True)  # start from a prepared cat to isolate its lifetime
baseline_parity = np.exp(-2 * alpha**2)
crossings = np.flatnonzero(long_cat["e"][4].real - baseline_parity < (1-baseline_parity)/np.e)
print(f"prepared-cat parity 1/e time: {coherence_t[crossings[0]]:.3f} μs; simple prediction: {1/(2*ka*alpha**2):.3f} μs")
zero_loss = full_run(28, 10, inflate_t, loss=0.0)  # remove only the term that breaks parity
print(f"maximum parity drift with ka=0: {np.max(abs(zero_loss['e'][4]-1)):.3e}")
assert np.max(abs(zero_loss["e"][4] - 1)) < 1e-7

# %% [markdown]
# for a definite logical bit, a coherent seed breaks the sign symmetry *before*
# inflation. this is not just initializing the final target: the seed sweep
# below starts with amplitudes below 2.1 and lets the same stabilizer grow them.
# the cost is a calibrated, phase-referenced memory displacement pulse (and its
# control port), not measurement postselection. the seed preparation itself is
# idealized as an instantaneous displacement here.

# %%
seed_rows = []
seed_runs = {}
target28 = np.asarray(dq.to_numpy(dq.coherent(28, alpha))).ravel()
for seed in [0.0, 0.5, 1.0, 1.5]:
    run = full_run(28, 10, inflate_t, seed=seed)
    seed_runs[seed] = run
    final_f = np.vdot(target28, run["rho"][-1] @ target28).real  # no square root convention ambiguity
    seed_rows.append({"seed amplitude": seed, "initial photons": seed**2,
                      "final real a": run["e"][2, -1].real, "final fidelity to +alpha": final_f})
table(pd.DataFrame(seed_rows), "logical_state_preparation")
fig, ax = plt.subplots(figsize=(6, 3))
for seed, run in seed_runs.items():
    ax.plot(inflate_t, run["e"][2].real, label=f"seed {seed:g}")
ax.axhline(alpha, color="black", ls=":")
ax.set(xlabel="time (μs)", ylabel="real ⟨a⟩")
ax.legend()
finish_figure(fig, "09_seeded_preparation")

# %% [markdown]
# 2.1 — measure a slow bit decay, without mistaking it for a numerical floor
#
# the prompt calls the decay of $\mathrm{Re}\langle a\rangle$ $\Gamma_X$.
# a symmetric classical bit with single-event rate $r$ has imbalance
# $e^{-2rt}$, so the prompt's $\Gamma_X=2r$. several papers quote $r$ instead.
# the same factor-of-two issue appears for parity and phase flips.
#
# short-time physics uses dynamiqs. for the very long, stiff small-cat decays i
# also use qutip's sparse master-equation solver, with the *same* matrices, and
# compare the fitted rates to slow eigenvalues of the full two-mode generator.
# this is not adiabatic elimination. a spectral residual, cutoff changes, and
# fit-window changes are reported; a pretty straight line is not enough.

# %%
def sparse_model(na, nb, amplitude, drive=0.0):
    h, jumps, ops, _, _ = two_mode(na, nb, amplitude, drive=drive)
    dims = [[na, nb], [na, nb]]  # keep tensor information for partial traces
    convert = lambda op: qt.Qobj(dq.to_numpy(op), dims=dims).to("csr")
    return convert(h), [convert(op) for op in jumps], [convert(op) for op in ops]


def slow_bit_mode(na, nb, amplitude, drive=0.0):
    h, jumps, ops = sparse_model(na, nb, amplitude, drive)
    generator = qt.liouvillian(h, jumps).data.as_scipy().tocsc()  # d² by d², but sparse
    vals, vecs = eigs(generator, k=6, sigma=1e-6, tol=1e-10, v0=np.ones(generator.shape[0], dtype=complex))
    d = na * nb
    candidates = []
    for i, value in enumerate(vals):
        rho_mode = vecs[:, i].reshape((d, d), order="F")  # qutip stacks columns
        trace = abs(np.trace(rho_mode))  # stationary mode has nonzero trace; decay modes do not
        overlap = abs(np.trace(ops[2].full() @ rho_mode))  # does this mode actually change the logical bit?
        if trace < 1e-4 and overlap > 1e-3 and value.real < -1e-12:
            candidates.append((value.real, i, overlap))
    if not candidates:
        raise RuntimeError("no resolved bit mode; enlarge the spectral search or precision")
    _, index, overlap = max(candidates)  # slowest decaying mode visible to a
    value = vals[index]
    residual = np.linalg.norm(generator @ vecs[:, index] - value * vecs[:, index])
    return {"rate": -value.real, "imaginary part": value.imag, "residual": residual, "a overlap": overlap}


def fit_decay(times, signal, start=0.0, offset=False):
    keep = np.asarray(times) >= start  # discard the fast adjustment to the metastable manifold
    t = np.asarray(times)[keep]
    y = np.asarray(signal).real[keep]
    scale_t = t[-1] - t[0]  # scale time so the optimizer handles very slow rates well
    x = (t - t[0]) / scale_t
    if offset:
        model = lambda x, amp, decay, floor: amp * np.exp(-decay * x) + floor
        guess = [y[0] - y[-1], 2.0, y[-1]]
        bounds = ([-2, 0, -1], [2, 100, 1])
    else:
        model = lambda x, amp, decay: amp * np.exp(-decay * x)
        guess = [y[0], 1.0]
        bounds = ([-10, 0], [10, 100])
    parameters, covariance = curve_fit(model, x, y, p0=guess, bounds=bounds, maxfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    prediction = model(x, *parameters)
    return {"rate": parameters[1] / scale_t, "rmse": np.sqrt(np.mean((prediction - y)**2)),
            "floor": parameters[2] if offset else 0.0, "fit t": t, "fit y": prediction}


def sparse_decay(na, nb, amplitude, times):
    h, jumps, ops = sparse_model(na, nb, amplitude)
    initial = qt.tensor(qt.coherent(na, amplitude), qt.basis(nb, 0))
    result = qt.mesolve(h, initial, times, jumps, e_ops=ops,
                       options={"method": "bdf", "rtol": 2e-8, "atol": 2e-10,
                                "nsteps": 2_000_000, "store_states": False})
    return np.asarray(result.expect)  # includes tails over the entire long run

# %%
bit_rows = []
bit_traces = {}
for size in [0.6, 0.8, 1.0, 1.2, 1.4]:
    mode = slow_bit_mode(18, 4, size)  # low buffer occupation here, unlike vacuum inflation
    end = min(100.0, 1.5 / mode["rate"])  # finite early decay is resolvable; don't simulate experimental seconds
    times = np.linspace(0, end, 161)
    signal = sparse_decay(18, 4, size, times)
    fit = fit_decay(times, signal[2].real, start=max(2.0, 0.05 * end))
    later = fit_decay(times, signal[2].real, start=0.3 * end)  # test sensitivity to the fitting window
    check_mode = slow_bit_mode(22, 5, size)  # rates, not only photon numbers, need cutoff tests
    bit_rows.append({"alpha": size, "alpha²": size**2, "fit rate": fit["rate"], "spectral rate": mode["rate"],
                     "larger cutoff rate": check_mode["rate"], "fit rmse": fit["rmse"],
                     "window relative change": abs(later["rate"] / fit["rate"] - 1),
                     "cutoff relative change": abs(check_mode["rate"] / mode["rate"] - 1),
                     "spectral residual": mode["residual"], "observed fraction of lifetime": end * mode["rate"],
                     "memory tail": signal[5].real.max(), "buffer tail": signal[6].real.max()})
    bit_traces[size] = (times, signal, fit)
    print(f"alpha={size:.1f}, fitted Γx={fit['rate']:.6g}/μs", flush=True)
bit_table = pd.DataFrame(bit_rows)
table(bit_table, "bit_flip_rates")
log_slope, log_intercept = np.polyfit(bit_table["alpha²"], np.log(bit_table["fit rate"]), 1)
print(f"finite-range suppression exponent gamma = {-log_slope:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for size, (times, signal, fit) in bit_traces.items():
    axes[0].plot(times, signal[2].real / size, label=f"α={size:g}")
    axes[0].plot(fit["fit t"], fit["fit y"] / size, "--", color="black", lw=0.6)
axes[0].set(xlabel="time (μs)", ylabel="real ⟨a⟩ / α")
axes[0].legend(fontsize=8)
axes[1].semilogy(bit_table["alpha²"], bit_table["fit rate"], "o", label="time-domain fit")
axes[1].semilogy(bit_table["alpha²"], bit_table["spectral rate"], "x", label="slow eigenvalue")
axes[1].semilogy(bit_table["alpha²"], np.exp(log_intercept + log_slope * bit_table["alpha²"]), "--", label=f"γ={-log_slope:.2f}")
axes[1].set(xlabel="α²", ylabel="Γx (1/μs)")
axes[1].legend(fontsize=8)
finish_figure(fig, "10_bit_flip_scaling")
cross_t = np.linspace(0, 4, 101)  # independent solver comparison on a common short window
cross_dq = full_run(18, 4, cross_t, amplitude=1.0, seed=1.0)
cross_qt = sparse_decay(18, 4, 1.0, cross_t)
cross_error = np.max(abs(cross_dq["e"] - cross_qt))
print(f"dynamiqs versus qutip: maximum observable difference = {cross_error:.3e}")
assert cross_error < 2e-5

# %% [markdown]
# the naive matrix element is
# $|\langle-\alpha|a|\alpha\rangle|^2=\alpha^2e^{-4\alpha^2}$.
# it ignores excursions outside the code space and the way stabilization
# returns that leaked probability. [dubovitskii, 2025](https://arxiv.org/abs/2407.17299)
# finds a second-order single-loss contribution proportional to
# $(\kappa_a^2/2\kappa_2)e^{-2\alpha^2}$ in the eliminated model's *event-rate*
# convention. that term can dominate even without thermal photons.
#
# so neither an exponent of exactly 2 nor a pure exponential over all sizes is
# required here. the prefactor, finite fitting interval, and two-mode dynamics
# matter. this also corrects the draft prompt's suggestion that zero temperature
# alone explains every exponent above 2. the small-cat fit is not an extrapolation
# of hardware lifetimes at α=2.1.

# %% [markdown]
# 2.2 — phase flips are fast enough to watch directly
#
# each lost photon changes parity. approximately independent jumps at rate
# $\kappa_a\alpha^2$ give $\Gamma_Z=2\kappa_a\alpha^2$ for the parity decay.
# i fit an offset as well: at smaller cats the limiting parity is not exactly
# zero. imposing a zero floor there biases the fitted rate.

# %%
phase_rows = []
phase_traces = {}
phase_t = np.linspace(0, 12, 241)  # includes several lifetimes at the physical target
for size in [1.2, 1.5, 1.8, 2.1, 2.4]:
    run = full_run(32, 5, phase_t, amplitude=size, even=True)
    fit = fit_decay(phase_t, run["e"][4].real, start=0.3, offset=True)
    phase_traces[size] = run
    phase_rows.append({"alpha": size, "alpha²": size**2, "fitted Γz": fit["rate"],
                       "2 ka alpha²": 2 * ka * size**2, "fitted parity floor": fit["floor"],
                       "fit rmse": fit["rmse"], "memory tail": run["e"][5].real.max(),
                       "buffer tail": run["e"][6].real.max()})
phase_table = pd.DataFrame(phase_rows)
table(phase_table, "phase_flip_rates")
phase_slope, phase_intercept = np.polyfit(phase_table["alpha²"], phase_table["fitted Γz"], 1)
print(f"fitted slope {phase_slope:.6f}; predicted 2ka = {2*ka:.6f}; intercept {phase_intercept:.3g}")
phase_check = full_run(36, 7, phase_t, amplitude=2.4, even=True)  # largest cat is the demanding endpoint
phase_check_fit = fit_decay(phase_t, phase_check["e"][4].real, start=0.3, offset=True)
print(f"largest-cat Γz cutoff change: {abs(phase_check_fit['rate']/phase_rows[-1]['fitted Γz']-1):.3e}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(phase_t, phase_traces[2.1]["e"][4].real, label="full model, α=2.1")
axes[0].plot(phase_t, np.exp(-2 * ka * alpha**2 * phase_t), "--", label="simple parity prediction")
axes[0].set(xlabel="time (μs)", ylabel="parity")
axes[0].legend(fontsize=8)
axes[1].plot(phase_table["alpha²"], phase_table["fitted Γz"], "o", label="fitted")
axes[1].plot(phase_table["alpha²"], phase_table["2 ka alpha²"], "--", label="2κa α²")
axes[1].set(xlabel="α²", ylabel="Γz (1/μs)")
axes[1].legend()
finish_figure(fig, "11_phase_flip_scaling")

# %% [markdown]
# 2.3 — ask the buffer which well the memory occupies
#
# a weak real memory drive pushes the memory in its imaginary quadrature.
# the two wells convert that displacement into opposite real buffer amplitudes.
# the buffer also has a small *common imaginary* displacement from memory loss;
# it is the difference between the two output signals that carries the bit.
# 'steady' here means a metastable measurement plateau, before rare bit flips
# eventually erase the initial sign. it is not two different infinite-time
# steady states of a mixing lindblad equation.

# %%
weak_drive = 0.1 * g2  # comfortably smaller than the exchange coupling
read_t = np.linspace(0, 3, 601)  # resolves the underdamped oscillation
read_plus = full_run(28, 6, read_t, seed=alpha, drive=weak_drive)
read_minus = full_run(28, 6, read_t, seed=-alpha, drive=weak_drive)
contrast = read_plus["e"][3] - read_minus["e"][3]  # common buffer offset cancels
linear_contrast = -weak_drive / (alpha * g2)  # twice the single-well displacement
plateau = np.mean(contrast[read_t > 2.0].real)
outside = np.flatnonzero(abs(contrast.real - plateau) > 0.02 * abs(plateau))
settling_time = read_t[outside[-1] + 1] if outside[-1] + 1 < len(read_t) else np.nan
print(f"contrast plateau {plateau:.6f}; linear estimate {linear_contrast:.6f}")
print(f"2% settling time within the 3 μs window: {settling_time:.3f} μs")
omega = 2 * g2 * alpha  # local memory-buffer exchange frequency
print(f"linearized ringing period: {2*np.pi/np.sqrt(omega**2-(kb/4)**2):.3f} μs")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(read_t, read_plus["e"][3].real, label="initial +α")
axes[0].plot(read_t, read_minus["e"][3].real, label="initial -α")
axes[0].set(xlabel="time (μs)", ylabel="real ⟨b⟩")
axes[0].legend()
axes[1].plot(read_t, contrast.real, label="full-model difference")
axes[1].axhline(linear_contrast, ls="--", color="black", label="linearized plateau")
axes[1].set(xlabel="time (μs)", ylabel="readout contrast")
axes[1].legend(fontsize=8)
finish_figure(fig, "12_buffer_readout")

# %% [markdown]
# 2.4a — put the drive back into the circuit
#
# one concrete implementation is a capacitive memory port:
# $H_{\mathrm{port}}=2\epsilon_Z(t)\cos(\omega_a t+\theta)(a+a^\dagger)$.
# in the rotating frame its resonant part is
# $\epsilon_Z(t)(e^{i\theta}a+e^{-i\theta}a^\dagger)$; there are also terms
# oscillating at $2\omega_a$. choosing $\theta=0$ gives the requested drive.
# this is an explicit extra control port, just as the problem already permits a
# separate capacitive buffer drive. it is not secretly another flux pump.
#
# an alternative flux pump at $\omega_a$ has additional resonant cubic terms.
# writing its amplitude as $\epsilon_r$, the first-order rotating-wave result is
#
# $$E_J\epsilon_r\varphi_a\left[\left(1-\frac{\varphi_a^2}{2}
# -\frac{\varphi_b^2}{2}\right)(a+a^\dagger)
# -\frac{\varphi_a^2}{2}(a^\dagger a^2+a^{\dagger2}a)
# -\varphi_b^2(a+a^\dagger)n_b\right].$$
#
# this follows from the resonant part of $X_a^3$, which is
# $3(a^\dagger a^2+a)+\mathrm{h.c.}$, and the dc part of $X_b^2=2n_b+1$.
# stronger flux readout therefore does not scale a clean linear drive alone.
# the numerical protocol below chooses the capacitive implementation to avoid
# those specific resonant cubic readout terms.
#
# there is a sign inconsistency worth making explicit in the supplied draft:
# at its stated +π/2 differential bias, the cubic term of
# $+2E_J\epsilon_p\cos(\omega_pt)\sin\varphi$ gives *negative* resonant g₂.
# the quoted positive formula is its magnitude. reversing the pump phase gives
# our positive-g₂ convention; the buffer-drive phase must be calibrated with it.
# the workshop slide's opposite differential bias also reverses this sign.
#
# for the stabilizing pump, the cubic rotating-wave term is the intended
# $a^2b^\dagger+\mathrm{h.c.}$, but fifth and higher odd orders dress its fock-state
# matrix elements. at fifth order the forward matrix element is multiplied by
#
# $$1-\frac{\varphi_a^2}{6}(2n_a-1)-\frac{\varphi_b^2}{2}(n_b+1).$$
#
# below i keep the *entire sine*, then retain exactly resonant matrix elements.
# this checks the small-phase expansion without resolving GHz time steps. it
# does not include second-order virtual off-resonant shifts, pump-induced changes
# to dissipators, thermal noise, or a real measurement amplifier. those are
# limitations, not effects this rotating-wave calculation has verified away.

# %%
wa, wb, ej = 2 * np.pi * np.array([4000.0, 4800.0, 40000.0])  # ghz converted to mhz, then angular units
phia, phib = 0.06, 0.3  # zero-point phase fluctuations from the circuit table
eps_p = 2 * g2 / (ej * phia**2 * phib)  # magnitude; use the pump phase discussed above
print(f"required pump amplitude = {eps_p:.8f} rad; pump frequency / 2π = {(2*wa-wb)/(2*np.pi):.1f} MHz")


def sine_exchange(na, nb, calibrate=True, order=None):
    aa = qt.tensor(qt.destroy(na + 8), qt.qeye(nb + 8))  # padding avoids boundary errors in the sine
    bb = qt.tensor(qt.qeye(na + 8), qt.destroy(nb + 8))
    phase = phia * (aa + aa.dag()) + phib * (bb + bb.dag())
    if order is None:
        da = qt.displace(na + 8, 1j * phia).full()[:na, :na]  # exp(i phi_a X_a)
        db = qt.displace(nb + 8, 1j * phib).full()[:nb, :nb]
        displacement = np.kron(da, db)  # commuting mode operators factorize exactly
        sine = (displacement - displacement.conj().T) / (2j)
    else:
        series = phase - phase**3 / 6  # cubic comparison
        if order >= 5:
            series = series + phase**5 / 120
        keep = np.array([i * (nb + 8) + j for i in range(na) for j in range(nb)])
        sine = series.full()[np.ix_(keep, keep)]  # crop only after expanding in the padded space
    energy = np.array([wa * i + wb * j for i in range(na) for j in range(nb)])
    resonant = np.isclose(abs(energy[:, None] - energy[None, :]), 2 * wa - wb, atol=1e-6, rtol=0)
    exchange = -ej * eps_p * sine * resonant  # minus = chosen π pump-phase shift; cos supplies 1/2
    raw_g = exchange[1, 2 * nb].real / np.sqrt(2)  # |2,0> -> |0,1> low-occupancy calibration
    if calibrate:
        exchange = exchange * g2 / raw_g  # separate a recalibratable g shift from number dependence
    return dq.asqarray(exchange, dims=(na, nb), layout=dq.dia), raw_g


exact_exchange, raw_g = sine_exchange(28, 6, calibrate=False)
cubic_exchange, _ = sine_exchange(28, 6, calibrate=False, order=3)
fifth_exchange, _ = sine_exchange(28, 6, calibrate=False, order=5)
entry_rows = []
for memory_n, buffer_n in [(2, 0), (5, 0), (10, 0), (5, 1), (10, 3)]:
    row, column = (memory_n - 2) * 6 + buffer_n + 1, memory_n * 6 + buffer_n
    cubic = dq.to_numpy(cubic_exchange)[row, column].real
    entry_rows.append({"initial memory n": memory_n, "initial buffer n": buffer_n,
                       "exact/cubic": dq.to_numpy(exact_exchange)[row, column].real / cubic,
                       "fifth/cubic": dq.to_numpy(fifth_exchange)[row, column].real / cubic})
table(pd.DataFrame(entry_rows), "circuit_matrix_elements")
print(f"uncalibrated low-occupancy coupling shift: {100*(raw_g/g2-1):.3f}%")

# %% [markdown]
# 2.4b — measure the contrast/error trade-off
#
# increasing εz increases the coherent buffer separation but also perturbs the
# protected manifold. the sweep uses α=1.2 so Γx remains independently resolvable;
# it does not pretend these are the target-size rates. the physical-size pulse
# comparison follows separately. all points use the same loss and exchange rates.

# %%
drive_rows = []
for fraction in [0.0, 0.05, 0.1, 0.2, 0.4]:
    drive = fraction * g2
    mode = slow_bit_mode(18, 4, 1.2, drive=drive)
    larger = slow_bit_mode(22, 5, 1.2, drive=drive)
    run = full_run(22, 5, np.linspace(0, 3, 301), amplitude=1.2, seed=1.2, drive=drive)
    contrast_here = 2 * abs(np.mean(run["e"][3, -51:].real))  # exact sign symmetry gives the other branch
    drive_rows.append({"drive/g2": fraction, "contrast at 2.5–3 us": contrast_here,
                       "Γx": mode["rate"], "larger-cutoff Γx": larger["rate"],
                       "relative cutoff change": abs(larger["rate"] / mode["rate"] - 1),
                       "spectral residual": mode["residual"]})
drive_table = pd.DataFrame(drive_rows)
table(drive_table, "dc_readout_tradeoff")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(drive_table["drive/g2"], drive_table["contrast at 2.5–3 us"], "o-")
axes[0].set(xlabel="εz / g2", ylabel="buffer contrast, α=1.2")
axes[1].semilogy(drive_table["drive/g2"], drive_table["Γx"], "o-")
axes[1].set(xlabel="εz / g2", ylabel="Γx (1/μs), α=1.2")
finish_figure(fig, "13_dc_tradeoff")

# %% [markdown]
# 2.4c — an experiment: use the transient, then listen during ringdown
#
# near either well the real buffer signal obeys
# $\ddot b+(\kappa_b/2)\dot b+\Omega^2b=\mp\Omega\epsilon_Z(t)$,
# where $\Omega=2g_2\alpha$. this is an underdamped filter, not an instantaneous
# displacement. a receiver that averages a changing-sign signal can throw away
# information. a matched filter weights the recorded quadrature by its predicted
# time-dependent difference, including the passive ringdown after the drive ends.
#
# i compare a constant drive, a short pulse with passive listening, and a smooth
# resonant pulse. these are candidates, not assumed winners. they have the same
# maximum memory-drive amplitude and the same total observation/recovery time.
# a second comparison later rescales amplitudes to equal measured information.
#
# for ideal homodyne detection,
# $dY=2\sqrt{\eta\kappa_b}\mathrm{Re}\langle b\rangle dt+dW$.
# assuming shot-noise-limited, approximately coherent output, define
# $\mathcal I=\eta\kappa_b\int[\mathrm{Re}(b_+-b_-)]^2dt$.
# then $\mathrm{SNR}^2=4\mathcal I$ (separation over one standard deviation) and
# $P_{\mathrm{assignment}}\simeq\frac12\mathrm{erfc}(\sqrt{\mathcal I/2})$.
# this is a mean-signal information proxy, not a stochastic-trajectory validation
# of assignment error. it excludes excess output noise and amplifier effects.
#
# a finite pulse has no stationary Γx. after the drive, i let stabilization recover
# the state, then report the wrong orthogonal-codeword population and residual
# leakage separately. i do not rename temporary displacement as a logical flip.

# %%
def pulse_run(na, nb, waveform, total=3.0, amplitude=alpha, corrected=False, sign=1, tolerance=2e-10):
    h, jumps, ops, a, b = two_mode(na, nb, amplitude)
    if corrected:
        exchange, _ = sine_exchange(na, nb, calibrate=True)  # include all resonant odd-phase orders
        h = exchange - g2 * amplitude**2 * (b + b.dag())
    h = h + dq.modulated(waveform, a + a.dag(), discontinuity_ts=[np.pi / omega, 1.5])  # known pulse edges
    times = np.linspace(0, total, int(total * 400) + 1)  # 2.5 ns output grid, independently of solver steps
    initial = dq.tensor(dq.coherent(na, sign * amplitude), dq.fock(nb, 0))
    run = evolve(h, jumps, initial, times, ops, (na, nb), rtol=100*tolerance, atol=tolerance)
    even = np.asarray(dq.to_numpy(cat_ket(na, amplitude, 1))).ravel()
    odd = np.asarray(dq.to_numpy(cat_ket(na, amplitude, -1))).ravel()
    wrong = (even - sign * odd) / np.sqrt(2)  # orthogonal logical basis; coherent wells only approximate it
    rho = run["rho"][-1]
    p_wrong = np.vdot(wrong, rho @ wrong).real
    in_code = np.vdot(even, rho @ even).real + np.vdot(odd, rho @ odd).real
    difference = 2 * run["e"][3].real  # opposite branch checked explicitly below
    information = kb * cumulative_trapezoid(difference**2, times, initial=0)  # eta=1; scale linearly for other efficiencies
    run.update({"wrong": p_wrong, "leakage": 1-in_code, "information": information,
                "box information": kb * np.trapezoid(difference, times)**2 / total})
    return run


def waveform_family(kind, peak, duration=1.5):
    if kind == "constant":
        return lambda t: jnp.where(t < duration, peak, 0.0)  # explicit discontinuity tested with tighter tolerance
    if kind == "short pulse":
        return lambda t: jnp.where(t < np.pi / omega, peak, 0.0)  # roughly half a local exchange cycle
    if kind == "resonant":
        return lambda t: jnp.where(t < duration, peak * jnp.sin(np.pi*t/duration)**2 * jnp.cos(omega*t), 0.0)
    raise ValueError(kind)

# %%
candidate_peak = 0.5 * g2  # explicit common hardware amplitude limit
candidate_runs = {}
candidate_rows = []
for kind in ["constant", "short pulse", "resonant"]:
    run = pulse_run(28, 6, waveform_family(kind, candidate_peak))
    candidate_runs[kind] = run
    candidate_rows.append({"pulse": kind, "peak limit/g2": 0.5, "matched information": run["information"][-1],
                           "unweighted information": run["box information"], "wrong codeword": run["wrong"],
                           "residual leakage": run["leakage"], "information/wrong": run["information"][-1]/run["wrong"]})
table(pd.DataFrame(candidate_rows), "pulse_candidates")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for kind, run in candidate_runs.items():
    waveform = waveform_family(kind, candidate_peak)
    axes[0].plot(run["t"], np.asarray(jax.vmap(waveform)(jnp.asarray(run["t"])))/g2, label=kind)
    axes[1].plot(run["t"], 2 * run["e"][3].real, label=kind)
    axes[2].plot(run["t"], run["information"], label=kind)
axes[0].set(ylabel="εz(t) / g2")
axes[1].set(ylabel="buffer signal difference")
axes[2].set(ylabel="matched information, η=1")
for axis in axes:
    axis.set(xlabel="time (μs)")
axes[0].legend(fontsize=8)
finish_figure(fig, "14_transient_pulse_candidates")

# %% [markdown]
# same peak is useful engineering context, but not a fair information/error
# comparison by itself: the waveforms deliver different amounts of information.
# next i match the two informative long pulses to the same measured $\mathcal I$,
# keeping the same 3 μs total protocol and never exceeding the shared peak limit.
# amplitude rescaling uses the quadratic small-signal prediction and then checks
# the full nonlinear master equation again. no result is selected by ignoring
# unsuccessful candidates.

# %%
target_information = 0.8 * min(candidate_runs[k]["information"][-1] for k in ["constant", "resonant"])
matched_runs = {}
matched_amplitudes = {}
for kind in ["constant", "resonant"]:
    peak = candidate_peak * np.sqrt(target_information / candidate_runs[kind]["information"][-1])
    for iteration in range(2):
        run = pulse_run(28, 6, waveform_family(kind, peak))
        if iteration == 0:
            peak *= np.sqrt(target_information / run["information"][-1])  # one nonlinear correction, then a fresh solve
    assert peak <= candidate_peak * (1+1e-8)
    assert abs(run["information"][-1]/target_information-1) < 0.01  # explicit 1% equal-information criterion
    matched_runs[kind] = run
    matched_amplitudes[kind] = peak
idle_pulse = pulse_run(28, 6, lambda t: 0.0 * t)  # same 3 μs, no readout drive
matched_rows = []
for kind, run in matched_runs.items():
    matched_rows.append({"pulse": kind, "peak/g2": matched_amplitudes[kind]/g2,
                         "information": run["information"][-1], "wrong codeword": run["wrong"],
                         "excess over idle": run["wrong"]-idle_pulse["wrong"], "leakage": run["leakage"],
                         "effective 2p/T": 2*run["wrong"]/3, "gaussian assignment estimate, eta=1": 0.5*erfc(np.sqrt(run["information"][-1]/2))})
matched_table = pd.DataFrame(matched_rows)
table(matched_table, "equal_information_comparison")
relative_reduction = 1 - matched_runs["resonant"]["wrong"] / matched_runs["constant"]["wrong"]
print(f"recovered wrong-codeword reduction at matched information: {100*relative_reduction:.2f}%")
print(f"idle wrong-codeword population: {idle_pulse['wrong']:.3e}")
print("2p/T is a protocol comparison, not a stationary Γx measurement.")
opposite_pulse = pulse_run(28, 6, waveform_family("resonant", matched_amplitudes["resonant"]), sign=-1)
sign_error = np.max(abs(opposite_pulse["e"][3].real + matched_runs["resonant"]["e"][3].real))
print(f"explicit opposite-codeword symmetry error: {sign_error:.3e}")
assert sign_error < 1e-6

# %% [markdown]
# 2.4d — try to break the apparent improvement
#
# readout starts near a well, so it does not need the same buffer cutoff as
# vacuum inflation. that is a reason to test smaller cutoffs, not permission to
# assume they work. the checks below vary the memory and buffer separately, then
# increase both, tighten the solver, and add the calibrated sine-rwa correction.
# frequency mistuning tests whether the result depends on hitting one exact number.
# the carrier is still at ωa; these few-mhz envelope sidebands are far below GHz.

# %%
robust_rows = []
for label, na, nb, corrected, atol in [
    ("smaller memory", 24, 6, False, 2e-10),
    ("smaller buffer", 28, 4, False, 2e-10),
    ("larger cutoffs + tighter solver", 32, 8, False, 2e-11),
    ("calibrated full sine, rwa", 28, 6, True, 2e-10),
]:
    pair = {}
    for kind in ["constant", "resonant"]:
        run = pulse_run(na, nb, waveform_family(kind, matched_amplitudes[kind]), corrected=corrected, tolerance=atol)
        pair[kind] = run
        robust_rows.append({"check": label, "pulse": kind, "information": run["information"][-1],
                            "wrong codeword": run["wrong"], "leakage": run["leakage"],
                            "max memory tail": run["e"][5].real.max(), "max buffer tail": run["e"][6].real.max(),
                            "min reduced eigenvalue": checks(run)["min eigenvalue"]})
    print(f"{label}: information ratios to nominal = "
          f"{pair['constant']['information'][-1]/target_information:.3f}, "
          f"{pair['resonant']['information'][-1]/target_information:.3f}; "
          f"wrong-codeword ratio resonant/constant = {pair['resonant']['wrong']/pair['constant']['wrong']:.3f}")
robust_table = pd.DataFrame(robust_rows)
table(robust_table, "readout_robustness")
mistuning_rows = []
for factor in [0.9, 1.0, 1.1]:
    envelope = lambda t: jnp.where(t < 1.5, matched_amplitudes["resonant"] * jnp.sin(np.pi*t/1.5)**2 * jnp.cos(factor*omega*t), 0.0)
    run = pulse_run(28, 6, envelope)
    mistuning_rows.append({"envelope frequency/nominal": factor, "information": run["information"][-1],
                           "wrong codeword": run["wrong"], "information/wrong": run["information"][-1]/run["wrong"]})
table(pd.DataFrame(mistuning_rows), "envelope_mistuning")

# %% [markdown]
# what this does and does not establish
#
# the proposal is an underdamped, sideband-shaped readout with a matched temporal
# receiver. the local transfer function motivates its frequency; the two-mode
# nonlinear simulations decide whether it helps. the primary comparison is
# information per recovered logical-bit disturbance, not peak buffer amplitude.
# no new nonlinearity or extra stabilization pump is assumed. an iq-capable memory
# drive and time-resolved homodyne processing are required.
#
# the drive is still disturbing: a favorable comparison against constant readout
# does not mean the idle Γx was preserved. neither does it prove a high-fidelity
# single-shot hardware measurement. the reported gaussian assignment estimate
# is only a shot-noise proxy, and the example information budget is modest.
# a further experiment should simulate conditional measurement records, include
# realistic detection efficiency and output noise, and optimize a target error
# budget with the full calibrated circuit and bath model.
#
# the full-sine check is first order in pump amplitude and rotating-wave only.
# at the supplied illustrative circuit parameters, neglecting the higher odd
# powers without calibration would already shift the low-occupancy coupling by
# several percent. the off-resonant linear pump term should first be handled by
# a displacement/calibration; virtual shifts and transformed dissipation need a
# controlled effective-model calculation, not an unsupported blanket 'rwa is fine'.
#
# the introductory reference had a small negative density eigenvalue at the
# explicit integrator's tolerance. inspect the printed diagnostic: this numerical
# violation is tiny compared with the introductory observable tolerance, but
# would be unacceptable as an unreported floor for rare-error claims. readout
# has its own tighter-tolerance and cutoff checks above.
#
# this is a numerically tested proposal within the stated model, not a claim
# that no one has used resonant readout or matched filtering before.

# %%
summary = {"versions": versions, "time_unit": "microsecond", "rates": "expectation-value decay rates",
           "bit_suppression_exponent": float(-log_slope), "phase_slope": float(phase_slope),
           "predicted_phase_slope": float(2*ka), "matched_information": float(target_information),
           "wrong_codeword_reduction_fraction": float(relative_reduction),
           "constant_wrong_codeword": float(matched_runs["constant"]["wrong"]),
           "resonant_wrong_codeword": float(matched_runs["resonant"]["wrong"]),
           "caveat": "finite-pulse ideal-model benchmark; not a stationary rate or hardware measurement fidelity"}
(artifacts / "summary.json").write_text(json.dumps(summary, indent=2))  # actual results only
display(pd.Series({key: value for key, value in summary.items() if key != "versions"}))
print("the plots, tables, and this summary were generated by the cells above.")

# %% [markdown]
# sources and further checks
#
# - supplied `duquantum_2026_challenge_prompt-v4.md` and `2026-10-24 DuQuantun Hackathon.pdf`:
#   task definitions, circuit parameters, sign conventions, and workshop context.
# - [réglade, bocquet et al., 2024](https://arxiv.org/abs/2307.06617): the two-mode
#   experiment, its nonadiabatic regime, circuit derivation, and buffer readout.
# - [dubovitskii, 2025](https://arxiv.org/abs/2407.17299): second-order bit-flip
#   mechanisms; its event-rate convention differs by a factor of two here.
# - [guillaud, cohen, mirrahimi, lecture notes](https://arxiv.org/abs/2203.03222):
#   dissipative cats and the protected-manifold viewpoint.
# - [gautier et al., zeno gates](https://doi.org/10.1103/PRXQuantum.4.040316):
#   nonadiabatic control and information leakage; gate protection is not the
#   same objective as making leaked information useful for readout.
# - [carde et al., flux-pump degradation](https://arxiv.org/abs/2410.00975): why
#   circuit-level corrections can involve dissipators, not just extra hamiltonian terms.
# - [rousseau et al., moon cats](https://arxiv.org/abs/2502.07892): squeezing is
#   an existing stabilization strategy, not an original discovery of this notebook.
# - [dynamiqs solver documentation](https://www.dynamiqs.org/stable/python_api/integrators/mesolve.html)
#   and [official challenge tutorials](https://github.com/YQuantum-2026/Alice-Bob-Challenge/tree/main/challenge/resources).
#
# the prior psiteratops submission supplied as context studies a different drift
# optimization problem. its results are not evidence for this readout proposal.
