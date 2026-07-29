"""
xTB Engine — GFN2-xTB semi-empirical quantum mechanics.

Wraps the xtb command-line binary for:
- Single-point energy
- Geometry optimization
- Solvation energy (ALPB/GBSA)
- Strain energy (on docked poses)
"""

import asyncio
import logging
import os
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("novomcp-qm.xtb")

XTB_BIN = os.getenv("XTB_BIN", "xtb")
SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR", "/app/scratch"))


@dataclass
class XtbResult:
    success: bool
    energy_hartree: Optional[float] = None
    energy_kcal_mol: Optional[float] = None
    solvation_energy_kcal_mol: Optional[float] = None
    dipole_debye: Optional[float] = None
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    partial_charges: Optional[list[float]] = None
    optimized_xyz: Optional[str] = None
    wall_time_seconds: Optional[float] = None
    method: str = "GFN2-xTB"
    error: Optional[str] = None
    raw_output: str = ""


@dataclass
class HessianResult:
    success: bool
    energy_hartree: Optional[float] = None
    energy_kcal_mol: Optional[float] = None
    zpe_kcal_mol: Optional[float] = None
    enthalpy_correction_kcal_mol: Optional[float] = None
    gibbs_correction_kcal_mol: Optional[float] = None
    entropy_cal_mol_k: Optional[float] = None
    temperature_k: float = 298.15
    frequencies_cm1: Optional[list[float]] = None
    imaginary_frequencies_cm1: Optional[list[float]] = None
    n_imaginary: int = 0
    is_true_minimum: bool = True
    optimized_xyz: Optional[str] = None
    wall_time_seconds: Optional[float] = None
    method: str = "GFN2-xTB-hessian"
    error: Optional[str] = None


@dataclass
class ExcitedStateResult:
    success: bool
    excited_states: Optional[list[dict]] = None  # [{energy_ev, wavelength_nm, osc_strength, is_singlet}]
    s1_energy_ev: Optional[float] = None  # First singlet excited state
    t1_energy_ev: Optional[float] = None  # First triplet excited state
    singlet_triplet_gap_ev: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None  # From S1 (fluorescence)
    phosphorescence_wavelength_nm: Optional[float] = None  # From T1
    n_states: int = 0
    ground_state_energy_hartree: Optional[float] = None
    wall_time_seconds: Optional[float] = None
    method: str = "sTDA-xTB"
    error: Optional[str] = None


XTB4STDA_BIN = os.getenv("XTB4STDA_BIN", "xtb4stda")
STDA_BIN = os.getenv("STDA_BIN", "stda")


def is_available() -> bool:
    """Check if xtb binary is on PATH."""
    return shutil.which(XTB_BIN) is not None


def is_stda_available() -> bool:
    """Check if xtb4stda and stda binaries are on PATH."""
    return shutil.which(XTB4STDA_BIN) is not None and shutil.which(STDA_BIN) is not None


async def run_single_point(
    xyz_content: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
) -> XtbResult:
    """Run a single-point energy calculation."""
    return await _run_xtb(
        xyz_content=xyz_content,
        args=["--sp"],
        charge=charge,
        uhf=uhf,
        solvent=solvent,
    )


async def run_optimization(
    xyz_content: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    level: str = "normal",
) -> XtbResult:
    """Run geometry optimization."""
    return await _run_xtb(
        xyz_content=xyz_content,
        args=["--opt", level],
        charge=charge,
        uhf=uhf,
        solvent=solvent,
    )


async def run_strain_energy(
    docked_xyz: str,
    charge: int = 0,
) -> XtbResult:
    """
    Calculate strain energy: E(docked_pose) - E(optimized).
    Useful for correcting docking scores.
    """
    # Single-point on the docked pose
    sp_result = await run_single_point(docked_xyz, charge=charge)
    if not sp_result.success or sp_result.energy_hartree is None:
        return XtbResult(success=False, error=f"Single-point failed: {sp_result.error}")

    # Optimize from docked pose
    opt_result = await run_optimization(docked_xyz, charge=charge)
    if not opt_result.success or opt_result.energy_hartree is None:
        return XtbResult(success=False, error=f"Optimization failed: {opt_result.error}")

    strain_hartree = sp_result.energy_hartree - opt_result.energy_hartree
    strain_kcal = strain_hartree * 627.509  # Hartree to kcal/mol

    return XtbResult(
        success=True,
        energy_hartree=strain_hartree,
        energy_kcal_mol=round(strain_kcal, 2),
        optimized_xyz=opt_result.optimized_xyz,
        wall_time_seconds=round(
            (sp_result.wall_time_seconds or 0) + (opt_result.wall_time_seconds or 0), 1
        ),
        method="GFN2-xTB-strain",
    )


async def run_hessian(
    xyz_content: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    temperature: float = 298.15,
    optimize_first: bool = False,
) -> HessianResult:
    """Run Hessian calculation for vibrational frequencies and thermochemistry.

    Args:
        xyz_content: XYZ geometry (ideally pre-optimized for meaningful thermo).
        charge: Molecular charge.
        uhf: Number of unpaired electrons.
        solvent: ALPB solvent model.
        temperature: Temperature in K for thermochemistry (default 298.15).
        optimize_first: If True, use --ohess (optimize then Hessian) instead of --hess.
    """
    import time
    import json as _json

    hess_flag = "--ohess" if optimize_first else "--hess"
    workdir = tempfile.mkdtemp(dir=SCRATCH_DIR, prefix="xtb_hess_")
    try:
        input_file = Path(workdir) / "input.xyz"
        input_file.write_text(xyz_content)

        cmd = [
            XTB_BIN, str(input_file), hess_flag,
            "--chrg", str(charge),
            "--json",
        ]
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if solvent:
            cmd.extend(["--alpb", solvent])
        if abs(temperature - 298.15) > 0.01:
            # xTB uses XTBPATH/.param or --etemp for electronic temp;
            # thermochemistry temperature is set in the xcontrol file
            xcontrol = Path(workdir) / "xcontrol"
            xcontrol.write_text(f"$thermo\n  temp={temperature}\n$end\n")
            cmd.extend(["--input", str(xcontrol)])

        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        wall_time = time.time() - t0

        output = stdout.decode("utf-8", errors="replace")
        err_output = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return HessianResult(
                success=False,
                error=f"xtb --hess exit {proc.returncode}: {err_output[:500]}",
                wall_time_seconds=round(wall_time, 1),
            )

        # Parse energy from stdout
        energy_hartree = None
        for line in output.splitlines():
            if "TOTAL ENERGY" in line and "Eh" in line:
                try:
                    energy_hartree = float(line.split()[-3])
                except (ValueError, IndexError):
                    pass

        # Parse thermochemistry from combined stdout + stderr.
        combined_text = output + "\n" + err_output
        thermo = _parse_thermochemistry(combined_text)

        # Fallback 1: parse xtbout.json for thermochemistry
        json_out = Path(workdir) / "xtbout.json"
        if json_out.exists():
            try:
                jdata = _json.loads(json_out.read_text())
                # xTB JSON may use various key names depending on version
                for zpe_key in ["zero point energy", "ZPVE", "zpve"]:
                    if zpe_key in jdata and not thermo.get("zpe_kcal"):
                        thermo["zpe_kcal"] = round(float(jdata[zpe_key]) * 627.509, 3)
                # Total free energy (includes ZPE + thermal + entropy)
                for gfe_key in ["gibbs free energy", "free energy", "total free energy", "G(RRHO)"]:
                    if gfe_key in jdata and not thermo.get("gibbs_kcal") and energy_hartree:
                        gfe = float(jdata[gfe_key])
                        thermo["gibbs_kcal"] = round((gfe - energy_hartree) * 627.509, 3)
            except Exception:
                pass

        # Fallback 2: parse the g98.out file (Gaussian-format output from xTB --hess)
        # This file contains thermochemistry in a well-defined, parseable format:
        #   Zero-point correction=           0.XXXXXX (Hartree/Particle)
        #   Thermal correction to Enthalpy=  0.XXXXXX
        #   Thermal correction to Gibbs Free Energy= 0.XXXXXX
        g98_file = Path(workdir) / "g98.out"
        if g98_file.exists() and not thermo.get("zpe_kcal"):
            try:
                g98_text = g98_file.read_text()
                for line in g98_text.splitlines():
                    stripped = line.strip()
                    if "Zero-point correction=" in stripped:
                        val = float(stripped.split("=")[1].split("(")[0].strip())
                        thermo["zpe_kcal"] = round(val * 627.509, 3)
                    elif "Thermal correction to Enthalpy=" in stripped:
                        val = float(stripped.split("=")[1].strip())
                        thermo["enthalpy_kcal"] = round(val * 627.509, 3)
                    elif "Thermal correction to Gibbs Free Energy=" in stripped:
                        val = float(stripped.split("=")[1].strip())
                        if energy_hartree:
                            thermo["gibbs_kcal"] = round((energy_hartree + val) * 627.509 - energy_hartree * 627.509, 3)
                            # Simpler: gibbs_correction = val * 627.509
                            thermo["gibbs_kcal"] = round(val * 627.509, 3)
                    elif "E (Thermal)" in stripped and "KCal/Mol" in stripped:
                        # This is the header line; next non-empty line has the values
                        pass
            except Exception:
                pass

        # Fallback 3: extract entropy from ZPE + Gibbs if we have both
        if thermo.get("zpe_kcal") and thermo.get("gibbs_kcal") and not thermo.get("entropy_cal"):
            # G = H - TS, roughly: gibbs_correction = enthalpy_correction - T*S
            # We can derive entropy if we have enthalpy and gibbs corrections
            h = thermo.get("enthalpy_kcal")
            g = thermo.get("gibbs_kcal")
            if h is not None and g is not None:
                ts_kcal = h - g  # T*S in kcal/mol
                thermo["entropy_cal"] = round((ts_kcal * 1000.0) / temperature, 3)

        # Diagnostic: log what files exist and key content for debugging thermo parsing
        if not thermo.get("zpe_kcal"):
            workdir_path = Path(workdir)
            diag_files = [f.name for f in workdir_path.iterdir() if f.is_file()]
            logger.warning(f"Hessian thermo parsing returned no ZPE. Files in workdir: {diag_files}")
            # Check g98.out content
            g98_check = workdir_path / "g98.out"
            if g98_check.exists():
                g98_head = g98_check.read_text()[:500]
                logger.warning(f"g98.out head: {g98_head}")
            else:
                logger.warning("g98.out does NOT exist in workdir")
            # Check xtbout.json keys
            json_check = workdir_path / "xtbout.json"
            if json_check.exists():
                try:
                    jkeys = list(_json.loads(json_check.read_text()).keys())
                    logger.warning(f"xtbout.json keys: {jkeys}")
                except Exception:
                    pass
            # Check stdout for thermo-related lines
            thermo_lines = [l.strip() for l in combined_text.splitlines()
                           if any(k in l.lower() for k in ["zero point", "zpve", "zpe", "enthalpy", "gibbs", "g(rrho)", "t*s", "thermo"])]
            logger.warning(f"Thermo-related stdout/stderr lines: {thermo_lines[:15]}")

        # Parse vibspectrum file for frequencies
        frequencies, imaginary = _parse_vibspectrum(Path(workdir) / "vibspectrum")

        # Check for optimized geometry (only present with --ohess)
        opt_xyz = None
        opt_file = Path(workdir) / "xtbopt.xyz"
        if opt_file.exists():
            opt_xyz = opt_file.read_text()

        n_imag = len(imaginary) if imaginary else 0

        return HessianResult(
            success=True,
            energy_hartree=energy_hartree,
            energy_kcal_mol=round(energy_hartree * 627.509, 2) if energy_hartree else None,
            zpe_kcal_mol=thermo.get("zpe_kcal"),
            enthalpy_correction_kcal_mol=thermo.get("enthalpy_kcal"),
            gibbs_correction_kcal_mol=thermo.get("gibbs_kcal"),
            entropy_cal_mol_k=thermo.get("entropy_cal"),
            temperature_k=temperature,
            frequencies_cm1=frequencies,
            imaginary_frequencies_cm1=imaginary if imaginary else [],
            n_imaginary=n_imag,
            is_true_minimum=n_imag == 0,
            optimized_xyz=opt_xyz,
            wall_time_seconds=round(wall_time, 1),
            method=f"GFN2-xTB-{'ohess' if optimize_first else 'hess'}",
        )

    except asyncio.TimeoutError:
        return HessianResult(success=False, error="xtb --hess timed out after 600s")
    except Exception as e:
        return HessianResult(success=False, error=str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def run_excited_states(
    xyz_content: str,
    charge: int = 0,
    num_states: int = 10,
) -> ExcitedStateResult:
    """Run sTDA-xTB excited state calculation.

    Two-step workflow:
      1. xtb4stda generates ground-state wavefunction (wfn.xtb)
      2. stda computes excited states from the wavefunction

    Returns singlet and triplet excitation energies, oscillator strengths,
    and emission wavelengths.
    """
    import time

    if not is_stda_available():
        return ExcitedStateResult(
            success=False,
            error="xtb4stda or stda not installed"
        )

    workdir = tempfile.mkdtemp(dir=SCRATCH_DIR, prefix="stda_")
    try:
        input_file = Path(workdir) / "input.xyz"
        input_file.write_text(xyz_content)

        t0 = time.time()

        # Step 1: Generate wavefunction with xtb4stda
        cmd1 = [XTB4STDA_BIN, str(input_file)]
        if charge != 0:
            cmd1.extend(["-chrg", str(charge)])

        proc1 = await asyncio.create_subprocess_exec(
            *cmd1,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout1, stderr1 = await asyncio.wait_for(proc1.communicate(), timeout=120)

        if proc1.returncode != 0:
            err = stderr1.decode("utf-8", errors="replace")[:300]
            out = stdout1.decode("utf-8", errors="replace")[:300]
            return ExcitedStateResult(
                success=False,
                error=f"xtb4stda failed (exit {proc1.returncode}): {err or out}",
                wall_time_seconds=round(time.time() - t0, 1),
            )

        # Capture xtb4stda output for diagnostics
        out1_text = stdout1.decode("utf-8", errors="replace")
        err1_text = stderr1.decode("utf-8", errors="replace")
        logger.info(f"xtb4stda exit={proc1.returncode}, stdout={len(out1_text)} chars, stderr={len(err1_text)} chars")

        # Check wavefunction file was created
        files_in_workdir = [f.name for f in Path(workdir).iterdir() if f.is_file()]
        wfn_file = None

        # xtb4stda may produce wfn.xtb, tda_kernel, or other files
        for candidate in ["wfn.xtb", "xtb4stda.wfn", "tda_kernel"]:
            p = Path(workdir) / candidate
            if p.exists():
                wfn_file = p
                break

        if wfn_file is None:
            # Log the actual output for debugging
            logger.warning(f"xtb4stda produced no wavefunction. Files: {files_in_workdir}")
            logger.warning(f"xtb4stda stdout tail: {out1_text[-500:]}")
            logger.warning(f"xtb4stda stderr tail: {err1_text[-300:]}")
            return ExcitedStateResult(
                success=False,
                error=(
                    f"xtb4stda produced no wavefunction file. "
                    f"Files: {files_in_workdir[:10]}. "
                    f"stdout tail: {out1_text[-200:]}. "
                    f"stderr tail: {err1_text[-200:]}"
                ),
                wall_time_seconds=round(time.time() - t0, 1),
            )

        # Parse ground state energy from xtb4stda output
        gs_energy = None
        for line in out1_text.splitlines():
            if "TOTAL ENERGY" in line and "Eh" in line:
                try:
                    gs_energy = float(line.split()[-3])
                except (ValueError, IndexError):
                    pass

        # Step 2: Run stda for singlet excited states.
        #
        # IMPORTANT: stda's `-e` flag is an ENERGY CUTOFF IN eV, not a state
        # count. Passing `num_states` (a count like 5 or 10) silently caps
        # the energy window — so for deep-UV chromophores whose S1 sits
        # above that threshold, stda finds zero states and we get
        # "no parseable excited states" (e.g. ethanol at num_states=5 → S1
        # is ~8 eV, above the 5 eV cap, so nothing).
        #
        # Always run with a generous eV cap (covers all normal chromophores
        # including aliphatic alcohols / water σ→σ* around 8-10 eV) and
        # truncate to `num_states` per spin manifold after parsing.
        STDA_ENERGY_CAP_EV = 15.0
        cmd2 = [STDA_BIN, "-xtb", "-e", str(STDA_ENERGY_CAP_EV)]

        proc2 = await asyncio.create_subprocess_exec(
            *cmd2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout2, stderr2 = await asyncio.wait_for(proc2.communicate(), timeout=120)

        out2_text = stdout2.decode("utf-8", errors="replace")
        err2_text = stderr2.decode("utf-8", errors="replace")

        if proc2.returncode != 0:
            return ExcitedStateResult(
                success=False,
                error=f"stda singlet failed (exit {proc2.returncode}): {(err2_text or out2_text)[:300]}",
                wall_time_seconds=round(time.time() - t0, 1),
            )

        # Parse excited states from stda output
        # Log the raw stda output to diagnose column format
        combined = out2_text + "\n" + err2_text
        stda_lines = [l for l in combined.splitlines()
                      if any(k in l.lower() for k in ["state", "excitation", "triplet", "singlet", "ev", "nm"])]
        logger.info(f"stda output lines ({len(stda_lines)}): {stda_lines[:20]}")

        states = _parse_stda_output(out2_text)

        if not states:
            states = _parse_stda_output(combined)

        if not states:
            return ExcitedStateResult(
                success=False,
                error=f"stda produced no parseable excited states. Output tail: {out2_text[-300:]}",
                wall_time_seconds=round(time.time() - t0, 1),
            )

        # Step 3: Run stda for triplet excited states.
        # stda -t computes triplet excitations from the same wavefunction.
        # Same energy-cap vs state-count gotcha as singlets above — use the
        # same generous eV cap, truncate after parse.
        triplet_states = []
        try:
            cmd3 = [STDA_BIN, "-xtb", "-t", "-e", str(STDA_ENERGY_CAP_EV)]
            proc3 = await asyncio.create_subprocess_exec(
                *cmd3,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
            stdout3, stderr3 = await asyncio.wait_for(proc3.communicate(), timeout=120)
            if proc3.returncode == 0:
                out3_text = stdout3.decode("utf-8", errors="replace")
                triplet_states = _parse_stda_output(out3_text)
                # Mark all as triplet (the parser defaults to singlet for the excitation block)
                for ts in triplet_states:
                    ts["is_singlet"] = False
                    ts["label"] = f"T{ts.get('state', '?')}"
                states.extend(triplet_states)
        except Exception as e:
            logger.debug(f"Triplet stda failed (non-fatal): {e}")

        wall_time = time.time() - t0

        # Truncate to the requested number of states per manifold. The stda
        # binary ran with a generous energy cap (15 eV), so the parsed list
        # can contain far more than num_states entries — slice before returning
        # so the caller gets the N lowest singlets and N lowest triplets, as
        # the num_states parameter implies.
        singlets = [s for s in states if s.get("is_singlet", True)][:num_states]
        triplets = [s for s in states if not s.get("is_singlet", True)][:num_states]
        states = singlets + triplets

        s1 = singlets[0] if singlets else None
        t1 = triplets[0] if triplets else None

        s1_ev = s1["energy_ev"] if s1 else None
        t1_ev = t1["energy_ev"] if t1 else None
        st_gap = round(s1_ev - t1_ev, 4) if (s1_ev and t1_ev) else None

        emission_nm = s1.get("wavelength_nm") if s1 else None
        phos_nm = t1.get("wavelength_nm") if t1 else None

        return ExcitedStateResult(
            success=True,
            excited_states=states,
            s1_energy_ev=s1_ev,
            t1_energy_ev=t1_ev,
            singlet_triplet_gap_ev=st_gap,
            emission_wavelength_nm=emission_nm,
            phosphorescence_wavelength_nm=phos_nm,
            n_states=len(states),
            ground_state_energy_hartree=gs_energy,
            wall_time_seconds=round(wall_time, 1),
        )

    except asyncio.TimeoutError:
        return ExcitedStateResult(success=False, error="stTDA timed out after 120s")
    except Exception as e:
        return ExcitedStateResult(success=False, error=str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_stda_output(output: str) -> list[dict]:
    """Parse stda v1.6.1 stdout for excited state data.

    stda output has multiple sections with numeric data. We ONLY want the
    final "excitation energies, transition moments and TDA amplitudes" block.
    The earlier "lowest CSF states" block also has numbered lines but the
    columns are different (no oscillator strength, column 3 is # centers).

    Singlet section:
      excitation energies, transition moments and TDA amplitudes
      state    eV      nm       fL        Rv(corr)
          1   3.249   381.6    0.23972    -2.1234
          2   3.580   346.6    0.00001     0.0000

    Triplet section (only when stda --triplet was used):
      triplet excitation energies
      state    eV      nm
          1   1.850   670.3
          2   3.100   400.0

    We require in_singlet_block or in_triplet_block to be set before
    parsing any numeric lines — this skips all preamble data.
    """
    states = []
    in_singlet_block = False
    in_triplet_block = False
    past_header_line = False  # Skip the "state eV nm fL..." column header

    for line in output.splitlines():
        stripped = line.strip()

        # Detect the singlet excitation block header
        if "excitation energies, transition moments" in stripped.lower():
            in_singlet_block = True
            in_triplet_block = False
            past_header_line = False
            states = []  # Reset — discard any garbage from earlier blocks
            continue

        # Detect triplet block header
        if "triplet" in stripped.lower() and "excitation" in stripped.lower():
            in_triplet_block = True
            in_singlet_block = False
            past_header_line = False
            continue

        # Skip column header line ("state    eV      nm       fL ...")
        if (in_singlet_block or in_triplet_block) and not past_header_line:
            if "state" in stripped.lower() and "eV" in stripped.lower():
                past_header_line = True
                continue

        # Only parse data lines when inside a recognized block
        if not (in_singlet_block or in_triplet_block):
            continue

        # Separator or empty line ends the block
        if stripped.startswith("---") or stripped.startswith("===") or not stripped:
            if in_singlet_block or in_triplet_block:
                # Only end block on a real separator after we've seen data
                if states and (stripped.startswith("---") or stripped.startswith("===")):
                    in_singlet_block = False
                    in_triplet_block = False
            continue

        # Parse state lines: "  1    3.249   381.6     0.23972   -2.1234"
        parts = stripped.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                state_num = int(parts[0])
                energy_ev = float(parts[1])
                wavelength_nm = float(parts[2])

                # Sanity check
                if energy_ev <= 0 or wavelength_nm < 50:
                    continue

                state = {
                    "state": state_num,
                    "label": f"S{state_num}" if in_singlet_block else f"T{state_num}",
                    "energy_ev": round(energy_ev, 4),
                    "wavelength_nm": round(wavelength_nm, 1),
                    "is_singlet": in_singlet_block,
                }

                # Oscillator strength (column 3 — only valid in singlet block)
                if in_singlet_block and len(parts) >= 4:
                    try:
                        osc = float(parts[3])
                        # Sanity: oscillator strength should be 0-5 for physical transitions
                        if 0 <= osc <= 5:
                            state["oscillator_strength"] = round(osc, 6)
                    except ValueError:
                        pass

                states.append(state)
            except (ValueError, IndexError):
                continue

    return states


def _parse_thermochemistry(output: str) -> dict:
    """Parse thermochemistry from xTB --hess output.

    xTB 6.7.1 prints values in Hartree (Eh), NOT kcal/mol.
    The actual format is:

    The `::` summary block:
        :: zero point energy           0.078129992099 Eh   ::
        :: G(RRHO) w/o ZPVE           -0.025370657764 Eh   ::
        :: G(RRHO) contrib.            0.052759334335 Eh   ::

    The `|` table block:
        | TOTAL ENTHALPY            -11.310998464631 Eh   |

    The data table (values in Eh):
        T/K    H(0)-H(T)+PV         H(T)/Eh          T*S/Eh         G(T)/Eh
        298.15  0.533849E-02  -0.113110E+02   0.253707E-01  -0.113363E+02

    We parse the :: block first (most reliable), then fall back to
    the data table for H and T*S.
    """
    HARTREE_TO_KCAL = 627.509
    result = {}
    lines = output.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()

        # ZPE: ":: zero point energy           0.078129992099 Eh   ::"
        if "zero point energy" in stripped.lower() and "Eh" in stripped:
            try:
                # Extract the number between the text and "Eh"
                parts = stripped.split()
                for j, p in enumerate(parts):
                    if p == "Eh":
                        val = float(parts[j - 1])
                        result["zpe_kcal"] = round(val * HARTREE_TO_KCAL, 3)
                        break
            except (ValueError, IndexError):
                pass

        # G(RRHO) contrib: ":: G(RRHO) contrib.            0.052759334335 Eh   ::"
        if "g(rrho) contrib" in stripped.lower() and "Eh" in stripped and "gibbs_kcal" not in result:
            try:
                parts = stripped.split()
                for j, p in enumerate(parts):
                    if p == "Eh":
                        val = float(parts[j - 1])
                        result["gibbs_kcal"] = round(val * HARTREE_TO_KCAL, 3)
                        break
            except (ValueError, IndexError):
                pass

        # TOTAL ENTHALPY: "| TOTAL ENTHALPY            -11.310998464631 Eh   |"
        if "total enthalpy" in stripped.lower() and "Eh" in stripped:
            try:
                parts = stripped.split()
                for j, p in enumerate(parts):
                    if p == "Eh":
                        result["total_enthalpy_eh"] = float(parts[j - 1])
                        break
            except (ValueError, IndexError):
                pass

        # Data table row: "298.15  0.533849E-02  -0.113110E+02   0.253707E-01  -0.113363E+02"
        # Headers: T/K    H(0)-H(T)+PV    H(T)/Eh    T*S/Eh    G(T)/Eh
        if stripped.startswith("298.15") or (stripped and stripped[0].isdigit()):
            parts = stripped.split()
            if len(parts) == 5:
                try:
                    temp = float(parts[0])
                    if abs(temp - 298.15) < 1.0:
                        h0_ht_pv = float(parts[1])  # H(0)-H(T)+PV correction
                        ts_eh = float(parts[3])      # T*S in Eh
                        gt_eh = float(parts[4])       # G(T) in Eh

                        if "enthalpy_kcal" not in result:
                            result["enthalpy_kcal"] = round(h0_ht_pv * HARTREE_TO_KCAL, 3)
                        if "entropy_cal" not in result:
                            # T*S in Eh → S in cal/mol/K = (T*S * 627509 cal/mol) / T
                            result["entropy_cal"] = round((ts_eh * HARTREE_TO_KCAL * 1000.0) / temp, 3)
                        if "gibbs_kcal" not in result:
                            # G(T) relative correction = G(T) - E_total
                            # But we have the G(RRHO) contrib from :: block which is cleaner
                            pass
                except (ValueError, IndexError):
                    pass

    return result


# Drop modes whose absolute frequency is below this threshold. xTB's vibspectrum
# file writes all 3N modes, including 6 (or 5 for linear molecules) translation
# and rotation modes that should ideally be exactly zero but numerically land
# in the ±0.01–1 cm⁻¹ range. Real vibrational modes for small organics start
# well above 50 cm⁻¹ (lowest torsions), so a 5 cm⁻¹ cutoff cleanly separates
# the two populations without risking the loss of a legitimate low-frequency
# vibration.
TRANS_ROT_CUTOFF_CM1 = 5.0


def _parse_vibspectrum(vibspec_path: Path) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """Parse the vibspectrum file for vibrational frequencies.

    Format:
        $vibrational spectrum
        #  mode    symmetry   wave number   IR intensity   ...
        #                      cm**(-1)       km/mol
             1                    -12.34        0.000
             2                     45.67        1.234
        ...
        $end

    Returns (vibrational_frequencies, imaginary_frequencies). Translation and
    rotation modes (|freq| < TRANS_ROT_CUTOFF_CM1) are excluded so the list
    length matches the expected 3N-6 (or 3N-5) vibrational DOF.
    Imaginary frequencies are reported as negative values.
    """
    if not vibspec_path.exists():
        return None, None

    frequencies = []
    imaginary = []
    try:
        for line in vibspec_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("$") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    freq = float(parts[2]) if len(parts) > 2 else float(parts[1])
                    # Skip numerical zeros corresponding to the 6 trans/rot
                    # modes. A real vibration's magnitude is always > cutoff;
                    # a true imaginary frequency (negative) passes the check
                    # by absolute value and stays in the list.
                    if abs(freq) < TRANS_ROT_CUTOFF_CM1:
                        continue
                    frequencies.append(round(freq, 2))
                    if freq < -TRANS_ROT_CUTOFF_CM1:
                        imaginary.append(round(freq, 2))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.warning(f"Failed to parse vibspectrum: {e}")
        return None, None

    return frequencies if frequencies else None, imaginary if imaginary else None


async def _run_xtb(
    xyz_content: str,
    args: list[str],
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    timeout: int = 300,
) -> XtbResult:
    """Run xtb with given arguments."""
    import time

    workdir = tempfile.mkdtemp(dir=SCRATCH_DIR, prefix="xtb_")
    try:
        input_file = Path(workdir) / "input.xyz"
        input_file.write_text(xyz_content)

        cmd = [XTB_BIN, str(input_file)] + args
        cmd.extend(["--chrg", str(charge)])
        cmd.append("--json")  # Write xtbout.json with per-atom charges
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if solvent:
            cmd.extend(["--alpb", solvent])

        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        wall_time = time.time() - t0

        output = stdout.decode("utf-8", errors="replace")
        err_output = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return XtbResult(
                success=False,
                error=f"xtb exit code {proc.returncode}: {err_output[:500]}",
                raw_output=output[:2000],
                wall_time_seconds=round(wall_time, 1),
            )

        result = _parse_xtb_output(output)
        result.wall_time_seconds = round(wall_time, 1)
        result.raw_output = output[:2000]

        # Check for optimized geometry
        opt_xyz = Path(workdir) / "xtbopt.xyz"
        if opt_xyz.exists():
            result.optimized_xyz = opt_xyz.read_text()

        # Extract per-atom Mulliken charges from xtbout.json
        json_file = Path(workdir) / "xtbout.json"
        if json_file.exists():
            try:
                import json
                with open(json_file) as f:
                    xtb_json = json.load(f)
                charges = xtb_json.get("partial charges")
                if isinstance(charges, list):
                    result.partial_charges = [round(c, 6) for c in charges]
            except Exception as e:
                logger.warning(f"Failed to parse xtbout.json: {e}")

        # Fallback: parse charges from stdout if JSON failed
        if result.partial_charges is None:
            result.partial_charges = _parse_mulliken_charges(output)

        return result

    except asyncio.TimeoutError:
        return XtbResult(success=False, error=f"xtb timed out after {timeout}s")
    except Exception as e:
        return XtbResult(success=False, error=str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_xtb_output(output: str) -> XtbResult:
    """Parse xtb stdout for key results."""
    energy = None
    homo = None
    lumo = None
    dipole = None

    for line in output.splitlines():
        line = line.strip()
        if "TOTAL ENERGY" in line and "Eh" in line:
            try:
                energy = float(line.split()[-3])
            except (ValueError, IndexError):
                pass
        elif "HOMO-LUMO GAP" in line:
            try:
                parts = line.split()
                gap_idx = parts.index("GAP") + 1
                gap = float(parts[gap_idx])
            except (ValueError, IndexError):
                gap = None
        elif "(HOMO)" in line:
            # xTB 6.7.1 format (4+ columns):
            #   31   2.0000   -0.248371   -6.759    (HOMO)
            # The eV energy is always the value immediately before (HOMO).
            # Column count varies — anchor on the marker, not a fixed index.
            try:
                parts = line.split()
                marker_idx = parts.index("(HOMO)")
                homo = float(parts[marker_idx - 1])
            except (ValueError, IndexError):
                pass
        elif "(LUMO)" in line:
            try:
                parts = line.split()
                marker_idx = parts.index("(LUMO)")
                lumo = float(parts[marker_idx - 1])
            except (ValueError, IndexError):
                pass
        elif "molecular dipole:" in line.lower():
            # Next line after "molecular dipole:" has the values
            pass
        elif line.startswith("full:") and dipole is None:
            try:
                parts = line.split()
                if len(parts) >= 5:
                    dipole = float(parts[4])
            except (ValueError, IndexError):
                pass

    # Prefer the explicitly printed HOMO-LUMO GAP (always positive, in eV).
    # Fall back to |LUMO - HOMO| if the GAP line wasn't found.
    gap_ev = None
    if gap is not None:
        gap_ev = round(abs(gap), 3)
    elif homo is not None and lumo is not None:
        gap_ev = round(abs(lumo - homo), 3)

    return XtbResult(
        success=energy is not None,
        energy_hartree=energy,
        energy_kcal_mol=round(energy * 627.509, 2) if energy else None,
        homo_ev=homo,
        lumo_ev=lumo,
        gap_ev=gap_ev,
        dipole_debye=dipole,
    )


def _parse_mulliken_charges(output: str) -> Optional[list[float]]:
    """Parse Mulliken charges from xTB stdout as fallback."""
    charges = []
    in_charges = False
    for line in output.splitlines():
        stripped = line.strip()
        if "Mulliken/CM5 charges" in stripped:
            in_charges = True
            continue
        if in_charges:
            if stripped == "" or "---" in stripped or "total" in stripped.lower():
                break
            parts = stripped.split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    charges.append(float(parts[2]))  # Mulliken charge column
                except (ValueError, IndexError):
                    pass
    return charges if charges else None
