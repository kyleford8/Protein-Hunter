"""
Cross-validation with the *other* structure stack vs. design:
  - Boltz-designed YAMLs → Chai re-prediction (run_chai_validation_step)
  - Chai-designed YAMLs → Boltz re-prediction (run_boltz_validation_step)

Artifacts mirror the AlphaFold3 validation layout so Rosetta scoring
(utils/pyrosetta_utils.measure_rosetta_energy) is unchanged.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def _write_af3_style_confidence_artifacts(
    model_dir: Path,
    base_name: str,
    mmcif_text: str,
    iptm: float,
    atom_plddts: np.ndarray,
    pae: Optional[np.ndarray],
) -> None:
    """Write AF3-compatible filenames next to the model CIF for convert.py / Rosetta."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{base_name}_model.cif").write_text(mmcif_text)
    summary_path = model_dir / f"{base_name}_summary_confidences.json"
    summary_path.write_text(json.dumps({"iptm": float(iptm)}, indent=2))
    pae_arr = np.asarray(pae) if pae is not None else np.zeros((1, 1), dtype=np.float32)
    conf = {
        "atom_plddts": [float(x) for x in np.asarray(atom_plddts).flatten()],
        "pae": pae_arr.tolist(),
    }
    (model_dir / f"{base_name}_confidences.json").write_text(json.dumps(conf, indent=2))


def _boltz_build_apo_dict(holo: Dict[str, Any], binder_chain: str) -> Optional[Dict[str, Any]]:
    """Single-chain binder YAML dict for apo re-prediction (Boltz schema)."""
    binder_block = None
    for ent in holo.get("sequences", []):
        if "protein" not in ent:
            continue
        if binder_chain in ent["protein"].get("id", []):
            binder_block = copy.deepcopy(ent)
            break
    if binder_block is None:
        return None
    apo: Dict[str, Any] = {"version": holo.get("version", 1), "sequences": [binder_block]}
    if "properties" in holo:
        apo["properties"] = copy.deepcopy(holo["properties"])
    if "templates" in holo:
        tpl = []
        for t in holo.get("templates", []) or []:
            if t.get("chain_id") == binder_chain:
                tpl.append(copy.deepcopy(t))
        if tpl:
            apo["templates"] = tpl
    return apo


def _chai_chain_tuples_from_yaml(
    yaml_data: Dict[str, Any], binder_chain: str, align_target_weight: float = 1.0
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Build Chai prep_inputs tuples: (sequence, chain_id, entity_type, opts).
    Target chains first, binder last (matches chai_ph.pipeline.fold_sequence).
    """
    sequences = yaml_data.get("sequences") or []
    targets: List[Tuple[str, str, str, Dict[str, Any]]] = []
    binder: Optional[Tuple[str, str, str, Dict[str, Any]]] = None

    for block in sequences:
        if "protein" in block:
            p = block["protein"]
            chain_id = p["id"][0]
            seq = p["sequence"]
            cyclic = bool(p.get("cyclic", False))
            opts = {"use_esm": False, "cyclic": cyclic}
            if chain_id == binder_chain:
                binder = (seq, chain_id, "protein", opts)
            else:
                targets.append(
                    (seq, chain_id, "protein", {"use_esm": True, "align": align_target_weight})
                )
        elif "ligand" in block:
            lig = block["ligand"]
            chain_id = lig["id"][0]
            if "smiles" in lig and lig["smiles"]:
                smi = lig["smiles"]
                targets.append(
                    (smi, chain_id, "ligand", {"use_esm": False, "align": align_target_weight})
                )
            elif lig.get("ccd"):
                raise ValueError(
                    "Chai validation does not support CCD-only ligands in YAML; use SMILES."
                )

    if binder is None:
        raise ValueError(f"No protein chain with id {binder_chain!r} found in YAML.")
    return targets + [binder]


def _chai_apo_tuples(binder_chain: str, binder_seq: str, cyclic: bool) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    return [(binder_seq, binder_chain, "protein", {"use_esm": False, "cyclic": cyclic})]


def _chai_plddt_per_real_atom(folder: Any, res: Dict[str, Any]):
    """
    Chai stores plddt_per_atom over the full padded atom batch; padding often repeats
    degenerate scores. Restrict to real atoms using atom_exists_mask (matches exported structure).
    """
    import torch

    raw = res.get("plddt_per_atom")
    if raw is None:
        return np.array([], dtype=np.float32)
    p = raw.detach().float().reshape(-1)
    st = getattr(folder, "state", None)
    bi = getattr(st, "batch_inputs", None) if st is not None else None
    if not isinstance(bi, dict) or "atom_exists_mask" not in bi:
        return p.cpu().numpy()
    mask = bi["atom_exists_mask"]
    if hasattr(mask, "squeeze"):
        mask = mask.squeeze(0)
    m = mask.to(device=p.device, dtype=torch.bool).reshape(-1)
    n = min(m.numel(), p.numel())
    if m.numel() != p.numel():
        print(
            f"Warning: atom_exists_mask length ({m.numel()}) != plddt_per_atom ({p.numel()}); "
            f"trimming to {n} for masked export."
        )
    return p[:n][m[:n]].cpu().numpy()


def run_boltz_validation_step(
    yaml_dir: str,
    ligandmpnn_dir: str,
    work_dir: str,
    binder_id: str = "A",
    gpu_id: int = 0,
    boltz_model_path: str = "",
    boltz_model_version: str = "boltz2",
    diffuse_steps: int = 200,
    recycling_steps: int = 3,
    contact_residues: str = "",
    ccd_path: str = "~/.boltz/mols",
    grad_enabled: bool = False,
    high_iptm: bool = True,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Re-predict each high-ipTM YAML with Boltz (holo + apo), same layout as AF3 validation.
    Returns (holo_cif_root, apo_cif_root, holo_pdb_dir, apo_pdb_dir) like run_alphafold_step.
    """
    work_dir = os.path.abspath(os.path.expanduser(work_dir or os.getcwd()))
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
    boltz_ph = os.path.join(work_dir, "boltz_ph")
    if boltz_ph not in sys.path:
        sys.path.insert(0, boltz_ph)

    import torch
    from boltz.data.mol import load_canonicals
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.write.mmcif import to_mmcif

    from model_utils import get_batch, get_boltz_model, run_prediction

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    ccd_path_exp = os.path.expanduser(ccd_path)
    ccd_lib = load_canonicals()
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": diffuse_steps,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
        "max_parallel_samples": 1,
    }
    no_potentials = not bool(str(contact_residues).strip())
    model = get_boltz_model(
        checkpoint=os.path.expanduser(boltz_model_path),
        predict_args=predict_args,
        device=device,
        model_version=boltz_model_version,
        no_potentials=no_potentials,
        grad_enabled=grad_enabled,
    )
    model = model.to(device)
    model.eval()

    af_output_dir = os.path.join(ligandmpnn_dir, "02_design_final_af3")
    af_output_apo_dir = os.path.join(ligandmpnn_dir, "02_design_final_af3_apo")
    os.makedirs(af_output_dir, exist_ok=True)
    os.makedirs(af_output_apo_dir, exist_ok=True)

    yaml_paths = sorted(Path(yaml_dir).glob("*.yaml"))
    if not yaml_paths:
        print("No YAML files for Boltz validation. Skipping.")
        return af_output_dir, af_output_apo_dir, None, None

    boltz2 = boltz_model_version == "boltz2"

    def _predict_and_write(data: Dict[str, Any], out_root: str, stem: str) -> None:
        pocket_conditioning = bool(data.get("constraints"))
        name = stem
        data_use = copy.deepcopy(data)
        target = parse_boltz_schema(
            name,
            data_use,
            ccd_lib,
            ccd_path_exp,
            boltz_2=boltz2,
        )
        _, structure = get_batch(
            target,
            ccd_path_exp,
            ccd_lib,
            pocket_conditioning=pocket_conditioning,
            boltz_model_version=boltz_model_version,
        )
        output, structure = run_prediction(
            data_use,
            binder_id,
            logmd=False,
            name=name,
            ccd_lib=ccd_lib,
            ccd_path=ccd_path_exp,
            boltz_model=model,
            randomly_kill_helix_feature=False,
            negative_helix_constant=0.1,
            device=device,
            boltz_model_version=boltz_model_version,
            pocket_conditioning=pocket_conditioning,
        )
        if output.get("exception"):
            raise RuntimeError(f"Boltz prediction failed for {stem}")

        n_atom = int(structure.atoms["coords"].shape[0])
        coords = output["coords"][0].detach().cpu().numpy()[:n_atom]
        structure.atoms["coords"] = coords
        plddt_t = output["plddt"][0].detach().cpu().numpy()[:n_atom]
        mmcif_text = to_mmcif(structure, torch.as_tensor(plddt_t), boltz2=boltz2)

        iptm = float(output["iptm"][0].detach().cpu().item()) if "iptm" in output else 0.0
        pae_t = output.get("pae")
        if pae_t is not None:
            pae_np = pae_t[0].detach().cpu().numpy()
            while pae_np.ndim > 2:
                pae_np = pae_np[..., 0]
        else:
            pae_np = None

        model_dir = Path(out_root) / stem
        _write_af3_style_confidence_artifacts(
            model_dir, stem, mmcif_text, iptm, plddt_t, pae_np
        )

    for ypath in yaml_paths:
        stem = ypath.stem
        try:
            with open(ypath) as f:
                holo = yaml.safe_load(f)
            _predict_and_write(holo, af_output_dir, stem)
            apo = _boltz_build_apo_dict(holo, binder_id)
            if apo:
                _predict_and_write(apo, af_output_apo_dir, stem)
            else:
                print(f"Warning: no apo YAML for {stem}; skipping apo Boltz prediction.")
        except Exception as e:
            print(f"ERROR: Boltz validation failed for {ypath}: {e}")

    from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb

    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"
    # Cross-validation: YAMLs are already high-ipTM *design* hits. Chai/Boltz ipTM
    # can be < 0.5 while still being valid folds — use i_ptm_cutoff=0 so every holo
    # CIF becomes a PDB for Rosetta and holo–apo RMSD (high_iptm=True keeps CSV).
    convert_cif_files_to_pdb(
        af_output_dir,
        af_pdb_dir,
        af_dir=True,
        high_iptm=high_iptm,
        **({"i_ptm_cutoff": 0.0} if high_iptm else {}),
    )
    if any(Path(af_output_apo_dir).rglob("*.cif")):
        convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True, high_iptm=False)
    else:
        os.makedirs(af_pdb_dir_apo, exist_ok=True)

    if any(Path(af_pdb_dir).glob("*.pdb")):
        calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_id)
    else:
        print("No PDBs after Boltz validation conversion. Skipping holo-apo RMSD.")

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo


def run_chai_validation_step(
    yaml_dir: str,
    ligandmpnn_dir: str,
    folder: Any,
    binder_id: str = "A",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    high_iptm: bool = True,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Re-predict each high-ipTM YAML with Chai (holo + apo). Same output layout as AF3.
    `folder` must be a chai_ph.predict.ChaiFolder instance (already on the right device).
    """
    work_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if work_root not in sys.path:
        sys.path.insert(0, work_root)

    af_output_dir = os.path.join(ligandmpnn_dir, "02_design_final_af3")
    af_output_apo_dir = os.path.join(ligandmpnn_dir, "02_design_final_af3_apo")
    os.makedirs(af_output_dir, exist_ok=True)
    os.makedirs(af_output_apo_dir, exist_ok=True)

    yaml_paths = sorted(Path(yaml_dir).glob("*.yaml"))
    if not yaml_paths:
        print("No YAML files for Chai validation. Skipping.")
        return af_output_dir, af_output_apo_dir, None, None

    import torch

    def _fold_and_write(yaml_data: Dict[str, Any], out_root: str, stem: str) -> None:
        chains = _chai_chain_tuples_from_yaml(yaml_data, binder_id)
        folder.prep_inputs(chains)
        folder.get_embeddings()
        folder.run_trunk(num_trunk_recycles=num_trunk_recycles, template_weight=None)
        folder.sample(
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=1,
            use_alignment=True,
            viewer=None,
            render_freq=max(num_diffn_timesteps, 10**9),
        )
        if folder.state.result is None:
            raise RuntimeError(f"Chai produced no result for {stem}")

        model_dir = Path(out_root) / stem
        model_dir.mkdir(parents=True, exist_ok=True)
        cif_path = model_dir / f"{stem}_model.cif"
        folder.save(cif_path, use_entity_names=False)

        res = folder.state.result
        iptm_t = res.get("iptm")
        iptm = float(iptm_t.item()) if iptm_t is not None else 0.0
        atom_plddts = _chai_plddt_per_real_atom(folder, res)
        pae = res["pae"].detach().cpu().numpy() if "pae" in res else None

        mmcif_text = cif_path.read_text()
        _write_af3_style_confidence_artifacts(model_dir, stem, mmcif_text, iptm, atom_plddts, pae)
        folder.full_cleanup()

    for ypath in yaml_paths:
        stem = ypath.stem
        try:
            with open(ypath) as f:
                holo = yaml.safe_load(f)
            _fold_and_write(holo, af_output_dir, stem)

            binder_block = next(
                (
                    s["protein"]
                    for s in holo.get("sequences", [])
                    if "protein" in s and binder_id in s["protein"].get("id", [])
                ),
                None,
            )
            if binder_block:
                seq = binder_block["sequence"]
                cyc = bool(binder_block.get("cyclic", False))
                apo_chains = _chai_apo_tuples(binder_id, seq, cyc)
                folder.prep_inputs(apo_chains)
                folder.get_embeddings()
                folder.run_trunk(num_trunk_recycles=num_trunk_recycles, template_weight=None)
                folder.sample(
                    num_diffn_timesteps=num_diffn_timesteps,
                    num_diffn_samples=1,
                    use_alignment=True,
                    viewer=None,
                    render_freq=max(num_diffn_timesteps, 10**9),
                )
                if folder.state.result is None:
                    print(f"Warning: Chai apo failed for {stem}")
                else:
                    model_dir = Path(af_output_apo_dir) / stem
                    model_dir.mkdir(parents=True, exist_ok=True)
                    cif_path = model_dir / f"{stem}_model.cif"
                    folder.save(cif_path, use_entity_names=False)
                    res = folder.state.result
                    iptm_t = res.get("iptm")
                    iptm = float(iptm_t.item()) if iptm_t is not None else 0.0
                    atom_plddts = _chai_plddt_per_real_atom(folder, res)
                    pae = res["pae"].detach().cpu().numpy() if "pae" in res else None
                    mmcif_text = cif_path.read_text()
                    _write_af3_style_confidence_artifacts(
                        model_dir, stem, mmcif_text, iptm, atom_plddts, pae
                    )
                folder.full_cleanup()
            else:
                print(f"Warning: no binder protein for apo Chai validation: {stem}")
        except Exception as e:
            print(f"ERROR: Chai validation failed for {ypath}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb

    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"
    convert_cif_files_to_pdb(
        af_output_dir,
        af_pdb_dir,
        af_dir=True,
        high_iptm=high_iptm,
        **({"i_ptm_cutoff": 0.0} if high_iptm else {}),
    )
    if any(Path(af_output_apo_dir).rglob("*.cif")):
        convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True, high_iptm=False)
    else:
        os.makedirs(af_pdb_dir_apo, exist_ok=True)

    if any(Path(af_pdb_dir).glob("*.pdb")):
        calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_id)
    else:
        print("No PDBs after Chai validation conversion. Skipping holo-apo RMSD.")

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo
