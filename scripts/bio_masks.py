"""
bio_masks.py  v3
================
Constructs boolean (float32) connectivity masks for the three integration
stages of the biologically-constrained ANN (Poirazi 2025).

Two-channel architecture (barn owl parallel pathways)
------------------------------------------------------
  Channel 1 — ITD pathway  (Nucleus Laminaris -> ICc lateral shell)
      Each spine is tuned to one ITD window AND receives the full
      frequency span. This implements the joint ITD x frequency map
      of the NL/ICc-ls — where place (ITD) and frequency are
      co-registered on the same spatial map.

  Channel 2 — ILD pathway  (Nucleus Angularis -> VLVp/LLDp -> ICc core)
      Each spine is tuned to one ILD window only.
      No frequency input — ILD is computed from broadband level
      differences in a midbrain region that does not carry fine
      frequency-place information.

  The two channels project independently through the dendrite and soma
  stages, converging only at the all-to-all output layer.

Sweepable axes
--------------
  Size:
    n_spines     : total spines (split evenly between channels)
    n_dendrites  : total dendrites (split evenly between channels)
    n_soma       : total soma    (split evenly between channels)

  Topology:
    itd_spine_frac  : fraction of spines allocated to ITD channel [0.5]
    overlap         : receptive-field overlap within each channel  [0.0]
    dendrite_rule   : topographic | interleaved | random
    channel_dend_split : if True, dendrites are split by channel
                         (Ch1 dendrites only receive Ch1 spines, etc.)
                         if False, all spines project to all dendrites
                         (tests whether channel separation at spine
                          level alone is sufficient)

Usage
-----
    from bio_masks import build_masks, sweep_masks
    from bio_masks import summary_dataframe

    # Single config
    masks = build_masks(n_spines=128, n_dendrites=4, n_soma=128)

    # Size sweep
    all_masks = sweep_masks()

    # Topology sweep
    all_masks = sweep_masks_topology(
        spine_sizes=[128, 256],
        dendrite_sizes=[4, 8],
        soma_sizes=[128],
    )
    df = summary_dataframe(all_masks)
"""

import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# Topology catalogues
# ---------------------------------------------------------------------------

# Fraction of spines allocated to ITD channel; ILD gets the remainder
ITD_SPINE_FRAC_CONFIGS = {
    "equal_split":  0.5,    # 50/50 — default
    "itd_heavy":    0.67,   # ITD gets 2/3 (more complex joint tuning)
    "ild_heavy":    0.33,   # ILD gets 2/3
}

# Receptive-field overlap within each channel
OVERLAP_CONFIGS = {
    "strict": 0.0,
    "25pct":  0.25,
    "50pct":  0.50,
}

# How spines are assigned to dendrites within each channel
DENDRITE_ASSIGN_CONFIGS = [
    "topographic",   # contiguous blocks (default)
    "interleaved",   # every Nth spine -> same dendrite
    "random",        # fixed random assignment (sparsity control)
]

# Whether dendrites respect channel boundaries
CHANNEL_DEND_SPLIT_CONFIGS = {
    "split":   True,    # Ch1 spines -> Ch1 dendrites only (default)
    "merged":  False,   # all spines can project to any dendrite
}


# ---------------------------------------------------------------------------
# Core mask builder
# ---------------------------------------------------------------------------

def build_masks(
    # Feature dimensions
    n_time=8, n_freq=32, n_itd=32, n_ild=32,
    # Layer sizes
    n_spines=128, n_dendrites_per_soma=4, n_soma=128,
    # Topology
    itd_spine_frac=0.5,
    overlap=0.0,
    dendrite_rule="topographic",
    channel_dend_split=True,
    random_seed=0,
):
    """
    Build all three connectivity masks.

    Parameters
    ----------
    n_dendrites_per_soma : each soma has this many dendrites (total dendrites = n_dendrites_per_soma * n_soma)

    Returns
    -------
    dict:
        'input_spine'    : float32 (input_dim, n_spines)
        'spine_dendrite' : float32 (n_spines, total_dendrites)
        'dendrite_soma'  : float32 (total_dendrites, n_soma)  [each soma receives only from its own dendrites]
        'summary'        : diagnostic dict
    """
    input_dim = n_time * (n_freq + n_itd + n_ild)
    total_dendrites = n_dendrites_per_soma * n_soma  # Each soma has its own dendrite band

    # Input index ranges
    freq_start = 0
    freq_end   = n_time * n_freq
    itd_start  = freq_end
    itd_end    = itd_start + n_time * n_itd
    ild_start  = itd_end
    ild_end    = ild_start + n_time * n_ild
    assert ild_end == input_dim

    # Channel spine boundaries
    n_itd_spines = max(1, int(round(itd_spine_frac * n_spines)))
    n_ild_spines = max(1, n_spines - n_itd_spines)
    n_itd_spines = n_spines - n_ild_spines   # recompute to guarantee sum == n_spines

    ch1_start, ch1_end = 0,           n_itd_spines
    ch2_start, ch2_end = n_itd_spines, n_spines

    # Channel dendrite boundaries (scaled to total_dendrites, used when channel_dend_split=True)
    n_itd_dends = max(1, int(round(itd_spine_frac * total_dendrites)))
    n_ild_dends = max(1, total_dendrites - n_itd_dends)
    n_itd_dends = total_dendrites - n_ild_dends

    cd1_start, cd1_end = 0,              n_itd_dends
    cd2_start, cd2_end = n_itd_dends,    total_dendrites

    # Channel soma boundaries (mirrors dendrite split)
    n_itd_soma = max(1, int(round(itd_spine_frac * n_soma)))
    n_ild_soma = max(1, n_soma - n_itd_soma)
    n_itd_soma = n_soma - n_ild_soma

    cs1_start, cs1_end = 0,           n_itd_soma
    cs2_start, cs2_end = n_itd_soma,  n_soma

    channels = {
        "ITD": {
            "spines":    (ch1_start, ch1_end),
            "dendrites": (cd1_start, cd1_end),
            "soma":      (cs1_start, cs1_end),
        },
        "ILD": {
            "spines":    (ch2_start, ch2_end),
            "dendrites": (cd2_start, cd2_end),
            "soma":      (cs2_start, cs2_end),
        },
    }

    # ------------------------------------------------------------------
    # Mask 1: Input -> Toric Spines
    # ------------------------------------------------------------------
    mask_input_spine = np.zeros((input_dim, n_spines), dtype=np.float32)

    # Channel 1 (ITD): joint ITD x frequency tuning
    # Each spine receives the full frequency span PLUS its topographic ITD window
    n_itd_inputs = itd_end - itd_start
    for s_local in range(n_itd_spines):
        s_global = ch1_start + s_local

        # Full frequency broadcast — NL/ICc-ls preserves tonotopic registration
        mask_input_spine[freq_start:freq_end, s_global] = 1.0

        # Topographic ITD window for this spine
        i0, i1 = _window(s_local, n_itd_spines, n_itd_inputs, overlap)
        mask_input_spine[itd_start + i0 : itd_start + i1, s_global] = 1.0

    # Channel 2 (ILD): ILD only, no frequency
    # VLVp/LLDp computes broadband level differences — no fine freq structure
    n_ild_inputs = ild_end - ild_start
    for s_local in range(n_ild_spines):
        s_global = ch2_start + s_local

        i0, i1 = _window(s_local, n_ild_spines, n_ild_inputs, overlap)
        mask_input_spine[ild_start + i0 : ild_start + i1, s_global] = 1.0

    # ------------------------------------------------------------------
    # Mask 2: Toric Spines -> Dendrites
    # ------------------------------------------------------------------
    mask_spine_dendrite = np.zeros((n_spines, total_dendrites), dtype=np.float32)

    if channel_dend_split:
        # Each channel's spines project only to that channel's dendrites
        _assign_spines_to_dends(
            mask_spine_dendrite,
            spine_start=ch1_start, spine_end=ch1_end,
            dend_start=cd1_start,  dend_end=cd1_end,
            rule=dendrite_rule, seed=random_seed,
        )
        _assign_spines_to_dends(
            mask_spine_dendrite,
            spine_start=ch2_start, spine_end=ch2_end,
            dend_start=cd2_start,  dend_end=cd2_end,
            rule=dendrite_rule, seed=random_seed + 1,
        )
    else:
        # Merged: all spines project to all dendrites 
        _assign_spines_to_dends(
            mask_spine_dendrite,
            spine_start=0, spine_end=n_spines,
            dend_start=0,  dend_end=total_dendrites,
            rule=dendrite_rule, seed=random_seed,
        )

    # ------------------------------------------------------------------
    # Mask 3: Dendrites -> Soma (each soma receives only from its own dendrite band)
    # ------------------------------------------------------------------
    mask_dendrite_soma = np.zeros((total_dendrites, n_soma), dtype=np.float32)
    _assign_dends_to_soma_per_soma(mask_dendrite_soma, n_dendrites_per_soma, n_soma)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    a1 = int(mask_input_spine.sum())
    a2 = int(mask_spine_dendrite.sum())
    a3 = int(mask_dendrite_soma.sum())
    ta = a1 + a2 + a3
    aa = input_dim * n_spines + n_spines * total_dendrites + total_dendrites * n_soma

    # Per-channel active weights in mask 1
    ch1_active = int(mask_input_spine[:, ch1_start:ch1_end].sum())
    ch2_active = int(mask_input_spine[:, ch2_start:ch2_end].sum())

    summary = dict(
        config=dict(input_dim=input_dim, n_spines=n_spines,
                    n_dendrites_per_soma=n_dendrites_per_soma, total_dendrites=total_dendrites, n_soma=n_soma),
        topology=dict(
            itd_spine_frac=itd_spine_frac, overlap=overlap,
            dendrite_rule=dendrite_rule,
            channel_dend_split=channel_dend_split,
        ),
        channels=channels,
        channel_sizes=dict(
            itd_spines=n_itd_spines, ild_spines=n_ild_spines,
            itd_dends=n_itd_dends,   ild_dends=n_ild_dends,
            itd_soma=n_itd_soma,     ild_soma=n_ild_soma,
        ),
        active_weights=dict(
            input_spine=a1, spine_dendrite=a2,
            dendrite_soma=a3, total=ta,
            ch1_input_spine=ch1_active,
            ch2_input_spine=ch2_active,
        ),
        all2all_weights=dict(total=aa),
        density=dict(
            input_spine=round(a1 / (input_dim * n_spines), 4),
            spine_dendrite=round(a2 / (n_spines * total_dendrites), 4),
            dendrite_soma=round(a3 / (total_dendrites * n_soma), 4),
            overall=round(ta / aa, 4),
        ),
        weight_reduction_pct=round((1 - ta / aa) * 100, 1),
        spines_per_dendrite=[int(mask_spine_dendrite[:, d].sum())
                             for d in range(total_dendrites)],
        soma_per_dendrite=[int(mask_dendrite_soma[d, :].sum())
                           for d in range(total_dendrites)],
    )

    return dict(
        input_spine=mask_input_spine,
        spine_dendrite=mask_spine_dendrite,
        dendrite_soma=mask_dendrite_soma,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Full topology x size sweep
# ---------------------------------------------------------------------------

def sweep_masks(#_topology(
    itd_spine_frac_names=None,
    overlap_names=None,
    dendrite_rules=None,
    channel_dend_split_names=None,
    spine_sizes=[128, 256, 512],
    dendrite_per_soma_sizes=[1, 2, 4, 8, 16],
    soma_sizes=[128, 256, 512],
    n_time=8, n_freq=32, n_itd=32, n_ild=32,
):
    """
    Build masks for every topology x size combination.

    Key: (itd_frac_name, overlap_name, dendrite_rule,
          channel_dend_split_name, n_spines, n_dendrites_per_soma_sizes, n_soma)

    Example
    -------
        all_masks = sweep_masks(
            itd_spine_frac_names    = ['equal_split', 'itd_heavy'],
            overlap_names           = ['strict', '25pct'],
            dendrite_rules          = ['topographic', 'interleaved'],
            channel_dend_split_names= ['split', 'merged'],
            spine_sizes             = [128, 256],
            dendrite_per_soma_sizes = [4, 8],
            soma_sizes              = [128],
        )
    """
    ifn = itd_spine_frac_names     or list(ITD_SPINE_FRAC_CONFIGS.keys())
    ovn = overlap_names            or list(OVERLAP_CONFIGS.keys())
    drn = dendrite_rules           or DENDRITE_ASSIGN_CONFIGS
    cdn = channel_dend_split_names or list(CHANNEL_DEND_SPLIT_CONFIGS.keys())

    results = {}
    for ifn_, ovn_, drn_, cdn_, ns, nd, nso in product(
        ifn, ovn, drn, cdn, spine_sizes, dendrite_per_soma_sizes, soma_sizes
    ):
        results[(ifn_, ovn_, drn_, cdn_, ns, nd, nso)] = build_masks(
            n_time=n_time, n_freq=n_freq, n_itd=n_itd, n_ild=n_ild,
            n_spines=ns, n_dendrites_per_soma=nd, n_soma=nso,
            itd_spine_frac=ITD_SPINE_FRAC_CONFIGS[ifn_],
            overlap=OVERLAP_CONFIGS[ovn_],
            dendrite_rule=drn_,
            channel_dend_split=CHANNEL_DEND_SPLIT_CONFIGS[cdn_],
        )
    return results


# ---------------------------------------------------------------------------
# Summary utilities
# ---------------------------------------------------------------------------

def summary_dataframe(all_masks):
    """
    Convert any sweep dict into a flat pandas DataFrame.
    Works for size-only keys (ns, nd, nso) and full topology keys.
    """
    import pandas as pd
    rows = []
    for key, masks in all_masks.items():
        s = masks["summary"]
        row = {}
        if len(key) == 3:
            row.update(dict(zip(["n_spines", "n_dendrites", "n_soma"], key)))
            row.update(itd_frac_name="equal_split", overlap_name="strict",
                       dendrite_rule="topographic", channel_dend_split="split")
        else:
            row.update(dict(zip(
                ["itd_frac_name", "overlap_name", "dendrite_rule",
                 "channel_dend_split", "n_spines", "n_dendrites", "n_soma"],
                key,
            )))
        cs = s["channel_sizes"]
        row.update({
            "input_dim":            s["config"]["input_dim"],
            "n_itd_spines":         cs["itd_spines"],
            "n_ild_spines":         cs["ild_spines"],
            "n_itd_dends":          cs["itd_dends"],
            "n_ild_dends":          cs["ild_dends"],
            "active_total":         s["active_weights"]["total"],
            "all2all_total":        s["all2all_weights"]["total"],
            "weight_reduction_pct": s["weight_reduction_pct"],
            "density_overall":      s["density"]["overall"],
            "density_input_spine":  s["density"]["input_spine"],
            "density_spine_dend":   s["density"]["spine_dendrite"],
            "density_dend_soma":    s["density"]["dendrite_soma"],
            "ch1_active_w":         s["active_weights"]["ch1_input_spine"],
            "ch2_active_w":         s["active_weights"]["ch2_input_spine"],
        })
        rows.append(row)

    col_order = [
        "itd_frac_name", "overlap_name", "dendrite_rule", "channel_dend_split",
        "n_spines", "n_dendrites", "n_soma",
        "n_itd_spines", "n_ild_spines", "n_itd_dends", "n_ild_dends",
        "active_total", "all2all_total", "weight_reduction_pct",
        "density_overall", "density_input_spine",
        "density_spine_dend", "density_dend_soma",
        "ch1_active_w", "ch2_active_w",
    ]
    return pd.DataFrame(rows)[col_order].sort_values(
        ["itd_frac_name", "overlap_name", "dendrite_rule",
         "channel_dend_split", "n_spines", "n_dendrites", "n_soma"]
    ).reset_index(drop=True)


def print_sweep_summary(all_masks):
    print(summary_dataframe(all_masks).to_string(index=False))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _window(s_local, n_spines, n_inputs, overlap):
    """Compute [i0, i1) input window for spine s_local."""
    window = n_inputs / n_spines
    step   = max(window * (1.0 - overlap), 1.0)
    i0 = min(int(round(s_local * step)),          n_inputs - 1)
    i1 = min(int(round(s_local * step + window)), n_inputs)
    if i0 >= i1:
        i1 = i0 + 1
    return i0, i1


def _assign_spines_to_dends(mask, spine_start, spine_end,
                             dend_start, dend_end, rule, seed=0):
    """Assign spines [spine_start, spine_end) to dendrites [dend_start, dend_end)."""
    n_sp = spine_end - spine_start
    n_de = dend_end  - dend_start
    if n_sp == 0 or n_de == 0:
        return

    if rule == "topographic":
        for s_local in range(n_sp):
            d0 = int(round(s_local       * n_de / n_sp))
            d1 = int(round((s_local + 1) * n_de / n_sp))
            d  = min(d0, n_de - 1)   # each spine -> one dendrite
            mask[spine_start + s_local, dend_start + d] = 1.0

    elif rule == "interleaved":
        for s_local in range(n_sp):
            mask[spine_start + s_local, dend_start + (s_local % n_de)] = 1.0

    elif rule == "random":
        rng = np.random.default_rng(seed)
        for s_local, d_local in enumerate(rng.integers(0, n_de, size=n_sp)):
            mask[spine_start + s_local, dend_start + d_local] = 1.0


def _assign_dends_to_soma_per_soma(mask, n_dendrites_per_soma, n_soma):
    """Each soma receives from exactly one band of n_dendrites_per_soma dendrites.
    
    Soma i receives from dendrites [i*n_dendrites_per_soma, (i+1)*n_dendrites_per_soma).
    This creates a block-diagonal structure.
    """
    for soma_idx in range(n_soma):
        dend_start = soma_idx * n_dendrites_per_soma
        dend_end = dend_start + n_dendrites_per_soma
        mask[dend_start:dend_end, soma_idx] = 1.0


# ---------------------------------------------------------------------------
# Validation + demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("Two-channel barn owl mask — validation")
    print("=" * 60)

    masks = build_masks(n_spines=128, n_dendrites_per_soma=4, n_soma=128)
    s   = masks["summary"]
    m1  = masks["input_spine"]
    m2  = masks["spine_dendrite"]
    m3  = masks["dendrite_soma"]
    ch  = s["channels"]
    cs  = s["channel_sizes"]

    ch1_s, ch1_e = ch["ITD"]["spines"]
    ch2_s, ch2_e = ch["ILD"]["spines"]
    cd1_s, cd1_e = ch["ITD"]["dendrites"]
    cd2_s, cd2_e = ch["ILD"]["dendrites"]

    n_freq_inputs = 8 * 32   # 256

    print(f"\nChannel 1 (ITD): {cs['itd_spines']} spines, "
          f"{cs['itd_dends']} dendrites, {cs['itd_soma']} soma")
    print(f"Channel 2 (ILD): {cs['ild_spines']} spines, "
          f"{cs['ild_dends']} dendrites, {cs['ild_soma']} soma")

    print("\n--- Biological rule checks ---\n")

    # 1. All ITD spines receive full frequency broadcast
    for s_idx in range(ch1_s, ch1_e):
        assert m1[:n_freq_inputs, s_idx].sum() == n_freq_inputs, \
            f"FAIL spine {s_idx} missing freq broadcast"
    print(f"PASS  All {cs['itd_spines']} ITD spines receive full frequency span")

    # 2. ILD spines receive NO frequency
    assert m1[:n_freq_inputs, ch2_s:ch2_e].sum() == 0, \
        "FAIL ILD spines receiving frequency input"
    print(f"PASS  All {cs['ild_spines']} ILD spines receive no frequency input")

    # 3. ITD inputs do not reach ILD spines
    assert m1[n_freq_inputs:n_freq_inputs*2, ch2_s:ch2_e].sum() == 0, \
        "FAIL ITD leaking into ILD channel"
    print("PASS  ITD inputs do not reach ILD channel")

    # 4. ILD inputs do not reach ITD spines
    assert m1[n_freq_inputs*2:, ch1_s:ch1_e].sum() == 0, \
        "FAIL ILD leaking into ITD channel"
    print("PASS  ILD inputs do not reach ITD channel")

    # 5. Channel dendrite separation
    assert m2[ch1_s:ch1_e, cd2_s:cd2_e].sum() == 0, \
        "FAIL Ch1 spines projecting to Ch2 dendrites"
    assert m2[ch2_s:ch2_e, cd1_s:cd1_e].sum() == 0, \
        "FAIL Ch2 spines projecting to Ch1 dendrites"
    print("PASS  Spine->dendrite connections respect channel boundaries")

    # 6. Each spine connects to exactly one dendrite
    assert (m2.sum(axis=1) == 1).all(), "FAIL spine->1 dendrite"
    print("PASS  Each spine connects to exactly 1 dendrite")


    print(f"\n  Active weights:    {s['active_weights']['total']:,}")
    print(f"  All-to-all total:  {s['all2all_weights']['total']:,}")
    print(f"  Weight reduction:  {s['weight_reduction_pct']}%")
    print(f"  Overall density:   {s['density']['overall']}")

    print("\n--- Topology sweep summary (subset) ---\n")
    topo = sweep_masks(
        itd_spine_frac_names    = ["equal_split", "itd_heavy"],
        overlap_names           = ["strict", "25pct"],
        dendrite_rules          = ["topographic", "interleaved"],
        channel_dend_split_names= ["split", "merged",'random'],
        spine_sizes=[128], dendrite_per_soma_sizes=[4], soma_sizes=[128],
    )
    df = summary_dataframe(topo)
    print(df[[
        "itd_frac_name", "overlap_name", "dendrite_rule",
        "channel_dend_split", "n_itd_spines", "n_ild_spines",
        "active_total", "weight_reduction_pct", "density_overall"
    ]].to_string(index=False))
    print(f"\n{len(topo)} configs in subset")

    full = sweep_masks()
    print(f"Full topology x all sizes: {len(full)} configs")