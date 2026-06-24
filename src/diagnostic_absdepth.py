# =============================================================================
# DIAGNOSTIC CELL: Absorption depth distribution + spectral shape around 1640nm
# Run this in 01_preprocessing.ipynb AFTER load_hdf5() (hdf5_data in memory)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

wl   = hdf5_data['wavelengths']
refl = hdf5_data['reflectance']   # shape (426, H, W)

# ============================================================
# 1. Identify exact band indices around 1640nm region
# ============================================================
region_mask = (wl >= 1300) & (wl <= 2000)
region_wl   = wl[region_mask]
print("Bands in 1300-2000nm region:")
for i, w in enumerate(region_wl):
    idx = int(np.where(wl == w)[0][0])
    print(f"  band {idx:3d} : {w:.1f} nm")

# ============================================================
# 2. Mean reflectance spectrum: mangrove vs non-mangrove
#    (use candidate_mask + pseudo-label proxy from indices)
# ============================================================
mvi  = indices['MVI']
ndmi = indices['NDMI']
man_mask = (
    (mvi > thresholds['MVI']) &
    (ndmi > thresholds['NDMI']) &
    candidate_mask
)
non_mask = (~man_mask) & candidate_mask & np.isfinite(mvi)

print(f"\nMangrove proxy pixels : {man_mask.sum():,}")
print(f"Non-mangrove pixels   : {non_mask.sum():,}")

# Mean spectrum per class, region 1300-2000nm
man_spec = np.nanmean(refl[:, man_mask], axis=1)
non_spec = np.nanmean(refl[:, non_mask], axis=1)

# ============================================================
# 3. Plot mean spectrum 1300-2000nm
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.plot(wl[region_mask], man_spec[region_mask],
        color='#2ca02c', label='Mangrove (proxy)')
ax.plot(wl[region_mask], non_spec[region_mask],
        color='#8c564b', label='Non-mangrove')
# Mark current shoulders and center
for nm, label, color in [(1500, 'left shoulder (current)', 'blue'),
                          (1640, 'center (1640nm)',         'red'),
                          (1780, 'right shoulder (current)','blue')]:
    idx = int(np.argmin(np.abs(wl - nm)))
    ax.axvline(wl[idx], color=color, ls='--', lw=1, alpha=0.7, label=label)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean reflectance')
ax.set_title('Mean spectrum 1300-2000nm\n(mangrove vs non-mangrove, candidate zone)')
ax.legend(fontsize=8)

# ============================================================
# 4. Histogram of absorption depth (after outlier fix)
# ============================================================
ax2 = axes[1]
depth_flat = abs_depth[np.isfinite(abs_depth)].ravel()
ax2.hist(depth_flat, bins=100, color='steelblue', alpha=0.7)
ax2.axvline(0, color='red', ls='--', lw=1, label='D=0 (no absorption)')
ax2.axvline(float(np.nanmean(abs_depth)), color='orange', ls='-', lw=1.5,
            label=f'mean={np.nanmean(abs_depth):.3f}')
ax2.set_xlabel('Absorption depth D')
ax2.set_ylabel('Pixel count')
ax2.set_title('Absorption depth distribution\n(1640nm, current shoulders 1500/1780)')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(ROOT / 'outputs' / 'figures' / f'absdepth_diagnostic_{SITE}.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. Key values for shoulder selection
# ============================================================
print("\nReflectance at key wavelengths (mangrove mean):")
for nm in [1400, 1450, 1500, 1550, 1600, 1640, 1680, 1700, 1750, 1780, 1850, 1900]:
    idx = int(np.argmin(np.abs(wl - nm)))
    print(f"  {wl[idx]:.0f} nm : R_mangrove={man_spec[idx]:.4f}  R_nonmang={non_spec[idx]:.4f}")
