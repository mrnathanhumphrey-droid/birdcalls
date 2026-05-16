# Bird Song Substrate — Pre-Registration DRAFT

**Status:** DRAFT — NOT YET LOCKED. Will be locked via commit-hash before any acoustic-feature or outcome analysis.

**Substrate:** 7 of the Paper 6 methodology corpus. Bioacoustic test of the partial-pooling-of-residual-classes framework.

**Geographic frame:** CONUS + North America breeding range.

---

## Open scoping decisions (TO BE FILLED before lock)

These are the load-bearing operationalization decisions. They MUST be locked before any acoustic-feature extraction or outcome analysis.

### A. Pilot species (3-5)

Candidates from the dialect-literature pantheon:

- [ ] **White-crowned sparrow** (*Zonotrichia leucophrys*) — Marler's classical dialect work, Pacific subspecies; geographic dialects well-documented
- [ ] **Song sparrow** (*Melospiza melodia*) — wide range, individual + regional dialect variation, ~52 subspecies
- [ ] **Bewick's wren** (*Thryomanes bewickii*) — population-level dialect work, song-learning literature
- [ ] **Indigo bunting** (*Passerina cyanea*) — well-documented song-sharing networks
- [ ] **Carolina chickadee** (*Poecile carolinensis*) — fee-bee call, dialect zones documented
- [ ] **Black-capped chickadee** (*Poecile atricapillus*) — fee-bee, regional variation
- [ ] **Marsh wren** (*Cistothorus palustris*) — eastern/western dialect divergence well-published

**Decision:** which 3-5? (user pick)

### B. Geographic-region partition for Arm 1

How is "region" defined for the dialect-geography cells? Options:

- Bird Conservation Regions (BCR, USFWS standard, 36 NA regions)
- L3 ecoregions (EPA, ~100 NA regions, finer)
- State-county aggregates (corpus-method-consistent with gun violence)
- Climate zones (Köppen, biogeographic)
- Custom dialect-cluster boundaries from prior literature (species-specific)

**Decision:** which partition? (user pick — affects cell N and degeneracy)

### C. Song-feature distance measure

For Arm 1 outcome ("song-feature distance from species-population centroid"):

- Syllable inventory edit distance (Levenshtein on syllable sequence)
- Spectral centroid distribution KL divergence
- Song-bout structural complexity (Shannon entropy of syllable transitions)
- BirdNET embedding-space distance (Cornell pre-trained, 224-dim or larger)
- Multiple-measure composite

**Decision:** which? (BirdNET embedding-space is the lowest-friction implementation; literature-standard syllable-distance is more interpretable)

### D. Community definition for Arm 2

How are "sympatric communities" defined?

- eBird hot-spots at hex resolution (H3 level 5/6/7)
- USFWS BCR + season
- BBS (Breeding Bird Survey) route + year
- Custom co-occurrence-based clustering on eBird checklists

**Decision:** which? (eBird hotspots at H3-5 is the easiest, BBS routes is the standard)

### E. Spectral-band partition for Arm 2

For the acoustic-niche-partition cells:

- Continuous spectral centroid (no partition; treat as covariate)
- 3-band partition (low 0-2kHz, mid 2-6kHz, high 6+kHz)
- Critical bands (Bark scale, 24 bands)
- Octave bands (~10 bands)

**Decision:** which? (3-band is interpretable, critical bands is more biophysically grounded)

### F. Decision rules (locked falsification criteria)

For Arm 1 (dialect-geography residual structure):

- **H_DIALECT_RESIDUAL:** Cell (species × region × dialect) partial-pooling explains ≥ X% additional variance in song-feature distance vs additive (species + region) baseline, with 95% CrI clean positive. *X to lock.*
- **H_ADDITIVE_SUFFICIENT (null):** Cell partial-pooling does NOT add ≥ X% variance; song-feature variation is fully captured by additive species + region effects.

For Arm 2 (acoustic-niche partition):

- **H_NICHE_PARTITION:** Cell (community × spectral-band × time-of-day) partial-pooling explains ≥ Y% additional variance in call-overlap rate vs additive (community + species) baseline, with 95% CrI clean positive. *Y to lock.*
- **H_NO_PARTITION (null):** Cell partial-pooling does not exceed Y%.

**Decisions:** X, Y thresholds + 95% CrI rule + handling of edge cases (e.g., what if 90% CrI is clean but 95% spans 0?).

### G. Sample size / power

Cell-availability scoping (metadata-only, no outcome inspection) needed:

- How many xeno-canto recordings per pilot species in CONUS?
- How many BCR/region cells have ≥ N recordings (N to be set, e.g. ≥ 10)?
- How many eBird checklists per community?

Output: cell-availability report analogous to gun violence's `cell_availability_v8_cem_proper.md`, locked BEFORE pre-reg lock.

---

## Pre-registration discipline

- This document is in DRAFT until the open scoping decisions above are filled
- Cell-availability scoping (metadata only, no outcome inspection) happens BEFORE lock
- Once locked, decisions are immutable. Any deviation must be logged in a `DEVIATION_LOG.md` with reason
- Commit hash + timestamp serves as priority date
- Public GitHub commit chain as audit trail

## Robustness arms (queued, pre-reg locked alongside main analysis)

- **Source replication:** swap xeno-canto for Macaulay Library on a subset, compare coefficients
- **Sample resampling:** k-fold cell-stratified split of recordings, refit, compare
- **Geographic refinement:** swap BCR for L3 ecoregion (or whichever finer partition is feasible), compare

Same 3-axis robustness pattern as gun violence substrate.

---

## Related

- [[gun_violence_state_2026_05_16]] — substrate 6, established the decoupled-cells + pre-reg + 3-axis robustness pattern this substrate inherits
- [[methodology_corpus_paper6_locked]] — corpus framework
- [[corpus_qb_2026_05_16]] — current corpus state
