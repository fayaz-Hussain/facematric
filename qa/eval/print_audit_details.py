import pickle
import numpy as np

with open('qa/eval/dataset_audit_results.pkl', 'rb') as f:
    res = pickle.load(f)

per_id = res['per_identity_audit']

print("--- Identities with No Face Detected in the 10 Images ---")
no_face_list = []
for k, v in per_id.items():
    if v['no_face'] > 0:
        no_face_list.append((k, v['source'], v['no_face']))
        print(f"  {k} ({v['source']}): {v['no_face']}/10 no-face, {v['single_face']}/10 single-face, {v['multi_face']}/10 multi-face")

print(f"Total identities with >=1 no-face: {len(no_face_list)}")

print("\n--- Identities with Multi-Face Detected in the 10 Images ---")
multi_face_list = []
for k, v in per_id.items():
    if v['multi_face'] > 0:
        multi_face_list.append((k, v['source'], v['multi_face']))
        print(f"  {k} ({v['source']}): {v['multi_face']}/10 multi-face")

print(f"Total identities with >=1 multi-face: {len(multi_face_list)}")

print("\n--- Identities with Near-Duplicates among the 10 Images (Hamming <= 6) ---")
near_dup_list = []
for k, v in per_id.items():
    if v['near_duplicates'] > 0:
        near_dup_list.append((k, v['source'], v['near_duplicates'], v['avg_pairwise_phash_dist']))
        print(f"  {k} ({v['source']}): {v['near_duplicates']} near-duplicate pair(s), avg_dist={v['avg_pairwise_phash_dist']:.1f}")

print(f"Total identities with near-duplicates: {len(near_dup_list)}")

print("\n--- Image Count Distribution Across Shipped 205 Identities ---")
counts = [v['images_available'] for v in per_id.values()]
print(f"Total evaluated identities: {len(counts)}")
print(f"Min available images: {min(counts)}")
print(f"Max available images: {max(counts)}")
print(f"Mean available images: {np.mean(counts):.1f}")
print(f"Median available images: {np.median(counts):.1f}")

western_counts = [v['images_available'] for v in per_id.values() if v['source'] == 'Western']
bollywood_counts = [v['images_available'] for v in per_id.values() if v['source'] == 'Bollywood']

print(f"Western (Pins): min={min(western_counts)}, max={max(western_counts)}, mean={np.mean(western_counts):.1f}, median={np.median(western_counts):.1f}")
print(f"Bollywood: min={min(bollywood_counts)}, max={max(bollywood_counts)}, mean={np.mean(bollywood_counts):.1f}, median={np.median(bollywood_counts):.1f}")

# Photo set analysis: check average pairwise distance distribution
dists = [v['avg_pairwise_phash_dist'] for v in per_id.values()]
print(f"\nAverage Pairwise pHash Distance Across All 205 Identities:")
print(f"Min avg dist: {min(dists):.2f}")
print(f"Max avg dist: {max(dists):.2f}")
print(f"Mean avg dist: {np.mean(dists):.2f}")
print(f"Median avg dist: {np.median(dists):.2f}")

# Identities with lowest average pairwise distance (most homogeneous / single photoset)
print("\nTop 10 Most Homogeneous Identities (Lowest Pairwise Distance):")
sorted_by_homog = sorted(per_id.items(), key=lambda x: x[1]['avg_pairwise_phash_dist'])
for k, v in sorted_by_homog[:10]:
    print(f"  {k} ({v['source']}): avg_dist={v['avg_pairwise_phash_dist']:.2f}, near_dups={v['near_duplicates']}")

# Check identities in all 275 folders with fewer than 20 images
print("\n--- Identities with fewer than 20 images across all 275 folders ---")
fewer_20 = res['fewer_than_20_list']
print(f"Count: {len(fewer_20)}")
