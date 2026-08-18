import json, os

with open(r'D:\THEMIS\human_anno_DiT-XL2-500\User_1_final_annotations.json', 'r', encoding='utf-8') as f:
    human_data = json.load(f)

matches = []
for img_name, h_info in human_data.items():
    idx = img_name.replace('.png', '')
    report_path = rf'D:\THEMIS\c2i_faster\output_DiT_ref_cap_1\final_reports\final_evaluation_report_{idx}.json'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        h_align = h_info['scores']['alignment_score']
        h_artifact = h_info['scores']['artifact_score']
        s_align = report.get('alignment_score', 0)
        s_artifact = report.get('artifact_score', 0)
        diff = abs(h_align - s_align) + abs(h_artifact - s_artifact)
        matches.append({
            'img': img_name, 'idx': idx,
            'h_align': h_align, 'h_artifact': h_artifact,
            's_align': s_align, 's_artifact': s_artifact,
            'diff': diff, 'class': h_info['class_name']
        })

# Find diverse examples - exclude already used (000038, 000008)
used = {'000038', '000008'}
remaining = [m for m in matches if m['idx'] not in used]

# Pick diverse score profiles
# 1. Low alignment, high artifact (semantic failure but good quality)
low_align_high_art = [m for m in remaining if m['h_align'] <= 1.5 and m['h_artifact'] >= 3.5]
low_align_high_art.sort(key=lambda x: x['diff'])

# 2. High alignment, low artifact (correct class but bad quality)
high_align_low_art = [m for m in remaining if m['h_align'] >= 4.5 and m['h_artifact'] <= 2.5]
high_align_low_art.sort(key=lambda x: x['diff'])

# 3. Both mid-range
mid_both = [m for m in remaining if 2.0 <= m['h_align'] <= 4.0 and 2.0 <= m['h_artifact'] <= 4.0]
mid_both.sort(key=lambda x: x['diff'])

# 4. Both low
both_low = [m for m in remaining if m['h_align'] <= 2.0 and m['h_artifact'] <= 2.0]
both_low.sort(key=lambda x: x['diff'])

# 5. High both (but not perfect)
high_both = [m for m in remaining if m['h_align'] >= 4.0 and m['h_artifact'] >= 4.0]
high_both.sort(key=lambda x: x['diff'])

print("=== Low alignment, high artifact ===")
for m in low_align_high_art[:5]:
    print(f"  {m['idx']:>6s} | {m['class']:40s} | H:A={m['h_align']:.1f} Au={m['h_artifact']:.1f} | S:A={m['s_align']:.2f} Au={m['s_artifact']:.2f} | diff={m['diff']:.2f}")

print("\n=== High alignment, low artifact ===")
for m in high_align_low_art[:5]:
    print(f"  {m['idx']:>6s} | {m['class']:40s} | H:A={m['h_align']:.1f} Au={m['h_artifact']:.1f} | S:A={m['s_align']:.2f} Au={m['s_artifact']:.2f} | diff={m['diff']:.2f}")

print("\n=== Both mid-range ===")
for m in mid_both[:5]:
    print(f"  {m['idx']:>6s} | {m['class']:40s} | H:A={m['h_align']:.1f} Au={m['h_artifact']:.1f} | S:A={m['s_align']:.2f} Au={m['s_artifact']:.2f} | diff={m['diff']:.2f}")

print("\n=== Both low ===")
for m in both_low[:5]:
    print(f"  {m['idx']:>6s} | {m['class']:40s} | H:A={m['h_align']:.1f} Au={m['h_artifact']:.1f} | S:A={m['s_align']:.2f} Au={m['s_artifact']:.2f} | diff={m['diff']:.2f}")

print("\n=== High both (not perfect) ===")
for m in high_both[:5]:
    print(f"  {m['idx']:>6s} | {m['class']:40s} | H:A={m['h_align']:.1f} Au={m['h_artifact']:.1f} | S:A={m['s_align']:.2f} Au={m['s_artifact']:.2f} | diff={m['diff']:.2f}")
