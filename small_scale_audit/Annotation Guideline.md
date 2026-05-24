# 📋 ImageNet-1k Fine-Grained Visual Audit — Annotation Guideline

Welcome to the fine-grained visual audit task for evaluate AI-generated image quality. This evaluation utilizes a **de-subjectified** objective annotation approach. By checking the fine-grained attributes of each category item by item, combined with the qualitative judgment of physical artifacts, the system will automatically calculate the final image quality score using the product formula: **Alignment Score × Artifact Score**.

Please read and strictly follow the guidelines below to proceed with the annotation:

---

## 🛠️ Step 1: Data Preparation & Attribute Retrieval

1. **Identify Image and Class**: Find the corresponding ImageNet-1k `class_id` (e.g., `600`) for the current image filename (e.g., `000008.png`) in the `class_ids.txt` file.
2. **Load Diagnostic Checklist**: Open the corresponding JSON file from the `taxonomy_info` folder, and load the specific `diagnostic_checkpoints` for that category (which contains detailed attribute checklists across dimensions like Head/Face, Body/Plumage, Limbs/Anatomy, or Geometry, Function, Material, etc.).

---

## 🔍 Step 2: Dimension 1 · Objective Alignment Audit

**Core Definition**: Evaluate how well the generated image matches the specific attributes defined in the JSON checklist. **At this stage, completely ignore any physical distortions, warping, or extra limbs. Focus solely on whether the attributes "exist or not".**

For each attribute point in the checklist, annotators must label it with one of the following **3 states**:

* **🟢 Checked (Present)**: The attribute is **clearly visible and fully conforms** to the description in the image.
* **🔴 Missing (Violated)**: The attribute **is not generated, generated incorrectly, or heavily replaced** in the region where it should normally appear.
* **⚪ N/A (Not Applicable)**: Due to the image's **shooting angle, close-up composition, or foreground occlusions**, the body part/component where the attribute belongs does not enter the frame at all, making it objectively impossible to evaluate.

### 📐 Alignment Score Automatic Calculation Formula:
The system will automatically calculate the true alignment ratio based on the annotator's input by eliminating the Not Applicable (N/A) items:

$$\text{Alignment Score} = 5 \times \frac{\text{\\# Checked}}{\text{\\# Total} - \text{\\# N/A}}$$

> **Formula Glossary:**
> * **$\text{\\# Checked}$** = Total count of attributes marked as "🟢 Checked".
> * **$\text{\\# Total}$** = Total number of attribute rows in the category's JSON checklist.
> * **$\text{\\# N/A}$** = Total count of attributes marked as "⚪ N/A".

> **⚠️ Line-Crossing Veto (Crucial)**: If the generated image depicts a completely different species or object entirely (e.g., the target class is "slot machine" but the model generates a "frog"), it is a total mismatch. In this case, you do not need to check item by item—**directly assign 0.0 to the final Alignment Score**.

---

## ❌ Step 3: Dimension 2 · Qualitative Artifact Judgment

**Core Definition**: Evaluate whether the image contains structural collapse, violations of physical laws, or visual distortions unique to AI generation. **At this stage, completely ignore whether the object category is correct or not. Focus solely on "the physical plausibility of this image as a realistic photograph or coherent graphic".**

Annotators please select a score directly based on the following **qualitative matrix**:

* **5 - Flawless**:
  The image is perfect, completely conforming to the physical and photographic structures of the real world. There is no physical collapse, ghosting, melting, or extra limbs.
* **4 - Minor Artifacts**:
  There are very subtle AI traces that are extremely difficult to detect without close inspection, and they absolutely do not compromise the primary anatomical or mechanical structure of the subject.
* **3 - Moderate Distortion**:
  Obvious structural errors or physical implausibilities are present, but the basic topological framework of the subject remains intact (e.g., a fish fin is unnaturally melted at the edge, or complex mechanical patterns become locally chaotic and blurry).
* **2 - Severe Artifacts**:
  Severe physical collapse or heavily misplaced organs/components occur, though the subject can still barely be pieced together as a recognizable silhouette (e.g., an animal's eyes and ears melt into one piece).
* **1 - Catastrophic Collapse (Obvious AI Fail)**:
  The overall image suffers from extreme artifacts, melting, or severe structural collapse. **It is immediately recognizable as a useless, failed AI generation** (e.g., multi-limbed animals, wings melting into mid-air, or the subject being completely fused and locked into the background).
* **0 - Total Chaos**:
  The frame completely degenerates into a colorful noise, meaningless lines of flesh/structures, or unrecognizable pixels with no identifiable shape definitions.

---

## 📊 Step 4: Total Score Calculation & Multiplier Effect

The system will automatically calculate the product of the two dimensions using the following formula:

$$\text{Total Score} = \text{Alignment Score} \times \text{Artifact Score}$$

> **Total Score Range**: 0.0 to 25.0 points.
> Annotators must keep these two dimensions **completely decoupled (judged independently)**. The system will automatically penalize low-quality images through the multiplier effect.

### 💡 Representative Case Studies:

* **Case A (Attributes Present but Physical Collapse)**: A slot machine is generated, and all 14 attributes perfectly match the checklist (Alignment = 5.0 after calculation). However, the control panel melts into a blob, and 3 extra unaligned levers sprout from the side (Artifact = 1).
  * **Final Total Score**: $5.0 \times 1 = 5.0$ points (The system automatically penalizes it as a low-quality failed image due to the multiplier effect).
* **Case B (Flawless Render but Wrong Object)**: The image looks pristine like a high-definition National Geographic photo with zero artifacts (Artifact = 5). However, the model generates a "goldfish" instead of the required "great white shark", resulting in a complete attribute mismatch (Alignment = 0).
  * **Final Total Score**: $0 \times 5 = 0.0$ points (Determined as a failed generation).
* **Case C (High Alignment & Zero Artifacts)**: After eliminating the N/A items, the attributes perfectly match (Alignment = 5.0), and the image contains absolutely no physical artifacts (Artifact = 5).
  * **Final Total Score**: $5.0 \times 5 = 25.0$ points (The Gold Standard).

---

### 📌 Annotator Tips

1. **Strictness on N/A**: Select N/A **only if** a component is completely outside the camera view (e.g., a straight front close-up shot cannot capture side ventilation grilles). If a component should be in view but the AI simply forgot to draw it, you must ruthlessly select **🔴 Missing**.
2. **Multi-Subject Scenarios**: If multiple identical objects appear in the image, always focus on the **one in the center of the frame that is the largest and clearest** as your core benchmark for verification and evaluation.