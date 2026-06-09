import os
import json
import gradio as gr

# ==========================================
# 1. 路径配置与核心数据加载
# ==========================================
BASE_DIR = "."  
IMAGE_DIR = os.path.join(BASE_DIR, "online_test_images")
TAXONOMY_DIR = os.path.join(BASE_DIR, "taxonomy_info")
CLASS_ID_FILE = os.path.join(IMAGE_DIR, "class_ids.txt")

ZH_GUIDE_PATH = os.path.join(BASE_DIR, "标注指南.md")
EN_GUIDE_PATH = os.path.join(BASE_DIR, "Annotation Guideline.md")

image_to_class = {}
class_to_taxonomy = {}
raw_img_list = [] 

def init_data():
    global image_to_class, class_to_taxonomy, raw_img_list
    if os.path.exists(CLASS_ID_FILE):
        with open(CLASS_ID_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    img_id, cls_id = line.strip().split()
                    image_to_class[f"{img_id}.png"] = int(cls_id)
                    image_to_class[f"{img_id}.jpg"] = int(cls_id)
                    image_to_class[f"{img_id}.jpeg"] = int(cls_id)
                    
    if os.path.exists(TAXONOMY_DIR):
        for file in os.listdir(TAXONOMY_DIR):
            if file.endswith(".json") and "structured" in file:
                try:
                    with open(os.path.join(TAXONOMY_DIR, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            class_to_taxonomy[item["class_id"]] = item
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    
    if os.path.exists(IMAGE_DIR):
        # 按照 class_ids.txt 的顺序加载图片
        raw_img_list = []
        if os.path.exists(CLASS_ID_FILE):
            with open(CLASS_ID_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        img_id, _ = line.strip().split()
                        # 查找实际存在的图片文件
                        for ext in ['.png', '.jpg', '.jpeg']:
                            img_name = f"{img_id}{ext}"
                            if os.path.exists(os.path.join(IMAGE_DIR, img_name)):
                                raw_img_list.append(img_name)
                                break
        else:
            # 如果没有 class_ids.txt，则按文件名排序
            raw_img_list = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

init_data()

# ==========================================
# 动态 MD 指南加载器
# ==========================================
def load_markdown_guide(language):
    target_path = ZH_GUIDE_PATH if language == "简体中文 (Chinese)" else EN_GUIDE_PATH
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"⚠️ Guide file not found at: `{target_path}`"

# ==========================================
# 单一用户独占主 JSON 数据库管理
# ==========================================
def get_user_file_path(annotator_id):
    out_dir = os.path.join(BASE_DIR, "output_results")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"User_{int(annotator_id)}_final_annotations.json")

def load_user_database(annotator_id):
    if not annotator_id:
        return {}
    file_path = get_user_file_path(annotator_id)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_to_user_database(annotator_id, img_name, single_record):
    file_path = get_user_file_path(annotator_id)
    db = load_user_database(annotator_id)
    db[img_name] = single_record
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

def build_dropdown_choices(user_db):
    choices = []
    for img in raw_img_list:
        if img in user_db:
            choices.append((f"√ {img}", img)) 
        else:
            choices.append((img, img))
    return choices

# ==========================================
# 2. 核心计算与业务逻辑
# ==========================================
def calculate_scores(artifact_score, veto, *checkmarks):
    if veto == "Yes (Total Mismatch - 0分)":
        return 0.0, 0.0
    checked_count = 0
    na_count = 0
    total_items = len(checkmarks)
    for val in checkmarks:
        if val == "🟢 Checked":
            checked_count += 1
        elif val == "⚪ N/A":
            na_count += 1
    effective_total = total_items - na_count
    alignment_score = 5 * (checked_count / effective_total) if effective_total > 0 else 0.0
    return round(alignment_score, 2), round(alignment_score * float(artifact_score), 2)

def load_image_ui_state(img_name, annotator_id):
    if not img_name or not annotator_id:
        return ("Unknown", "None", 5, "No", 0.0, 0.0, *[gr.update(visible=False) for _ in range(30)])
    
    class_id = image_to_class.get(img_name, None)
    if class_id is None or class_id not in class_to_taxonomy:
        return (f"Unknown (ID: {class_id})", "None", 5, "No", 0.0, 0.0, *[gr.update(visible=False) for _ in range(30)])
        
    tax_data = class_to_taxonomy[class_id]
    class_name = tax_data.get("class_name", "Unknown")
    super_cat = tax_data.get("super_category", "Unknown")
    checkpoints = tax_data.get("diagnostic_checkpoints", {})
    
    flat_points = []
    for section, points in checkpoints.items():
        for p in points:
            flat_points.append((section, p))
            
    user_db = load_user_database(annotator_id)
    has_history = img_name in user_db
    hist_data = user_db.get(img_name, {}) if has_history else {}
    
    saved_artifact = hist_data.get("scores", {}).get("artifact_score", 5) if has_history else 5
    saved_veto = "Yes (Total Mismatch - 0分)" if (has_history and hist_data.get("veto_activated", False)) else "No"
    
    updates = []
    for i in range(30):
        if i < len(flat_points):
            section, p = flat_points[i]
            saved_val = "🔴 Missing"
            if has_history:
                saved_val = hist_data.get("fine_grained_details", {}).get(section, {}).get(p, "🔴 Missing")
            updates.append(gr.update(label=p, value=saved_val, visible=True))
        else:
            updates.append(gr.update(visible=False, value="⚪ N/A"))
            
    meta_str = f"**Class ID**: {class_id}  |  **Super Category**: {super_cat}"
    
    flat_vals = [u['value'] for u in updates if 'value' in u]
    if not flat_vals:
        flat_vals = ["🔴 Missing"] * len(flat_points)
        
    align_calc, total_calc = calculate_scores(saved_artifact, saved_veto, *flat_vals)
    
    return (
        class_name, meta_str, 
        saved_artifact, saved_veto, 
        align_calc, total_calc,
        *updates
    )

def save_annotation(img_name, class_name, align_s, artifact_s, total_s, veto, annotator_id, *checkmarks):
    if not img_name or not annotator_id:
        return gr.update(value="⚠️ Save Failed: Context Missing"), gr.update()
    
    class_id = image_to_class.get(img_name)
    tax_data = class_to_taxonomy.get(class_id, {})
    checkpoints = tax_data.get("diagnostic_checkpoints", {})
    
    flat_points = []
    for section, points in checkpoints.items():
        for p in points:
            flat_points.append((section, p))
            
    details = {}
    for i, val in enumerate(checkmarks):
        if i < len(flat_points):
            section, p = flat_points[i]
            if section not in details:
                details[section] = {}
            details[section][p] = val

    single_record = {
        "image_name": img_name, 
        "class_id": class_id, 
        "class_name": class_name,
        "veto_activated": True if veto == "Yes (Total Mismatch - 0分)" else False,
        "scores": {
            "alignment_score": float(align_s),   
            "artifact_score": float(artifact_s),   
            "total_score": float(total_s)
        },
        "fine_grained_details": details
    }
    
    save_to_user_database(annotator_id, img_name, single_record)
    
    updated_db = load_user_database(annotator_id)
    new_choices = build_dropdown_choices(updated_db)
    
    # 体验增强：保存后自动探寻下一张未标注的图，无缝推进流水线
    next_img = img_name
    for img in raw_img_list:
        if img not in updated_db:
            next_img = img
            break
    
    return gr.update(value=f"💾 Saved {img_name} successfully!"), gr.update(choices=new_choices, value=next_img)

# ==========================================
# 用户隔离会话控制处理函数
# ==========================================
def start_session(total_users, current_user):
    # 注意这里增加到返回 7 个基础控制组件 + 6 个文字组件 + 30 个插槽
    if not current_user:
        return [gr.update() for _ in range(7)] + ["⚠️ Please specify your Annotator ID!", "None", 5, "No", 0.0, 0.0] + [gr.update() for _ in range(30)]
    
    if int(current_user) > int(total_users) or int(current_user) <= 0:
        return [gr.update() for _ in range(7)] + [f"⚠️ Invalid ID. Must be 1 ~ {total_users}", "None", 5, "No", 0.0, 0.0] + [gr.update() for _ in range(30)]
        
    user_db = load_user_database(current_user)
    completed_imgs = list(user_db.keys())
    
    target_img = raw_img_list[0] if raw_img_list else None
    for img in raw_img_list:
        if img not in completed_imgs:
            target_img = img
            break
    if not target_img and raw_img_list:
        target_img = raw_img_list[-1]
            
    progress_status = f"👋 **Welcome Back User {int(current_user)}!** Audited {len(completed_imgs)}/{len(raw_img_list)} images. Resuming from YOUR exclusive checkpoint."
    dropdown_choices = build_dropdown_choices(user_db)
    
    cls_name, meta, art_v, veto_v, al_v, tot_v, *slots = load_image_ui_state(target_img, current_user)
    
    # 【彻底修复】：不仅更新菜单和文字，将最底层的隐式图名容器和画面播放器一并强制清洗！
    return (
        gr.update(interactive=True), 
        gr.update(visible=True), 
        gr.Tabs(selected="workspace"), 
        gr.update(choices=dropdown_choices, value=target_img), 
        gr.update(value=progress_status),
        gr.update(value=target_img),                                                  # 强刷隐式图名
        gr.update(value=os.path.join(IMAGE_DIR, target_img) if target_img else None), # 强刷画面
        cls_name, meta, art_v, veto_v, al_v, tot_v, *slots
    )

def request_exit():
    return gr.update(visible=True)

def execute_exit(generate_report, annotator_id):
    msg = f"🚪 Session safely closed for User {annotator_id}."
    if generate_report == "Yes, generate summary report":
        user_db = load_user_database(annotator_id)
        out_dir = os.path.join(BASE_DIR, "output_results")
        report_path = os.path.join(out_dir, f"User_{int(annotator_id)}_audit_summary.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== ANNOTATOR {annotator_id} CENTRAL REPORT ===\n")
            f.write(f"Total Dataset Count: {len(raw_img_list)}\n")
            f.write(f"Progress Reached: {len(user_db)} / {len(raw_img_list)}\n")
            f.write(f"Status: Interrupted Session Backup Done.\n")
        msg += f"\n📊 Individual summary report compiled at: {report_path}"
    
    return gr.Tabs(selected="guideline"), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=False), msg

# ==========================================
# 3. UI 布局与组件配置
# ==========================================
custom_css = """
/* 全局字体统一设置 */
body, .gradio-container, .main, .wrap, .container {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 标题字体 */
h1, h2, h3, h4, h5, h6, .prose h1, .prose h2, .prose h3 {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-weight: bold !important;
}

/* Markdown 内容字体 */
.prose, .markdown-content, .md-preview {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* 标签和说明文字 */
label, .form-label, .block-label {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 按钮字体 */
button, .lg.svelte-1vft05d, .primary, .secondary, .stop {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-weight: bold !important;
}

/* 下拉菜单和输入框字体 */
select, input, textarea, .dropdown, .textbox, .number {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 单选框和复选框标签 */
.radio label, .checkbox label, .form-radio label {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 代码块保持等宽字体 */
code, pre, .code-block {
    font-family: "Consolas", "Courier New", monospace !important;
}

/* 表格字体 */
table, .dataframe, td, th {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 状态栏和信息提示 */
.status, .info, .warning, .error {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 只读数字输入框样式 */
input[readonly], .number input[readonly] {
    background-color: #f0f0f0 !important;
    cursor: not-allowed !important;
}

/* 图片缩放容器样式 */
.image-viewer-wrapper {
    overflow: hidden !important;
    position: relative !important;
    cursor: default !important;
}

.image-viewer-wrapper.zoomed {
    cursor: grab !important;
}

.image-viewer-wrapper:active {
    cursor: grabbing !important;
}

.image-viewer-wrapper img {
    transition: transform 0.15s ease-out !important;
    transform-origin: 0 0 !important;
    max-width: 100% !important;
}

.image-viewer-wrapper img.dragging {
    transition: none !important;
}
"""

# ==========================================
# 图片缩放功能
# ==========================================
ZOOM_STEP = 25  # 每次缩放 25%
ZOOM_MIN = 25
ZOOM_MAX = 400

def zoom_image(current_zoom, direction):
    """处理图片缩放"""
    new_zoom = current_zoom + direction * ZOOM_STEP
    new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, new_zoom))  # 限制在 25% - 400%
    return new_zoom

def reset_zoom():
    """重置缩放"""
    return 100

custom_theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="neutral").set(
    button_primary_background_fill="*primary_200", 
    button_primary_background_fill_hover="*primary_300",
    button_primary_text_color="*neutral_800"
)

custom_js = """
(function() {
    let scale = 1;
    let panX = 0, panY = 0;
    let isPanning = false;
    let startX, startY;
    let lastMouseX = 0, lastMouseY = 0;
    let initialized = false;
    
    function getElements() {
        const group = document.getElementById('image-viewer-group');
        if (!group) return null;
        
        const img = group.querySelector('img');
        if (!img) return null;
        
        const wrapper = group.querySelector('.wrap') || group;
        
        return { img, wrapper, group };
    }
    
    function applyTransform(img) {
        if (!img) return;
        img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        img.style.transformOrigin = '0 0';
        img.style.transition = 'transform 0.2s ease-out';
    }
    
    window.zoomIn = function() {
        const elements = getElements();
        if (!elements) return 100;
        
        const { img, wrapper } = elements;
        const oldScale = scale;
        scale = Math.min(scale + 0.25, 4);
        
        const rect = wrapper.getBoundingClientRect();
        const centerX = lastMouseX - rect.left;
        const centerY = lastMouseY - rect.top;
        
        panX = centerX - (centerX - panX) * (scale / oldScale);
        panY = centerY - (centerY - panY) * (scale / oldScale);
        
        if (scale > 1) {
            wrapper.style.cursor = 'grab';
        }
        
        applyTransform(img);
        return Math.round(scale * 100);
    };
    
    window.zoomOut = function() {
        const elements = getElements();
        if (!elements) return 100;
        
        const { img, wrapper } = elements;
        const oldScale = scale;
        scale = Math.max(scale - 0.25, 0.25);
        
        if (scale <= 1) {
            scale = 1;
            panX = 0;
            panY = 0;
            wrapper.style.cursor = 'default';
        } else {
            const rect = wrapper.getBoundingClientRect();
            const centerX = lastMouseX - rect.left;
            const centerY = lastMouseY - rect.top;
            
            panX = centerX - (centerX - panX) * (scale / oldScale);
            panY = centerY - (centerY - panY) * (scale / oldScale);
        }
        
        applyTransform(img);
        return Math.round(scale * 100);
    };
    
    window.resetZoom = function() {
        const elements = getElements();
        if (!elements) return 100;
        
        const { img, wrapper } = elements;
        scale = 1;
        panX = 0;
        panY = 0;
        wrapper.style.cursor = 'default';
        img.style.transform = '';
        
        return 100;
    };
    
    function setupPan() {
        if (initialized) return;
        
        const elements = getElements();
        if (!elements) return;
        
        const { img, wrapper } = elements;
        initialized = true;
        
        wrapper.style.overflow = 'hidden';
        wrapper.style.position = 'relative';
        wrapper.style.cursor = 'default';
        
        wrapper.addEventListener('mousemove', function(e) {
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });
        
        wrapper.addEventListener('mousedown', function(e) {
            if (scale <= 1) return;
            isPanning = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            wrapper.style.cursor = 'grabbing';
            img.style.transition = 'none';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', function(e) {
            if (!isPanning) return;
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        });
        
        document.addEventListener('mouseup', function() {
            if (isPanning) {
                isPanning = false;
                wrapper.style.cursor = scale > 1 ? 'grab' : 'default';
                img.style.transition = 'transform 0.2s ease-out';
            }
        });
    }
    
    const observer = new MutationObserver(setupPan);
    observer.observe(document.body, { childList: true, subtree: true });
    
    setTimeout(setupPan, 500);
    setTimeout(setupPan, 1500);
    setTimeout(setupPan, 3000);
    setTimeout(setupPan, 5000);
})();
"""

with gr.Blocks(title="Fine-Grained Visual Audit System", css=custom_css, js=custom_js) as demo:
    gr.Markdown("# 📋 ImageNet-1k 细粒度视觉审计平台")
    
    raw_img_holder = gr.Textbox(visible=False, value="")
    
    with gr.Tabs() as main_tabs:
        
        # --- TAB 1: 门口页 ---
        with gr.Tab("📜 Annotation Guideline", id="guideline"):
            with gr.Group():
                gr.Markdown("### 👥 Step 1: Initialize User Context & Audit Team")
                with gr.Row():
                    total_user_num = gr.Number(value=3, label="Total Number of Annotators in Team", precision=0)
                    current_user_id = gr.Number(value=1, label="Your Assigned Annotator ID (e.g., 1, 2, 3)", precision=0)
            
            with gr.Group():
                gr.Markdown("### 📜 Step 2: Review Full Audit Documentation")
                lang_selector = gr.Dropdown(choices=["简体中文 (Chinese)", "English (UK/US)"], value="简体中文 (Chinese)", label="选择指南语言 / Select Documentation Language")
                md_viewer = gr.Markdown(value=load_markdown_guide("简体中文 (Chinese)"))
                lang_selector.change(load_markdown_guide, inputs=[lang_selector], outputs=[md_viewer])

            start_btn = gr.Button("🚀 Start/Resume My Audit Session", variant="primary", size="lg")
            exit_msg_box = gr.Markdown("")

        # --- TAB 2: 主标注工作区 ---
        with gr.Tab("💻 Active Workspace", id="workspace", interactive=False) as workspace_tab:
            with gr.Column(visible=False) as workspace_container:
                
                with gr.Row():
                    user_status_bar = gr.Markdown("🟢 **Current Status: Session Active**")
                    exit_req_btn = gr.Button("🚪 Exit & Save Mid-Progress", variant="stop", size="sm")
                
                with gr.Group(visible=False) as exit_confirm_container:
                    gr.Markdown("### 🛠️ Interrupted Mid-Progress Session Save")
                    report_radio = gr.Radio(choices=["Yes, generate summary report", "No, just exit directly"], value="Yes, generate summary report", label="Do you want to export your personal progress report?", interactive=True)
                    with gr.Row():
                        confirm_exit_btn = gr.Button("Confirm Exit & Close", variant="primary")
                        cancel_exit_btn = gr.Button("Cancel", variant="secondary")

                gr.Markdown("---")
                
                with gr.Row():
                    # --- 左侧：图片展示区 ---
                    with gr.Column(scale=4):
                        image_selector = gr.Dropdown(choices=[], label="🖼️ Select Target Image")
                        
                        # 图片查看器 + 缩放控制
                        with gr.Group(elem_id="image-viewer-group"):
                            image_viewer = gr.Image(label="AI Generated Image", type="filepath", elem_id="target-image")
                            with gr.Row():
                                zoom_in_btn = gr.Button("🔍 Zoom In", variant="secondary", size="sm")
                                zoom_out_btn = gr.Button("🔍 Zoom Out", variant="secondary", size="sm")
                                zoom_reset_btn = gr.Button("🔄 Reset", variant="secondary", size="sm")
                            zoom_level = gr.Number(value=100, label="Zoom Level (%)", precision=0, interactive=False)
                        
                        def on_dropdown_select(selected_val):
                            if not selected_val:
                                return "", None, 100
                            return selected_val, os.path.join(IMAGE_DIR, selected_val), 100
                        
                        # 图片信息区
                        class_name_disp = gr.Textbox(label="🏷️ Target Class Name", interactive=False, value="Loading...")
                        meta_disp = gr.Markdown("**Class ID**: --  |  **Super Category**: --")
                        
                        # 总分和保存按钮
                        total_disp = gr.Number(label="🏆 Total Score (Alignment × Artifact)", value=0.0, precision=2, interactive=False)
                        save_btn = gr.Button("💾 Save & Submit This Image", variant="primary")
                        msg_box = gr.Markdown("")

                    # --- 右侧：Dimension 1 (Attribute) ---
                    with gr.Column(scale=6):
                        gr.Markdown("### 📏 维度一 · 语义对齐客观核对 (Alignment Audit)")
                        gr.Markdown("### 📏 Dimension 1: Fine-Grained Attribute Alignment")
                        
                        with gr.Accordion("💡 Dimension 1: Attribute Tri-State Guide", open=True):
                            gr.Markdown("""
                            * **🟢 Checked**: Perfectly present, matches description.
                            * **🔴 Missing**: Missing or heavily violated in its corresponding region.
                            * **⚪ N/A**: Out of frame due to extreme close-up/cropping.
                            * 忽略具体的长度、重量。如果一条描述中，涉及到了具体的长度、重量，标注员请直接忽略，只关注其他属性是否存在；如果只有长度、重量，标注员请直接标注为 N/A。
                            * 如果发现有内容重复的判断点，请在第二次以及之后遇到的时候选择N/A。
                            """)
                        
                        gr.Markdown("### � Fine-Grained Attribute Checklists (Clean View)")
                        with gr.Group():
                            radio_slots = []
                            for _ in range(30):
                                r = gr.Radio(choices=["🟢 Checked", "🔴 Missing", "⚪ N/A"], value="🔴 Missing", visible=False, label="")
                                radio_slots.append(r)
                        
                        gr.Markdown("---")
                        
                        # Veto Trigger（属于 Dimension 1）
                        veto_radio = gr.Radio(choices=["No", "Yes (Total Mismatch - 0分)"], value="No", label="🚨 Veto Trigger", info="Force Alignment to 0 if wrong object.")
                        
                        # Alignment Score（Dimension 1 的结果）
                        alignment_disp = gr.Number(label="📊 Alignment Score (Auto-calculated from Attributes)", value=0.0, precision=2)
                    
                    # --- 中间：Dimension 2 (Artifact) ---
                    with gr.Column(scale=3):
                        gr.Markdown("### 🎨 维度二 · 物理真实性定性判定 (Authentication Score)")
                        gr.Markdown("### 🎨 Dimension 2: Authentication Quality Assessment")
                        
                        with gr.Accordion("💡 Dimension 2: Authentication Slider Quick Guide", open=True):
                            gr.Markdown("""
                            * **5 (Flawless)**: Photographic quality, zero physical distortion.
                            * **4 (Minor)**: Tiny AI traces, framework entirely intact.
                            * **3 (Moderate)**: Clear structural errors, local chaotic patterns.
                            * **2 (Severe)**: Misplaced components, heavily deformed details.
                            * **1 (Collapse)**: Massive structural melting. Useless AI failure.
                            * **0 (Chaos)**: Pure pixel/structural noise.
                            * 如果标注员不太确定伪影、物理错误等情况的严重程度，可以参考之前的判断和附近图片的结果。
                            * 针对模糊，如果标注员自行判断模糊是否由AI生成，是否影响画面结构。
                            """)
                        
                        gr.Markdown("### 🎯 Authentication Score")
                        artifact_disp = gr.Slider(minimum=0, maximum=5, step=1, value=5, label="Authentication Score (Manual)")

    # ==========================================
    # 4. 高级事件链流转
    # ==========================================
    start_btn.click(
        start_session, 
        inputs=[total_user_num, current_user_id], 
        outputs=[
            workspace_tab, workspace_container, main_tabs, image_selector, user_status_bar, 
            raw_img_holder, image_viewer,  # <==== 补上了这里！
            class_name_disp, meta_disp, artifact_disp, veto_radio, alignment_disp, total_disp, 
            *radio_slots
        ]
    )
    
    image_selector.change(
        on_dropdown_select, 
        inputs=[image_selector], 
        outputs=[raw_img_holder, image_viewer, zoom_level]
    ).then(
        load_image_ui_state, 
        inputs=[raw_img_holder, current_user_id], 
        outputs=[class_name_disp, meta_disp, artifact_disp, veto_radio, alignment_disp, total_disp, *radio_slots]
    )
    
    # 缩放按钮事件
    zoom_in_btn.click(
        None,
        inputs=[],
        outputs=[zoom_level],
        js="() => { return window.zoomIn ? window.zoomIn() : 100; }"
    )
    
    zoom_out_btn.click(
        None,
        inputs=[],
        outputs=[zoom_level],
        js="() => { return window.zoomOut ? window.zoomOut() : 100; }"
    )
    
    zoom_reset_btn.click(
        None,
        inputs=[],
        outputs=[zoom_level],
        js="() => { return window.resetZoom ? window.resetZoom() : 100; }"
    )
    
    exit_req_btn.click(request_exit, inputs=[], outputs=[exit_confirm_container])
    cancel_exit_btn.click(lambda: gr.update(visible=False), inputs=[], outputs=[exit_confirm_container])
    confirm_exit_btn.click(execute_exit, inputs=[report_radio, current_user_id], outputs=[main_tabs, workspace_container, exit_confirm_container, workspace_tab, exit_msg_box])

    score_inputs = [artifact_disp, veto_radio] + radio_slots
    for widget in score_inputs:
        widget.change(calculate_scores, inputs=score_inputs, outputs=[alignment_disp, total_disp])
        
    save_btn.click(
        save_annotation,
        inputs=[raw_img_holder, class_name_disp, alignment_disp, artifact_disp, total_disp, veto_radio, current_user_id] + radio_slots,
        outputs=[msg_box, image_selector]
    )

if __name__ == "__main__":
    # demo.launch(server_name="127.0.0.1", server_port=7860, theme=custom_theme)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=False,
        # 加上下面这一行，为 3 位标注员和自己设置专属账号密码
        auth=[
            ("admin", "AdminPassword3502"),
            ("Annotater1", "I am Annotater1*"),
            ("Annotater2", "I am Annotater2*"),
            ("Annotater3", "I am Annotater3*")
        ]
    )