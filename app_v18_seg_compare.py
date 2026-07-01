"""Lightweight viewer for v18 street/satellite semantic comparison images."""

from __future__ import annotations

from pathlib import Path

import gradio as gr


DEFAULT_DIR = Path(r"C:\Users\31133\Desktop\v18街景卫星语义对比图")


def list_images(folder: str) -> list[str]:
    root = Path(folder)
    if not root.exists():
        return []
    return [p.name for p in sorted(root.glob("*.jpg"))]


def load_selected(folder: str, image_name: str | None):
    root = Path(folder)
    images = list_images(folder)
    if not images:
        return None, [], "未找到 JPG 对比图。"
    if not image_name or image_name not in images:
        image_name = images[0]
    return str(root / image_name), gr.update(choices=images, value=image_name), f"当前图片: {image_name}"


def refresh(folder: str):
    images = list_images(folder)
    first = images[0] if images else None
    image_path = str(Path(folder) / first) if first else None
    status = f"已加载 {len(images)} 张对比图。" if images else "未找到 JPG 对比图。"
    return gr.update(choices=images, value=first), image_path, status


with gr.Blocks(title="v18 Street/Satellite Semantic Viewer") as demo:
    gr.Markdown(
        """
        # v18 Street/Satellite Semantic Comparison
        # v18 街景 / 卫星语义对比查看器

        Columns in each image:
        Street RGB | SegFormer-B3 street semantic | DINOv3 refined street semantic | Satellite RGB | Satellite DINOv3 semantic

        每张图从左到右：
        街景原图 | SegFormer-B3 街景语义 | DINOv3 优化街景语义 | 卫星图 | 卫星 DINOv3 语义
        """
    )

    folder = gr.Textbox(label="Image folder / 图片文件夹", value=str(DEFAULT_DIR))
    with gr.Row():
        image_name = gr.Dropdown(label="Comparison image / 对比图", choices=list_images(str(DEFAULT_DIR)))
        refresh_btn = gr.Button("Refresh / 刷新")
    output = gr.Image(label="Comparison / 对比结果", type="filepath", height=760)
    status = gr.Textbox(label="Status / 状态", interactive=False)

    image_name.change(load_selected, inputs=[folder, image_name], outputs=[output, image_name, status])
    refresh_btn.click(refresh, inputs=[folder], outputs=[image_name, output, status])
    demo.load(refresh, inputs=[folder], outputs=[image_name, output, status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
