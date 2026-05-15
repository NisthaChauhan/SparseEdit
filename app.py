"""
app.py
------
Nano-Banana-style prompt-driven image editor for SparseEdit.
Defensive against Gradio API drift across 4.x / 5.x / 6.x.

Run:
    python app.py
"""

from __future__ import annotations

import gradio as gr
from PIL import Image

from ui_backend import EditRequest, edit_image, get_pipeline


# --------------------------------------------------------------------- #
# Detect Gradio version once and pick the right chat message format     #
# --------------------------------------------------------------------- #
_GRADIO_VERSION = tuple(int(x) for x in gr.__version__.split(".")[:2]
                        if x.isdigit())
print(f"[app] Gradio version detected: {gr.__version__}")


def _make_chat_messages(history: list, user_msg: str, bot_msg: str) -> list:
    """Append a turn in OpenAI-style dict format (Gradio 5.0+ default)."""
    history = list(history) if history else []
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": bot_msg})
    return history



# --------------------------------------------------------------------- #
# Warm up the model at startup                                          #
# --------------------------------------------------------------------- #
print("[app] Warming up SparseEdit pipeline ...")
get_pipeline()
print("[app] Ready.")


# --------------------------------------------------------------------- #
# Core edit handler                                                     #
# --------------------------------------------------------------------- #
def run_edit(
    source_image: Image.Image,
    prompt: str,
    source_prompt: str,
    num_steps: int,
    guidance_scale: float,
    lambda_sparsity: float,
    tau: float,
    eta: float,
    seed: int,
    resolution: int,
    chat_history: list,
    progress=gr.Progress(track_tqdm=True),
):
    if source_image is None:
        raise gr.Error("Please upload a source image first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please type an edit prompt.")

    progress(0.0, desc="Encoding source image...")

    req = EditRequest(
        source_image=source_image,
        prompt=prompt.strip(),
        source_prompt=(source_prompt or "").strip(),
        num_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        lambda_sparsity=float(lambda_sparsity),
        tau=float(tau),
        eta=float(eta),
        seed=int(seed),
        height=int(resolution),
        width=int(resolution),
    )

    def _cb(step, total):
        progress(step / max(total, 1), desc=f"Denoising step {step}/{total}")

    result = edit_image(req, progress_cb=_cb)

    m = result.metrics
    metric_lines = [
        f"⏱  {result.elapsed_s:.1f}s   ({result.backend} mode)",
        f"PSNR_bg: {m.get('PSNR_bg', float('nan')):.2f} dB  (target > 40)",
        f"SSIM_bg: {m.get('SSIM_bg', float('nan')):.4f}     (target > 0.98)",
        f"Leakage: {m.get('Leakage', float('nan')):.4f}    (target < 0.01)",
        f"CLIP:    {m.get('CLIP_score', float('nan')):.4f}    (target > 0.28)",
        f"Sparsity:{m.get('Sparsity', float('nan')):.4f}    (target > 0.85)",
    ]
    metrics_md = "```\n" + "\n".join(metric_lines) + "\n```"

    new_history = _make_chat_messages(chat_history, prompt, metrics_md)
    return result.edited_image, new_history, metrics_md


def clear_all():
    return None, None, [], "*Metrics will appear here after the first edit.*"


# --------------------------------------------------------------------- #
# UI layout — minimal kwargs only                                       #
# --------------------------------------------------------------------- #
CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
#title { text-align: center; }
.metric-panel { font-family: ui-monospace, SFMono-Regular, monospace; }
"""

with gr.Blocks(title="SparseEdit") as demo:
    gr.Markdown(
        "# 🍌 SparseEdit — Prompt-Driven Image Editor\n"
        "Zero-leakage diffusion editing on Apple Silicon. "
        "Drop an image, type what you want changed, hit **Edit**.",
        elem_id="title",
    )

    with gr.Row():
        # ---------------- LEFT: input image + result ---------------- #
        with gr.Column(scale=1):
            source = gr.Image(label="Source image", type="pil")
            edited = gr.Image(label="Edited result", type="pil",
                              interactive=False)

        # ---------------- RIGHT: prompt + history + metrics ---------- #
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Edit prompt",
                placeholder='e.g. "make the cabinets oak wood"',
                lines=2,
            )
            with gr.Row():
                edit_btn = gr.Button("✨  Edit", variant="primary")
                clear_btn = gr.Button("Clear")

            # Bare-minimum Chatbot — only `label` is universally accepted
            chat = gr.Chatbot(label="Edit history")

            metrics_box = gr.Markdown(
                value="*Metrics will appear here after the first edit.*",
                elem_classes=["metric-panel"],
            )

            with gr.Accordion("Advanced controls", open=False):
                source_prompt = gr.Textbox(
                    label="Source caption (optional)",
                    placeholder="e.g. 'a photo of a kitchen'",
                    lines=1,
                )
                num_steps = gr.Slider(10, 100, value=30, step=1,
                                      label="DDIM steps")
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5,
                                     label="Guidance scale")
                lam = gr.Slider(0.0, 0.5, value=0.05, step=0.005,
                                label="λ — L1 sparsity weight")
                tau = gr.Slider(0.0, 0.2, value=0.02, step=0.001,
                                label="τ — soft-threshold")
                eta = gr.Slider(0.0, 1.0, value=0.1, step=0.01,
                                label="η — proximal step size")
                seed = gr.Number(value=0, precision=0, label="Seed")
                resolution = gr.Dropdown([512, 768, 1024], value=1024,
                                         label="Resolution")

            try:
                gr.Examples(
                    examples=[
                        ["input.png",
                         "a fluffy orange cat sitting on the floor"],
                        ["input_kitchen.png",
                         "replace the cabinets with dark oak wood"],
                    ],
                    inputs=[source, prompt],
                    label="Try an example",
                )
            except Exception as e:
                print(f"[app] Skipping examples block: {e}")

    # ---------------- wiring ---------------- #
    edit_btn.click(
        fn=run_edit,
        inputs=[source, prompt, source_prompt,
                num_steps, guidance, lam, tau, eta, seed, resolution, chat],
        outputs=[edited, chat, metrics_box],
    )

    prompt.submit(
        fn=run_edit,
        inputs=[source, prompt, source_prompt,
                num_steps, guidance, lam, tau, eta, seed, resolution, chat],
        outputs=[edited, chat, metrics_box],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[source, edited, chat, metrics_box],
    )


if __name__ == "__main__":
    # Try the new launch signature first; fall back if needed.
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            inbrowser=True,
            css=CSS,
        )
    except TypeError:
        # Older Gradio that doesn't accept `css` in launch()
        demo.queue(max_size=8).launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            inbrowser=True,
        )
