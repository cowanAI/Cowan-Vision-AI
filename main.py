import gradio as gr
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

def infer(prompt):
    negative_prompt = "text, watermark, blurry, nsfw"
    video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=40, height=480, width=720, num_frames=24).frames
    video_path = export_to_video(video_frames)
    print(video_path)
    return video_path




ui= gr.Interface(
    fn=infer,
    theme='gradio/soft',
    inputs=gr.Textbox(label="Prompt", placeholder="Northern lights in Iceland", elem_id="prompt-in"),
    outputs=gr.Video(label="Video Output", elem_id="video-output"),
    live=False,
    examples =[ ["Darth vader fighting jedis, Cinematic lighting."],
    ["Northern lights in Iceland landscape, Cinematic lighting,"],
               
],    
    css="""footer {visibility: hidden}

    
    """, 
    allow_flagging=('never'),
    analytics_enabled=False
)

ui.launch(server_name="0.0.0.0", server_port=80)