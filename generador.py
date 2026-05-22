import os
import torch
import torch_directml
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.schedulers import EulerAncestralDiscreteScheduler
import gc

loras_activos = []
pesos_activos = []
device = torch_directml.device()
model_path = "./meinamix_hf"  # Cambiar nombre de carpeta donde tengas tu modelo

# Cambiar solo en caso de que tu checkpoint (modelo base) use un VAE especifico
print("-> Cargando VAE...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", 
    torch_dtype=torch.float16
)

print("-> Cargando imagen...")
pipeline = StableDiffusionPipeline.from_pretrained(
    model_path,
    vae=vae, 
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True,
    torch_dtype=torch.float16
)
pipeline = pipeline.to(device)

# Para tener LoRAs: debes colocar nombre real de los archivos en todos los lugares donde dice "desactivado". Si el archivo no existe simplemente no se ejecutara.
# (Solo se necesita el .safetensor, puedes descargar cualquier lora en Civitai y colocarlos en carpeta "loras")
# Trata que la suma de los pesos (weights) activados de los loras juntos no pasen el 1.0, para que el modelo base tenga mas fuerza y salga con el estilo deseado. (Sí solamente usas uno, su peso puede ser de 1.0 sin problemas)
print("Comprobando y cargando LoRAs...")
ruta_lora_1 = "./loras/desactivado.safetensors" 
if os.path.exists(ruta_lora_1):
    pipeline.load_lora_weights("./loras", weight_name="desactivado.safetensors", adapter_name="lora_1", local_files_only=True)
    loras_activos.append("lora_1")
    pesos_activos.append(0.5) #modificar este valor a conveniencia

ruta_lora_2 = "./loras/desactivado.safetensors"  
if os.path.exists(ruta_lora_2):
    pipeline.load_lora_weights("./loras", weight_name="desactivado.safetensors", adapter_name="lora_2", local_files_only=True)
    loras_activos.append("lora_2")
    pesos_activos.append(0.5) #modificar este valor a conveniencia

ruta_lora_3 = "./loras/desactivado.safetensors"  
if os.path.exists(ruta_lora_3):
    pipeline.load_lora_weights("./loras", weight_name="desactivado.safetensors", adapter_name="lora_3", local_files_only=True)
    loras_activos.append("lora_3")
    pesos_activos.append(0.5) #modificar este valor a conveniencia

if loras_activos:
   
    pipeline.set_adapters(loras_activos, adapter_weights=pesos_activos)

pipeline.enable_attention_slicing(slice_size="max")

if hasattr(pipeline, "enable_vae_slicing"):

    pipeline.enable_vae_slicing()


pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config
)
 # Esto es lo mas IMPORTANTE, acá puedes modificar tanto el "prompt" como el "negative_prompt"
 # Asegurarse de que sea de 75 tokens, si tiene mas, el generador ignorara el prompt sobrante, notificandote lo que no leyó
prompt = (
    "cinematic close-up portrait of a cute dog, border collie, under vibrant red umbrella, dark rainy street background, drmatic volumetric lighting, highly detailed, sharp focus, mood lighting, masterpiece"
)
negative_prompt = (
    " human, blur, monochrome, lowres, bad anatomy, worst quality, 3d, render, smooth shading, noise, nipples deformation, asymmetrical eyes"
)
print("-> Generando imagen desde texto...")
gc.collect()

# Probar distintas seed a gusto hasta encontrar la que se ajuste a tus preferencias con tus loras y prompt
# Recomendado elegir seed fija para más consistencia en varias generaciones
seed = 777  
generator = torch.manual_seed(seed) 

resultado = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,                      
    num_inference_steps=25,  # No bajar a menos de 20
    width=512,      # Usar siempre resolución 512x512 o 768x512 o 512x768
    height=512,     
    guidance_scale=7.0,    # Si no hace lo que le pedis en el prompt subir a 7.5 
    generator=generator
).images[0]

output_path = f"resultado_{seed}.png"
resultado.save(output_path)
print(f"¡Imagen guardada con éxito como '{output_path}'!")

del pipeline
if 'vae' in locals():
    del vae
gc.collect()