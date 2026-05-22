from huggingface_hub import snapshot_download

# Cambiar ID por el del modelo que busques en Hugging Face (Asegurarse que sea SD 1.5) o de instrucciones.txt
repo_id = "songkey/meinamix_v12Final" 
carpeta_destino = "meinamix_hf" #Si cambias este nombre, recorda tambien cambiar el model_path en generador.py

print(f"Descargando el modelo completo {repo_id}...")
snapshot_download(
    repo_id=repo_id,
    local_dir=carpeta_destino,
    local_dir_use_symlinks=False
)
print(f"¡Listo! Modelo guardado en {carpeta_destino}")