import os

# Diretórios das imagens e máscaras
imagens_dir = "/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_VAL/images"
mascaras_dir = "/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_VAL/masks/"

# Lista de arquivos nas pastas
imagens = os.listdir(imagens_dir)
mascaras = os.listdir(mascaras_dir)

# Filtrar apenas arquivos com extensão .png nas imagens e .bmp nas máscaras
imagens = [img for img in imagens if img.endswith(".jpg")]
mascaras = [mask for mask in mascaras if mask.endswith(".bmp")]

# Ordenar as listas
imagens.sort()
mascaras.sort()

# Verificar se o número de imagens é o mesmo que o número de máscaras
if len(imagens) != len(mascaras):
    print("Número de imagens e máscaras não correspondem.")
    exit()

# Criar e escrever no arquivo txt
with open("val.txt", "w") as f:
    for img, mask in zip(imagens, mascaras):
        f.write(f"{os.path.join(imagens_dir, img)} {os.path.join(mascaras_dir, os.path.splitext(mask)[0] + '.bmp')}\n")
