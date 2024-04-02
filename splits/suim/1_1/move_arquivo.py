import os
import shutil

def mover_imagens_e_mascaras(origem_imagens, origem_mascaras, destino_imagens, destino_mascaras):
    # Verifica se os destinos existem, caso contrário, cria as pastas de destino
    total_movidas = 0
    limite = 150
    if not os.path.exists(destino_imagens):
        os.makedirs(destino_imagens)
    if not os.path.exists(destino_mascaras):
        os.makedirs(destino_mascaras)

    # Move as imagens e suas respectivas máscaras
    for imagem in os.listdir(origem_imagens):
        if total_movidas >= limite:
            break

        if imagem.endswith('.jpg') or imagem.endswith('.bmp') or imagem.endswith('.jpeg'):
            nome_base = os.path.splitext(imagem)[0]  # Remove a extensão do nome do arquivo
            mascara = nome_base + '.bmp'  # Supondo que as máscaras tenham extensão .png

            if mascara in os.listdir(origem_mascaras):
                shutil.move(os.path.join(origem_imagens, imagem), destino_imagens)
                shutil.move(os.path.join(origem_mascaras, mascara), destino_mascaras)
                total_movidas += 1

# Exemplo de uso
origem_imagens = '/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_Labeled/images'
origem_mascaras = '/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_Labeled/masks'
destino_imagens = '/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_VAL/images'
destino_mascaras = '/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_VAL/masks'

mover_imagens_e_mascaras(origem_imagens, origem_mascaras, destino_imagens, destino_mascaras)


def contar_arquivos(caminho):
    return len(os.listdir(caminho))

# Verificar a quantidade de arquivos em cada pasta após mover0
quantidade_imagens_treino = contar_arquivos(os.path.join('/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_VAL/images/'))

#verificar número de máscaras
quantidade_mascara = contar_arquivos(os.path.join('/home/pdi/EduardoLawson/Unimeu/UniMatch/data/SUIM_VAL/masks/'))

#quantidade = contar_arquivos(os.path.join('/content/drive/MyDrive/archive/treino/images'))
print("Quantidade de imagens na pasta de treino:", quantidade_imagens_treino)
print("Quantidade de máscaras no SUIM:", quantidade_mascara)
#print(quantidade)