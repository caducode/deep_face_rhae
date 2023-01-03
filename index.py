from deepface import DeepFace
#import pickle
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str)
args = parser.parse_args()



if args.img == None:
    print("você deve especificar uma imagem de entrada")
    exit()

resized_folder = "/home/cadu/Documents/projeto_arezzo/deep_face/img_faces/resized"
imgs_path = "/home/cadu/Documents/projeto_arezzo/deep_face/img_faces/banco_faces/"

def verifica_face(path):
    img_detect = DeepFace.detectFace(path)
    plt.imshow(img_detect)
    plt.show()

def redimensiona_img(img_path,folder_resized,img_name):
    width = 224
    height = 224
    dim = (width, height)
    sup_name=img_name.split(".")
    new_name=sup_name[0]+"_224."+sup_name[1]
    imgteste = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(imgteste,dim,interpolation = cv2.INTER_AREA)
    cv2.imwrite((str(folder_resized+new_name)),resized)

def consulta_id(img_name):
    result = 0
    with open('clientes.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(row['imagem']==img_name):
                result =  row['ID']
        return result


#folder_teste = "/home/cadu/Documents/projeto_arezzo/deep_face/img_faces/banco_faces/"
#redimensiona_img("/home/cadu/Documents/projeto_arezzo/deep_face/img_faces/banco_faces/ben_affleck_teste.jpg",folder_teste,"ben_affleck_teste.jpg")
#exit()

##### REDIMENSIONA AS IMAGENS PARA O TAMANHO ESPERADO PELO DEEPFACE ########
#for i, path in enumerate(tqdm(os.listdir(imgs_path))):
#    redimensiona_img(os.path.join(imgs_path, path),resized_folder,path)

#img1dect = DeepFace.detectFace(img_teste)
#plt.imshow(img1dect)
#plt.show()


backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

img_test = args.img
actl_dir = os.getcwd()
image_path = os.path.join(actl_dir,img_test)


df = DeepFace.find(img_path = image_path, 
        db_path = resized_folder, 
        detector_backend = backends[0]
)

arr = df['identity'][0].split("/")
arr2 = arr[::-1]
nome_result  = arr2[0]

print(consulta_id(nome_result))

# for i, path in enumerate(tqdm(os.listdir(resized_folder))):
#     try:
#         faces.append(DeepFace.detectFace(os.path.join(resized_folder, path)))
#     except:
#         print(os.path.join(resized_folder, path))


