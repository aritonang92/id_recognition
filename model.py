import face_recognition as fr
import numpy as np
import sys

# loading images id
bP03_img = fr.load_image_file("images/id_img/id_bP03.jpg")
iW29_img = fr.load_image_file("images/id_img/id_iW29.jpg")
kE27_img = fr.load_image_file("images/id_img/id_kE27.jpg")
Pz02_img = fr.load_image_file("images/id_img/id_Pz02.jpg")
QW00_img = fr.load_image_file("images/id_img/id_QW00.jpg")
uD88_img = fr.load_image_file("images/id_img/id_uD88.jpg")
xK34_img = fr.load_image_file("images/id_img/id_xK34.jpg")
jH77_img = fr.load_image_file("images/id_img/id_jH77.jpg")


# encoding them
try:
    bP03_encoding = fr.face_encodings(bP03_img)[0]
    iW29_encoding = fr.face_encodings(iW29_img)[0]
    kE27_encoding = fr.face_encodings(kE27_img)[0]
    Pz02_encoding = fr.face_encodings(Pz02_img)[0]
    QW00_encoding = fr.face_encodings(QW00_img)[0]
    uD88_encoding = fr.face_encodings(uD88_img)[0]
    xK34_encoding = fr.face_encodings(xK34_img)[0]
    jH77_encoding = fr.face_encodings(jH77_img)[0]
except IndexError as e:
    print(e)
    sys.exit(1) # stops code execution in my case you could handle it differently

# Create known face encoding and known face names
known_face_encoding = [
    bP03_encoding,
    iW29_encoding,
    kE27_encoding,
    Pz02_encoding,
    QW00_encoding,
    uD88_encoding,
    xK34_encoding,
    jH77_encoding,
]

known_face_names = [
    "bp03",
    "iW29",
    "kE27",
    "Pz02",
    "QW00",
    "uD88",
    "xK34",
    "jH77"
]

# Create function for predictiing unknown image
def matcher(foto):
    face_pict = fr.load_image_file(foto)
    face_encoding = fr.face_encodings(face_pict)

    for encoding_face in face_encoding:
        cocok = fr.compare_faces(known_face_encoding, encoding_face, tolerance = 0.55)
        distance = fr.face_distance(known_face_encoding, encoding_face)

        if True in cocok:
            index_cocok = cocok.index(True)
            nama = known_face_names[index_cocok]
            euclidean_distance = np.around(distance[index_cocok],2)
            print(f"Name : {nama}\nDistance to {nama}: {euclidean_distance}")
            break
        else:
            print("Photo is not match to anyone")

matcher("images/tester_img/tester5.jpg")


