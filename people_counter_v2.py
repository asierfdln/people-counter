import numpy as np
import argparse
import cv2

import jetson.utils

# no sé de dónde hemos sacado esto, pero lo utiliza todo peter para sacar el recurso famoso este del gstreamer de la 
# jetson (hay que decir tb que me suena haberlo visto en algun foro de estos en los que el bueno de dusty-nv contestaba 
# cosas pero no estoy muy seguro...)

# Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
def gstreamer(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2):
    return (
        f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
        f'width=(int){capture_width}, height=(int){capture_height}, ' +
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
        f'nvvidconv flip-method={flip_method} ! ' +
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
        'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="pednet/snapshot_iter_70800.caffemodel", help="path to Caffe pre-trained model")
    parser.add_argument("-p", "--prototxt", default="pednet/deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
    parser.add_argument("-l", "--labels", default="pednet/class_labels.txt", help="path to class labels")
    parser.add_argument("-c", "--confidence", type=float, default=0.7, help="minimum probability to filter weak detections")
    parser.add_argument("-o", "--output", type=str, help="path to optional output video file")
    parser.add_argument("-cw", "--capwidth", type=int, default=1280, help="capture width (default 1280)")
    parser.add_argument("-ch", "--capheight", type=int, default=720, help="capture height (default 720)")
    parser.add_argument("-dw", "--dispwidth", type=int, default=1280, help="display width (default 1280)")
    parser.add_argument("-dh", "--dispheight", type=int, default=720, help="display height (default 720)")
    args = vars(parser.parse_args())

    CW = args["capwidth"]
    CH = args["capheight"]
    DW = args["dispwidth"]
    DH = args["dispheight"]

    print("[INFO] loading classes...")
    CLASSES = [] # lista (un array de toalavida) en la que van a estar todas las clases que puede detectar el modelo

    # esta mierda abre el archivo definido en la ruta del parámetro de labels del argumentParser 
    # (se cierra magicamente tb porque python mola así de mucho)
    with open(args["labels"]) as f:
        CLASSES = f.readlines() # pilla toaslaslineas del archivo y me las pones en una listica

    # magia en python que te pone en una lista (un array de toalavida) todas las lineas del fichero f de antes 
    # pero stripeadas de espacios al final y \n y toalapesca
    CLASSES = [line.strip() for line in CLASSES]

    print("[INFO] loading model...")
    # cargamos el modelo a pelo
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    print("[INFO] starting video stream...")
    # inisiamos la camara (TODO creo que el doorbell hacía cosas chachis de detectar en qué plataforma 
    # se estaba ejecutando el tinglao para ver si era la jetson o un pc a secas o qué, sería interesante poner eso tb...)
    capture = cv2.VideoCapture(gstreamer(capture_width=CW, capture_height=CH, display_width=DW, display_height=DH), cv2.CAP_GSTREAMER)

    # aquí después del "capture" el people_counter OG hace un time.sleep(2.0) que me da muy mala espina, como que sobra un 
    # poco... TODO probar si de verdad importa o q...

    while True:
        # capture.read() aquí devuelve dos cosas:
        #   ret -> valor booleano que indica si se ha pillao algo por cámara o k ase
        #   fram -> pues eso, el frame
        ret, frame = capture.read()

        if ret:  # si hay frame pillao pues lo muestras
            cv2.imshow("People Counter", frame)

            # RESTO DE LÓGICA DEL PROGRAMILLA
            # TODO conteo de FPS en la propia ventana...


            # El cambio este de cvtColores necesario por cómo maneja opencv bytes de imágenes REFERIDO A 
            # LA FUNC imread(), no veo nada de la func VideoCapture.read()... Porque el vídeo se ve 
            # "normal" y no invertido... ¿Hicimos pruebas viendo a ver si reconocía cosas sin hacer lo de rgb? TODO...

            # https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/
            # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor

            """
            Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR 
            (the bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit 
            Blue component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, 
            and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
            """

            # frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Haciendo lectura de este bloque de wtfismo...
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (CW, CH), 127.5)
            net.setInput(blob)
            detections = net.forward()

        # esperamos a una 'q' de teclado para salir del bisho este
        key = cv2.waitKey(1) & 0xFF # La función waitKey devuelve - 1 cuando no se realiza ninguna entrada. Tan pronto como ocurre el evento es decir. Se presiona un botón devuelve un entero de 32 bits.
                                    # El 0xFF en este escenario representa binario 11111111 a 8 bit binary, ya que solo necesitamos 8 bits para representar un carácter que AND Y waitKey(0) a 0xFF. Como resultado, se obtiene un número entero por debajo de 255.
                                    # ord(char) devuelve el valor ASCII del carácter que nuevamente sería máximo 255.
                                    # Por lo tanto, al comparar el entero con el valor de ord(char), podemos verificar si hay un evento de tecla presionada y romper el ciclo.
        if key == ord("q") or key == ord("Q"):
            break

    capture.release()


if __name__ == "__main__":
    main()
