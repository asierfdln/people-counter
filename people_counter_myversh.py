import numpy as np
import argparse
import cv2


# no sé de dónde hemos sacado esto, pero lo utiliza todo peter para sacar el recurso famoso este del gstreamer de la 
# jetson (hay que decir tb que me suena haberlo visto en algun foro de estos en los que el bueno de dusty-nv contestaba 
# cosas pero no estoy muy seguro...)

# Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2):
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

    # TODO debería haber moar resols?? tipo de salida de ventana y tal... lo cual me lleva a pensar tb que el 
    # get_jetson_gstreamer_source() debería cambiarse según parámetros de consola como de toda la vida de dios se ha hecho...
    parser.add_argument("-x", "--width", type=int, default=1280, help="capture width (default 1280)")
    parser.add_argument("-y", "--height", type=int, default=720, help="capture height (default 720)")
    args = vars(parser.parse_args())

    # estos dos parámetros de W y H deberían ir en concordancia con lo puesto en el get_jetson_gstreamer_source() de arriba
    W = args["width"] # anchura de (de qué?? de lo que pilla la imagen, tipo la resolucion de la imagen o la de la ventana que vamos a mostrar)
    H = args["height"] # altura de (~)

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
    capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)

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

        # esperamos a una 'q' de teclado para salir del bisho este
        key = cv2.waitKey(1) & 0xFF # La función waitKey devuelve - 1 cuando no se realiza ninguna entrada. Tan pronto como ocurre el evento es decir. Se presiona un botón devuelve un entero de 32 bits.
                                    #El 0xFF en este escenario representa binario 11111111 a 8 bit binary, ya que solo necesitamos 8 bits para representar un carácter que AND Y waitKey(0) a 0xFF. Como resultado, se obtiene un número entero por debajo de 255.
                                    #ord(char) devuelve el valor ASCII del carácter que nuevamente sería máximo 255.
                                    #Por lo tanto, al comparar el entero con el valor de ord(char), podemos verificar si hay un evento de tecla presionada y romper el ciclo.
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
