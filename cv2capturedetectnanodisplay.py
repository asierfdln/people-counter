
# comando de botellas -> python3 cv2capturedetectnanodisplay.py --network=coco-bottle
# comando de personas -> python3 cv2capturedetectnanodisplay.py --network=multiped

import cv2
import dlib
import numpy as np
import sys

# ojo con poner cv2 despues de estos imports de jetson, a veces petan cosas...
import jetson.inference
import jetson.utils


# funcion para sacar la cámara
def gstreamer_pipeline (capture_width=800, capture_height=600, display_width=800, display_height=600, framerate=30, flip_method=2):
	return (
		'nvarguscamerasrc ! '
		'video/x-raw(memory:NVMM), '
		'width=(int)%d, height=(int)%d, '
		'format=(string)NV12, framerate=(fraction)%d/1 ! '
		'nvvidconv flip-method=%d ! '
		'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
		'videoconvert ! '
		'video/x-raw, format=(string)BGR ! appsink' \
		% (capture_width, capture_height, framerate, flip_method, display_width, display_height)
	)


def main():

	# dimensiones de la ventana Y de las imagenes a capturar por la
	# camara (TODO con argparse plz...)
	WIDTH = 800
	HEIGHT = 600

	# lista de trackers de dlib
	trackers= []

	# flag para pasar otra vez el frame por net.Detect() y detectar objetos
	redo_detection = False

	# numero de frames tras los cuales hacer un refresh de detecciones
	contador_frames = 0
	FRAMES_DETECT = 50

	# contadores de padonde van las cosis
	contador_yendo_derecha = 0
	contador_yendo_izquierda = 0

	# el modelo a cargar por defecto, poner en consola "--network={algo}"
	# para sobreescribir a ssd-mobilenet-v2
	net = jetson.inference.detectNet('ssd-mobilenet-v2', sys.argv, 0.99)

	# camara de opencv
	cap = cv2.VideoCapture( \
		gstreamer_pipeline( \
			capture_width=WIDTH, \
			capture_height=HEIGHT, \
			display_width=WIDTH, \
			display_height=HEIGHT, \
			flip_method=2 \
		), \
		cv2.CAP_GSTREAMER \
	)

	if cap.isOpened():

		# bucle principal
		while True:

			# captura imagen
			ret, img = cap.read()

			# este ret es una variable que se pone a "True"
			# si se ha pillao la imagen bien
			if ret:

				# vamos sumando el contador de frames capturados por la camara
				contador_frames += 1

				# dlib luego quiere las imagenes en rgb
				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				# zona del bucle para las detecciones (TODO multiprocessing
				# para if y else a la vez???, pensar un poco...)

				if contador_frames % FRAMES_DETECT == 0 or redo_detection:

					# reseteamos el flag de darle a la tecla R de "refresh"
					redo_detection = False

					# vaciamos la lista de trackers para empezar de cero
					trackers = [] # TODO vaciar no parece como muy buena idea, en funcion de la
								  # de las posiciones de las detecciones se puede ver si quitar
								  # o no
								  # 
								  # TODO medir tb tiempo de creacion de dos trackers y
								  # la comparacion...

					# convierte la imagen en formato opencv (BGR unit8) a RGBA float32
					img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)

					# pasa la imagen RGBA float32 a memoria CUDA y haz las detecciones
					img_cuda = jetson.utils.cudaFromNumpy(img_rgba)
					detections = net.Detect(img_cuda, 800, 600, 'none')

					# cogemos la informacion de todas las detecciones y calculamos las
					# esquinas superior izquierda y la inferior derecha porque dlib 
					# y sus trackers son asi de especiales
					for detection in detections:

						# cogemos las coordenadas de los rectangulos de las detecciones y
						# se las pasamos a un tracker por cada deteccion realizada, luego
						# guardamos un numerillo junto con el tracker para ver en que mitad de
						# la imagen esta el objeto y asi contar luego las cosas de un lado
						# palotro (-1 izquierda, 1 derecha)

						rectangle = [ \
							int(detection.Center[0] - detection.Width / 2), \
							int(detection.Center[1] - detection.Height / 2), \
							int(detection.Center[0] + detection.Width / 2), \
							int(detection.Center[1] + detection.Height / 2) \
						]

						# lista con (1) el tracker y (2) el numerillo de izq/dcha
						list_w_tracker = [] # TODO objetos nuestros definidos??...

						# creamos un tracker de dlib
						tracker = dlib.correlation_tracker()

						# le pasamos las esquinas del rectangulo
						rect_dlib = dlib.rectangle( \
							rectangle[0], \
							rectangle[1], \
							rectangle[2], \
							rectangle[3] \
						)

						# empesamos el trackeo
						tracker.start_track(img_rgb, rect_dlib)

						# añadimos el tracker a la lista magica esta
						list_w_tracker.append(tracker)

						# miramos si la deteccion esta en la izq
						if (rectangle[0] + (rectangle[2] - rectangle[0]) / 2) <= (WIDTH / 2):

							list_w_tracker.append(-1)

						# miramos si la deteccion esta en la dcha
						elif (rectangle[0] + (rectangle[2] - rectangle[0]) / 2) >= (WIDTH / 2):

							list_w_tracker.append(1)

						# lista con las listas de los trackers
						trackers.append(list_w_tracker)

				# zona de trackeo, actualizacion de los trackers de dlib
				else:

					# pillamos cada una de las magicolistas con (1) el tracker y
					# (2) el numerillo izq/dcha
					for list_w_tracker in trackers:

						# para updatear un tracker hay que pasarle la imagen en formato rgb
						list_w_tracker[0].update(img_rgb)
						# sacamos el objeto de posicion que devuelve el tracker
						pos = list_w_tracker[0].get_position()

						# esquina superior izquierda
						startX = int(pos.left())
						startY = int(pos.top())

						# esquina inferior derecha
						endX = int(pos.right())
						endY = int(pos.bottom())

						# miramos si (1) el objeto se pasa de la mitad y (2) si
						# el numerillo de antes indicaba que estaba en la otra mitad;
						# los casos extremos de justo se vuelve a detectar algo cuando ya
						# se ha pasado de la mitadpues nos jodemos...
						if (startX + (endX - startX) / 2) <= (WIDTH / 2) and list_w_tracker[1] == 1:

							# el objeto se ha movido para la izquierda, cambiamos a -1
							list_w_tracker[1] = -1

							# actualizamos contadores
							contador_yendo_izquierda = contador_yendo_izquierda + 1
							if contador_yendo_derecha > 0:
								contador_yendo_derecha = contador_yendo_derecha - 1

						elif (startX + (endX - startX) / 2) >= (WIDTH / 2) and list_w_tracker[1] == -1:

							# el objeto se ha movido para la izquierda, cambiamos a -1
							list_w_tracker[1] = 1

							# actualizamos contadores
							contador_yendo_derecha = contador_yendo_derecha + 1
							if contador_yendo_izquierda > 0:
								contador_yendo_izquierda = contador_yendo_izquierda - 1

						# pintamos el cuadradico
						cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

				# MUCHO TEXTO
				cv2.putText( \

					# imagen sobre la que pintar
					img, \

					# texto
					f'Hacia la izquierda -> {contador_yendo_izquierda}', \

					# posicion del texto
					(0, 30), \

					# fuente
					cv2.FONT_HERSHEY_SIMPLEX, \

					# tamaño letra
					1.25, \

					# color
					(0, 255, 0), \

					# grosor linea
					1 \
				)

				# MUCHO TEXTO
				cv2.putText( \
					img, \
					f'Hacia la derecha -> {contador_yendo_derecha}', \
					(0, 65), \
					cv2.FONT_HERSHEY_SIMPLEX, \
					1.25, \
					(0, 255, 0), \
					1 \
				)

				# sacamos imagen a la ventana
				cv2.imshow('sth...', img)

			# pillamos pulsacion de tecla
			keyCode = cv2.waitKey(1) & 0xFF

			# para salir del bucle pulsar 'q' o 'esc'
			if keyCode == 27 or keyCode == ord('q'): # TODO salir con X en ventana no funciona??
				break

			# para resetear los contadores pulsar 'a' (de "again"...)
			elif keyCode == ord('a'):
				contador_yendo_izquierda = 0
				contador_yendo_derecha = 0

			# para resetear detecciones de forma manual pulsar 'r' (de "refresh"...)
			elif keyCode == ord('r'):
				redo_detection = True

		# cerramos todo el cotarro
		cap.release()
		cv2.destroyAllWindows()

	else:

		print('Unable to open camera')


if __name__ == '__main__':
	main()