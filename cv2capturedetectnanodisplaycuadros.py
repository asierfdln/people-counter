
# comando de botellas -> python3 cv2capturedetectnanodisplaycuadros.py --network=coco-bottle
# comando de personas -> python3 cv2capturedetectnanodisplaycuadros.py --network=multiped

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


# funcion para verificar que una deteccion se encuentra dentro de una zona de interes
def detection_in_area(counting_area, rect_of_detection):

	# centro de la deteccion
	x_of_detection = rect_of_detection[0] + (rect_of_detection[2] - rect_of_detection[0]) / 2
	y_of_detection = rect_of_detection[1] + (rect_of_detection[3] - rect_of_detection[1]) / 2

	if ((x_of_detection >= counting_area[0] and x_of_detection <= counting_area[2])
		and
	   (y_of_detection >= counting_area[1] and y_of_detection <= counting_area[3])):

		return True

	else:
		return False


# TODO separar esto en un solo bucle para evitar 1 deteccion n zonas, 2 deteccion n zonas...
# funcion para determinar si las coordenadas centrales de una deteccion entran dentro de una zona de interes
def zone_explorer(list_of_counting_areas, rect_of_detection):

	# numero del area en el que se situa la deteccion 
	counter_of_area_to_return = 1

	# miramos
	for counting_area in list_of_counting_areas:

		if detection_in_area(counting_area, rect_of_detection):

			return counter_of_area_to_return

		else:
			counter_of_area_to_return = counter_of_area_to_return + 1

	else:
		return 0 # no estamos en ninguna zona


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

	# el modelo a cargar por defecto, poner en consola "--network={algo}"
	# para sobreescribir a ssd-mobilenet-v2
	net = jetson.inference.detectNet('ssd-mobilenet-v2', sys.argv, 0.99)

	# camara de opencv
	cap = cv2.VideoCapture( \
		gstreamer_pipeline(
			capture_width=WIDTH,
			capture_height=HEIGHT,
			display_width=WIDTH,
			display_height=HEIGHT,
			flip_method=0
		),
		cv2.CAP_GSTREAMER
	)

	# TODO cap.capture, imshow y con raton/teclado definir zonas...

	# lista de areas [arribaizqx, arribaizqy, debajodchax, debajodchay, contador_detecciones_in] en las que contar peoples
	counting_areas = [
		[50, 50, 220, 220, 0],
		[600, 250, 750, 350, 0],
	]

	# constante a utilizar para el conteo de detecciones en las diferentes zonas
	NUM_OF_COUNTING_AREAS_plusone = len(counting_areas) + 1

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

				# zona del bucle para las detecciones 

				# TODO mmmmmmmmmh??
				# if contador_frames % FRAMES_DETECT == 0 or contador_frames == 0 or redo_detection:
				# if contador_frames % FRAMES_DETECT == 0 or redo_detection:
				if redo_detection:

					# reseteamos el flag de darle a la tecla R de "refresh"
					redo_detection = False

					# vaciamos la lista de trackers para empezar de cero
					trackers = [] # TODO RENDIMIENTO vaciar no parece como muy buena idea, en funcion de la
								  # de las posiciones de las detecciones se puede ver si quitar
								  # o no
								  # 
								  # TODO RENDIMIENTO medir tb tiempo de creacion de dos trackers y
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

						rectangle = [
							int(detection.Center[0] - detection.Width / 2),
							int(detection.Center[1] - detection.Height / 2),
							int(detection.Center[0] + detection.Width / 2),
							int(detection.Center[1] + detection.Height / 2)
						]

						# lista con (1) el tracker y (2) el numerillo de izq/dcha
						list_w_tracker = [] # TODO objetos nuestros definidos??...

						# creamos un tracker de dlib
						tracker = dlib.correlation_tracker()

						# le pasamos las esquinas del rectangulo
						rect_dlib = dlib.rectangle(
							rectangle[0],
							rectangle[1],
							rectangle[2],
							rectangle[3]
						)

						# empesamos el trackeo
						tracker.start_track(img_rgb, rect_dlib)

						# añadimos el tracker a la lista magica esta
						list_w_tracker.append(tracker)

						# asignamos al tracker una de las zonas de interes (o fuera, 0)
						number_of_zone = zone_explorer(counting_areas, rectangle)
						list_w_tracker.append(number_of_zone)

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
						upperLeftCornerX = int(pos.left())
						upperLeftCornerY = int(pos.top())

						# esquina inferior derecha
						lowerRightCornerX = int(pos.right())
						lowerRightCornerY = int(pos.bottom())

						# TODO mismo que list_w_tracker[0].get_position()??
						# QOL notationz
						rectangle_from_trackerpos = [upperLeftCornerX, upperLeftCornerY, lowerRightCornerX, lowerRightCornerY]

						for i in range(1, NUM_OF_COUNTING_AREAS_plusone):
							if detection_in_area(counting_areas[i-1], rectangle_from_trackerpos):
								if list_w_tracker[1] != i:
									list_w_tracker[1] = i
									counting_areas[i-1][4] = counting_areas[i-1][4] + 1
									break
								else:
									break
							else:
								pass
						else:
							list_w_tracker[1] = 0

						# pintamos el cuadradico de la deteccion
						cv2.rectangle(

							# imagen sobre la que pintar
							img,

							# coordenadas de la esquina superior izquierda
							(upperLeftCornerX, upperLeftCornerY),

							# coordenadas de la esquina inferior derecha
							(lowerRightCornerX, lowerRightCornerY),

							# color BGR
							(0, 255, 0),

							# grosor de linea
							2
						)

				# MUCHO TEXTO
				# cv2.putText(

				# 	# imagen sobre la que pintar
				# 	img,

				# 	# texto
				# 	f'Hacia la izquierda -> {contador_yendo_izquierda}',

				# 	# posicion del texto
				# 	(0, 30),

				# 	# fuente
				# 	cv2.FONT_HERSHEY_SIMPLEX,

				# 	# tamaño letra
				# 	1.25,

				# 	# color BGR
				# 	(0, 255, 0),

				# 	# grosor linea
				# 	1
				# )

				# pintamos los cuadradicos de conteo
				for cuadradico in counting_areas:
					cv2.rectangle(img, (cuadradico[0], cuadradico[1]), (cuadradico[2], cuadradico[3]), (0, 0, 255), 2)
					cv2.putText(
						img,
						f'Count: {cuadradico[4]}',
						(cuadradico[0], cuadradico[1] - 15),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.65,
						(0, 0, 255),
						2
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
				for cuadradico in counting_areas:
					cuadradico[4] = 0

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
