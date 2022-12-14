import numpy as np
import argparse
import imutils
import time
import cv2
import os
import imutils

#EJEMPLO python3 yolovid.py --input videos/jefvid.mp4 --output output/jefvid.avi
#ARGUMENTOS QUE USA EL SCRIPT
#INPUT: video a procesar
#OUTPUT: video de salida

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

#CARGA DE LAS ETIQUETAS DE CLASES
LABELS = open("yolo-coco/coco.names").read().strip().split("\n")
#INICIALIZAR COLORES PARA DIBUJAR BOUNDING BOX
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

#CARGA DE LOS PESOS Y CONFIGURACION DEL MODELO, USADO: YOLOV3 80 CLASES
#net = cv2.dnn.readNet('yolo-coco/yolov5.onnx')
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")

#DETECTAR LAS CAPAS DE SALIDA DEL MODELO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()] ##errorrr


#INICIALIZAR EL VIDEO Y DIMENSIONES DEL FRAME
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
#INTENTAR DETERMINAR EL NUMERO TOTAL DE FRAMES EN EL VIDEO
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print(" {} frames en total".format(total))
#EXCEPCION SI NO SE PUEDE DETERMINAR EL NUMERO DE FRAMES
except:
	print("error en determinar los frames")
	total = -1

#HACER UN LOOP SOBRE LOS FRAMES DEL VIDEO
while True:
	#LEER EL SIGUIENTE FRAME DEL VIDEO
	(grabbed, frame) = vs.read()
	#SI NO SE PUDO LEER EL FRAME, SALIR DEL LOOP
	if not grabbed:
		break
	#SI LAS DIMENSIONES DEL FRAME SON VACIAS, OBTENERLAS
	if W is None or H is None:
		(H, W) = frame.shape[:2]

#CONSTRUIR UN BLOB DEL FRAME Y LUEGO HACER UNA PASADA HACIA ADELANTE DEL DETECTOR DE OBJETOS YOLO
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	#DETECTAR BOUNDING BOXES, CONFIDENCIAS Y CLASES DE LOS OBJETOS
	boxes = []
	confidences = []
	classIDs = []

#HACER UN LOOP SOBRE LAS SALIDAS DE LAS CAPAS
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			#DIBUJAR UNA BOUNDING BOX Y UNA ETIQUETA EN EL FRAME
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				#VERIFICAR SI EL VIDEO WRITER ES NULO
	if writer is None:
		#INICIALIZAR EL VIDEO WRITER
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		if total > 0:
			elap = (end - start)
			print("Tiempo estimado: {:.4f} seconds".format(
				elap * total))
	#Escribir el frame de salida en el disco
	writer.write(frame)
print("FINALIZANDO")
writer.release()
vs.release()
                

