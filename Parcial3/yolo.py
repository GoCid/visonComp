import numpy as np
import argparse
import time
import cv2
import os

#EJEMPLO python3 yolo.py --image images/baggage_claim.jpg

#ARGUMENTOS QUE USA EL SCRIPT
#IMAGE: imagen a procesar
#CONFIDENCE: confianza minima para considerar un objeto detectado
#THRESHOLD: umbral para la supresion de no maximos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
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

#CARGA DE LA IMAGEN
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
#DETECTAR LAS CAPAS DE SALIDA DEL MODELO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
#CONVERTIR LA IMAGEN A UN BLOB Y PASARLA POR EL MODELO, DANDONOS LAS CAJAS Y PROBABILIDADES
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
#start = time.time()
layerOutputs = net.forward(ln)
#end = time.time()
#TIEMPO DE EJECUCION
#print("YOLO {:.6f} seconds".format(end - start))

#LISTAS PARA ALMACENAR LAS CAJAS, PROBABILIDADES Y CLASES
boxes = []
confidences = []
classIDs = []

#RECORRER LAS CAPAS DE SALIDA
for output in layerOutputs:
    #RECORRER LAS DETECCIONES
	for detection in output:
        #EXTRAER EL ID DE LA CLASE Y LA PROBABILIDAD DE LA DETECCION DEL OBJETO
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
        #FILTRAR LAS DETECCIONES CON PROBABILIDAD MENOR A LA PROBABILIDAD MINIMA
		if confidence > args["confidence"]:
            #CALCULAR LAS COORDENADAS DE LA CAJA QUE CONTIENE AL OBJETO
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top
			# and left corner of the bounding box
            #USANDO LAS CORDENADAS DEL CENTRO DE LA CAJA, CALCULAR LAS CORDENADAS DE LA ESQUINA SUPERIOR IZQUIERDA
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
            #ACTUALIZAR LA LISTA DE LAS CAJAS, PROBABILIDADES Y CLASES
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

#SUPRESION DE NO MAXIMOS PARA ELIMINAR CAJAS SOBREPUESTAS
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

#SI AL MENOS EXISTE UNA DETECCION
if len(idxs) > 0:
    #RECORRER LOS INDICES DE LAS CAJAS QUE QUEDAN
	for i in idxs.flatten():
        #EXTRAER LAS CORDENADAS DE LA CAJA QUE CONTIENE AL OBJETO
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
        #DIBUJAR LA CAJA Y LA ETIQUETA DE LA CLASE
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			.7, color, 2)
#MOSTRAR LA IMAGEN
cv2.imshow("Imagen procesada", image)
cv2.waitKey(0)
