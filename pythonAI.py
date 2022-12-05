import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
width_height = 320

# Umbral de seguridad (Se pueden variar para ver el comportamiento)
confidence_threshold = 0.5

# Umbral NMS: a mayor umbral, más "boxes" serán dibujadas en la imagen
nms_threshold = 0.3

classes_file = "coco.names"
class_names = []

# Se imprimen los objetos que puede detectar el modelo
with open(classes_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
print(class_names)

# Cargamos el modelo (Coco names)
# Archivos de configuración y pesos del modelo
model_configuration = "yolov3-tiny.cfg"
model_weights = "yolov3-tiny.weights"

# Se crea la red neuronal (con opencv y usando la CPU)
net = cv.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Se buscan los objetos a detectar
def SearchObjects(outputs, img):
    input_image_height, input_image_width, input_image_channels = img.shape
    box = []
    class_ids = []
    confidence_values = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence_net = scores[class_id]
            # Se compara si la red posee una seguridad mayor al 50% de detectar un objeto
            if confidence_net > confidence_threshold:
                object_width, object_height = int(detection[2] * input_image_width), int(detection[3] * input_image_height)
                coordinate_x, coordinate_y = int((detection[0] * input_image_width) - object_width / 2), int((detection[1] * input_image_height) - object_height / 2)
                box.append([coordinate_x, coordinate_y, object_width, object_height])
                class_ids.append(class_id)
                confidence_values.append(float(confidence_net))

    # Se guardan los indices de las "boxes" (rectángulos en los que se encuentran los objetos) apropiadas mediante el método NMSBoxes
    indices = cv.dnn.NMSBoxes(box, confidence_values, confidence_threshold, nms_threshold)

    # Se dibuja un rectángulo en el objeto que se ha detectado
    # Se escribe el nombre del objeto detectado
    for i in indices:
        i = i[0]
        result_box = box[i]
        box_coordinate_x, box_coordinate_y, object_width, object_height = result_box[0], result_box[1], result_box[2], result_box[3]
        cv.rectangle(img, (box_coordinate_x, box_coordinate_y), (box_coordinate_x + object_width, box_coordinate_y + object_height), (0, 0, 255), 2)
        cv.putText(img, f'{class_names[class_ids[i]].upper()} {int(confidence_values[i] * 100)}%',
                   (box_coordinate_x, box_coordinate_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

while True:
    success, img = cap.read()

    # Se convierte la imagen principal (input) en otra con distinto formato (blob)
    blob = cv.dnn.blobFromImage(img, 1 / 255, (width_height, width_height), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Se obtienen las capas de salida de la red
    layers_names = net.getLayerNames()
    output_layers_names = [(layers_names[i[0] - 1]) for i in net.getUnconnectedOutLayers()]

    # Salidas de las capas finales
    outputs = net.forward(output_layers_names)

    SearchObjects(outputs, img)

    cv.imshow('Image', img)

    if cv.waitKey(1) & 0XFF ==ord("q"):
        break