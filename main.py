import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.strip().split(',')[:-1])) for line in lines[9:]]
        labels = [line.strip().split(',')[-1].strip() for line in lines[9:]]
    return np.array(data), np.array(labels)

def kmeans(X, k, max_iterations=100):
    output = ""
    
    # Inicialización aleatoria de centroides
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]
    
    for iteration in range(max_iterations):
        # Calcular distancias entre puntos y centroides
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Encontrar la etiqueta de clúster más cercana para cada punto
        labels = np.argmin(distances, axis=1)
        
        # Actualizar posiciones de centroides
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Construir tabla de distancias en cada iteración
        output += f"Iteracion: {iteration + 1}\n"
        output += "Punto \t|\t X \t,\t Y \t,\t Z \t,\t W \t|\t Distancia C1 \t|\t Distancia C2 \t|\t Distancia C3 \t|\t Distancia mínima\n"
        output += "--------------------------------------------------------------------------------------------------------------------\n"
        for i, (x, y, z, w) in enumerate(X):
            dist_to_centroids = distances[i]
            min_distance = dist_to_centroids[labels[i]]
            output += f"{i+1:5} \t|\t {x:.2f} \t,\t {y:.2f} \t,\t {z:.2f} \t,\t {w:.2f} \t|\t {dist_to_centroids[0]:.4f} \t|\t {dist_to_centroids[1]:.4f} \t|\t {dist_to_centroids[2]:.4f} \t| {min_distance:.4f}\n"
        output += "--------------------------------------------------------------------------------------------------------------------\n"
        
        # Mostrar posiciones de los centroides en cada iteración
        output += "Posiciones de los centroides:\n"
        for i, centroid in enumerate(centroids):
            output += f"C{i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}, {centroid[3]:.2f})\n"
        output += "--------------------------------------------------------------------------------------------------------------------\n"
        
        # Verificar si los centroides se han estabilizado
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Guardar la salida en un archivo de texto
    with open("output.txt", "w") as file:
        file.write(output)
    
    return centroids, labels

def kmeans_predict(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

# Lista para almacenar los resultados de precisión de cada fold
precisions_5 = []
precisions_10 = []

# Realizar validación cruzada para iris-fold-5
fold_5_dir = os.path.join('iris-5-fold')
fold_5_files_train = sorted([file for file in os.listdir(fold_5_dir) if file.startswith('iris-5-') and file.endswith('tra.dat')])
fold_5_files_test = sorted([file for file in os.listdir(fold_5_dir) if file.startswith('iris-5-') and file.endswith('tst.dat')])

label_encoder = LabelEncoder()

for train_file, test_file in zip(fold_5_files_train, fold_5_files_test):
    # Cargar datos de entrenamiento y prueba
    train_data, train_labels = load_data(os.path.join(fold_5_dir, train_file))
    test_data, test_labels = load_data(os.path.join(fold_5_dir, test_file))

    # Convertir las etiquetas a valores numéricos
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Verificar si hay datos para calcular la precisión
    if len(np.unique(test_labels)) == 3:
        # Ejecutar el algoritmo K-means
        k = 3
        centroids, labels = kmeans(train_data, k)

        # Calcular la intersección de las etiquetas
        test_predictions = kmeans_predict(test_data, centroids)

        # Calcular la precisión en el conjunto de prueba
        correct = np.sum(test_predictions == test_labels)
        precision = correct / len(test_labels)
        precisions_5.append(precision)

        # Mostrar el nivel de precisión en el fold actual
        print(f"Precision en fold 5: {precision:.4f}")

# Calcular media y desviación estándar para iris-fold-5
mean_5 = np.mean(precisions_5)
std_5 = np.std(precisions_5)

# Realizar validación cruzada para iris-fold-10
fold_10_dir = os.path.join('iris-10-fold')
fold_10_files_train = sorted([file for file in os.listdir(fold_10_dir) if file.startswith('iris-10-') and file.endswith('tra.dat')])
fold_10_files_test = sorted([file for file in os.listdir(fold_10_dir) if file.startswith('iris-10-') and file.endswith('tst.dat')])

for train_file, test_file in zip(fold_10_files_train, fold_10_files_test):
    # Cargar datos de entrenamiento y prueba
    train_data, train_labels = load_data(os.path.join(fold_10_dir, train_file))
    test_data, test_labels = load_data(os.path.join(fold_10_dir, test_file))

    # Convertir las etiquetas a valores numéricos
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Verificar si hay datos para calcular la precisión
    if len(np.unique(test_labels)) == 3:
        # Ejecutar el algoritmo K-means
        k = 3
        centroids, labels = kmeans(train_data, k)

        test_predictions = kmeans_predict(test_data, centroids)

        # Calcular la precisión en el conjunto de prueba
        correct = np.sum(test_predictions == test_labels)
        precision = correct / len(test_labels)
        precisions_10.append(precision)

        # Mostrar el nivel de precisión en el fold actual
        print(f"Precision en fold 10: {precision:.4f}")

# Calcular media y desviación estándar para iris-fold-10
mean_10 = np.mean(precisions_10)
std_10 = np.std(precisions_10)

# Mostrar los resultados en una tabla
print("Resultados:")
print("------------------------------------------------")
print("Fold 5  | Fold 10")
print("------------------------------------------------")
print(f"Mean    | {mean_5:.4f} | {mean_10:.4f}")
print(f"Std     | {std_5:.4f} | {std_10:.4f}")
print("------------------------------------------------")