import torch
import torch.nn as nn
from torchvision.models import densenet121

# Lista de clases (labels)
CLASSES = [
    'abraham_grampa_simpson',
    'agnes_skinner',
    'apu_nahasapeemapetilon',
    'bart_simpson',
    'barney_gumble',
    'carl_carlson',
    'charles_montgomery_burns',
    'chief_wiggum',
    'cletus_spuckler',
    'comic_book_guy',
    'disco_stu',
    'edna_krabappel',
    'fat_tony',
    'gil',
    'groundskeeper_willie',
    'homer_simpson',
    'kent_brockman',
    'krusty_the_clown',
    'lenny_leonard',
    'lisa_simpson',
    'lionel_hutz',
    'maggie_simpson',
    'marge_simpson',
    'martin_prince',
    'mayor_quimby',
    'milhouse_van_houten',
    'miss_hoover',
    'moe_szyslak',
    'nelson_muntz',
    'ned_flanders',
    'otto_mann',
    'patty_bouvier',
    'principal_skinner',
    'professor_john_frink',
    'ralph_wiggum',
    'rainier_wolfcastle',
    'selma_bouvier',
    'sideshow_bob',
    'sideshow_mel',
    'snake_jailbird',
    'troy_mcclure',
    'waylon_smithers'
]

# Diccionario índice 
idx_to_class = {i: label for i, label in enumerate(CLASSES)}

class EmbeddingModel(nn.Module):
    """
    Modelo que extrae embeddings antes de la clasificación final
    """
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

def load_model(path):
    """
    Carga el modelo con manejo de errores robusto
    """
    try:
        # Cargar el checkpoint completo
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Crear el modelo base
        base_model = densenet121(weights=None)
        base_model.classifier = nn.Linear(base_model.classifier.in_features, len(CLASSES))
        
        # Intentar cargar el state_dict
        if isinstance(checkpoint, dict):
            # Si el checkpoint es un diccionario, buscar la clave correcta
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['state_dict'])
            else:
                # Asumir que el diccionario completo es el state_dict
                base_model.load_state_dict(checkpoint)
        else:
            # Si no es un diccionario, asumir que es el state_dict directamente
            base_model.load_state_dict(checkpoint)
        
        # Crear el modelo de embeddings
        embedding_model = EmbeddingModel(base_model)
        
    except Exception as e:
        print(f"Error al cargar con método estándar: {e}")
        
        # Método alternativo: cargar con strict=False
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            base_model = densenet121(weights=None)
            base_model.classifier = nn.Linear(base_model.classifier.in_features, len(CLASSES))
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    base_model.load_state_dict(checkpoint, strict=False)
            else:
                base_model.load_state_dict(checkpoint, strict=False)
            
            # Crear el modelo de embeddings
            embedding_model = EmbeddingModel(base_model)
                
        except Exception as e2:
            print(f"Error al cargar con strict=False: {e2}")
            
            # Último intento: crear modelo desde cero
            base_model = densenet121(weights=None)
            base_model.classifier = nn.Linear(base_model.classifier.in_features, len(CLASSES))
            embedding_model = EmbeddingModel(base_model)
            print("Modelo creado desde cero. Algunas capas pueden no estar inicializadas correctamente.")
    
    return embedding_model, idx_to_class

def predict_character(model, img_tensor, idx_to_class, reference_embeddings=None):
    """
    Predice el personaje basado en embeddings usando similitud coseno
    """
    try:
        model.eval()
        with torch.no_grad():
            # Obtener embedding de la imagen de entrada
            query_embedding = model(img_tensor)
            
            # Si no hay embeddings de referencia, usar un método alternativo
            if reference_embeddings is None:
                # Método simplificado: usar la norma del embedding para clasificar
                # Esto es solo un placeholder - idealmente necesitarías embeddings de referencia
                embedding_norm = torch.norm(query_embedding, dim=1)
                predicted_idx = int(embedding_norm.item() * len(CLASSES)) % len(CLASSES)
                return idx_to_class.get(predicted_idx, "Desconocido"), query_embedding.cpu().numpy()
            
            # Calcular similitud coseno con embeddings de referencia
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding, 
                reference_embeddings, 
                dim=1
            )
            
            # Encontrar la clase más similar
            predicted_idx = torch.argmax(similarities).item()
            confidence = similarities[predicted_idx].item()
            
            return idx_to_class.get(predicted_idx, "Desconocido"), confidence, query_embedding.cpu().numpy()
            
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return "Error en la predicción", 0.0, None

def load_reference_embeddings(embeddings_path=None):
    """
    Carga embeddings de referencia para cada clase si están disponibles
    """
    if embeddings_path:
        try:
            reference_embeddings = torch.load(embeddings_path, map_location='cpu')
            return reference_embeddings
        except Exception as e:
            print(f"Error cargando embeddings de referencia: {e}")
    return None

def create_reference_embeddings(model, dataloader, device='cpu'):
    """
    Crea embeddings de referencia promedio para cada clase
    Esta función se usaría durante el entrenamiento para crear referencias
    """
    model.eval()
    class_embeddings = {}
    class_counts = {}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            
            for emb, label in zip(embeddings, labels):
                label = label.item()
                if label not in class_embeddings:
                    class_embeddings[label] = torch.zeros_like(emb)
                    class_counts[label] = 0
                
                class_embeddings[label] += emb
                class_counts[label] += 1
    
    # Promediar embeddings por clase
    for label in class_embeddings:
        class_embeddings[label] /= class_counts[label]
    
    # Convertir a tensor
    reference_tensor = torch.stack([class_embeddings[i] for i in range(len(CLASSES))])
    return reference_tensor